"""Hardware-friendly approximations from eMamba (Section 4 of the paper).

These are the building blocks the accelerator implements in fixed hardware:
  - Range normalization replaces LayerNorm (Eq. 2)
  - Piecewise-linear SiLU with 17 segments in [-7, 7]
  - Piecewise-linear exponential with 11 segments in [-4, 1]
  - ReLU substituted for Softplus in the SSM discretization
"""

import numpy as np


def range_normalization(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    rng = (centered.max(axis=-1, keepdims=True)
           - centered.min(axis=-1, keepdims=True))
    return gamma * centered / (rng + eps) + beta


def relu(x):
    return np.maximum(x, 0)


def _piecewise_linear(f, x_lo, x_hi, n_segments,
                      below=None, above=None):
    breakpoints = np.linspace(x_lo, x_hi, n_segments + 1, dtype=np.float32)
    values = f(breakpoints).astype(np.float32)
    slopes = np.diff(values) / np.diff(breakpoints)
    intercepts = values[:-1] - slopes * breakpoints[:-1]

    def approx(x):
        x = np.asarray(x, dtype=np.float32)
        out = np.empty_like(x)
        m_lo = x < x_lo
        m_hi = x > x_hi
        m_in = ~(m_lo | m_hi)

        if below is None:
            out[m_lo] = f(x[m_lo])
        elif callable(below):
            out[m_lo] = below(x[m_lo])
        else:
            out[m_lo] = below

        if above is None:
            out[m_hi] = f(x[m_hi])
        elif callable(above):
            out[m_hi] = above(x[m_hi])
        else:
            out[m_hi] = above

        x_in = x[m_in]
        idx = np.clip(np.searchsorted(breakpoints, x_in, side='right') - 1,
                      0, n_segments - 1)
        out[m_in] = slopes[idx] * x_in + intercepts[idx]
        return out
    return approx


def _silu(x):
    return x / (1.0 + np.exp(-x))


# 17 segments in [-7, 7]; below set to 0, above set to identity (linear region)
silu_pw = _piecewise_linear(
    _silu, -7.0, 7.0, n_segments=17,
    below=0.0,
    above=lambda x: x,
)

# 11 segments in [-4, 1]; below clamped to 0, above clamped to e^1
exp_pw = _piecewise_linear(
    np.exp, -4.0, 1.0, n_segments=11,
    below=0.0,
    above=np.float32(np.e),
)
