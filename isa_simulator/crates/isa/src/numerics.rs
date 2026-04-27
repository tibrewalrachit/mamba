//! Hardware-friendly numeric primitives — port of `mamba/approximations.py`.
//!
//! Spec §0.9 makes these primitives normative: any conformant simulator MUST
//! produce bit-identical FP32 outputs against the Python reference. The LUT
//! construction algorithm (linspace endpoints inclusive, slope/intercept via
//! finite differences, `searchsorted(side='right') - 1` lookup) is reproduced
//! here exactly.

/// `relu(x) = max(x, 0)`. NaN handling matches `f32::max` (`max(NaN, 0) = 0`).
#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Range normalization (spec §0.9.2 / `approximations.py:range_normalization`).
///
/// `dst[i] = γ[i] · (x[i] − μ) / (max(centered) − min(centered) + ε) + β[i]`
///
/// Mean and min/max are computed in f32 with left-to-right summation
/// (no Kahan compensation), to match NumPy.
pub fn range_normalization(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), gamma.len());
    assert_eq!(x.len(), beta.len());
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sum = 0.0_f32;
    for &v in x {
        sum += v;
    }
    let mu = sum / (n as f32);
    let mut max_c = f32::NEG_INFINITY;
    let mut min_c = f32::INFINITY;
    for &v in x {
        let c = v - mu;
        if c > max_c {
            max_c = c;
        }
        if c < min_c {
            min_c = c;
        }
    }
    let denom = (max_c - min_c) + eps;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(gamma[i] * (x[i] - mu) / denom + beta[i]);
    }
    out
}

// ─── Piecewise-linear LUT (spec §0.9.3) ─────────────────────────────────

/// Saturation rule applied outside the PWL's `[lo, hi]` domain.
#[derive(Copy, Clone, Debug)]
pub enum Saturation {
    /// Return a fixed constant (e.g., 0 for silu/exp below; e for exp above).
    Const(f32),
    /// Return the input unchanged (silu's linear region above `hi = 7`).
    Identity,
}

impl Saturation {
    fn eval(self, x: f32) -> f32 {
        match self {
            Self::Const(c) => c,
            Self::Identity => x,
        }
    }
}

/// PWL approximation built from a target function `f` over `[lo, hi]` divided
/// into `n_segments` evenly spaced segments. See spec §0.9.3 for the
/// construction algorithm — this implementation matches it exactly.
pub struct PiecewiseLinear {
    lo: f32,
    hi: f32,
    breakpoints: Vec<f32>,
    slopes: Vec<f32>,
    intercepts: Vec<f32>,
    below: Saturation,
    above: Saturation,
}

impl PiecewiseLinear {
    pub fn new<F: Fn(f32) -> f32>(
        f: F,
        lo: f32,
        hi: f32,
        n_segments: usize,
        below: Saturation,
        above: Saturation,
    ) -> Self {
        assert!(n_segments >= 1);
        let n = n_segments;
        // Mirror np.linspace(lo, hi, n+1, dtype=float32):
        //   step = (hi - lo) / n
        //   bp[i] = lo + step * i, with endpoints forced to lo / hi exactly.
        let step = (hi - lo) / (n as f32);
        let mut breakpoints = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let bp = if i == 0 {
                lo
            } else if i == n {
                hi
            } else {
                lo + step * (i as f32)
            };
            breakpoints.push(bp);
        }
        let mut values = Vec::with_capacity(n + 1);
        for &bp in &breakpoints {
            values.push(f(bp));
        }
        let mut slopes = Vec::with_capacity(n);
        let mut intercepts = Vec::with_capacity(n);
        for i in 0..n {
            let s = (values[i + 1] - values[i]) / (breakpoints[i + 1] - breakpoints[i]);
            slopes.push(s);
            intercepts.push(values[i] - s * breakpoints[i]);
        }
        Self {
            lo,
            hi,
            breakpoints,
            slopes,
            intercepts,
            below,
            above,
        }
    }

    pub fn n_segments(&self) -> usize {
        self.slopes.len()
    }

    pub fn breakpoints(&self) -> &[f32] {
        &self.breakpoints
    }

    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    pub fn intercepts(&self) -> &[f32] {
        &self.intercepts
    }

    /// Apply the PWL to a single value.
    ///
    /// Lookup matches NumPy `searchsorted(side='right') - 1` via Rust's
    /// `partition_point(.., |bp| bp <= x)` (spec §0.9.3).
    pub fn apply(&self, x: f32) -> f32 {
        if x < self.lo {
            return self.below.eval(x);
        }
        if x > self.hi {
            return self.above.eval(x);
        }
        let n = self.n_segments();
        // partition_point with predicate `bp <= x` returns the count of bps
        // that are <= x. Subtract 1 and clamp to [0, n-1] to handle the right
        // endpoint x == hi (which lands on bp[n], outside the slopes array).
        let raw = self.breakpoints.partition_point(|&bp| bp <= x);
        let idx = raw.saturating_sub(1).min(n - 1);
        self.slopes[idx] * x + self.intercepts[idx]
    }

    /// Apply the PWL element-wise. Mutates `out`; `xs` and `out` must be
    /// the same length.
    pub fn apply_slice(&self, xs: &[f32], out: &mut [f32]) {
        assert_eq!(xs.len(), out.len());
        for (i, &x) in xs.iter().enumerate() {
            out[i] = self.apply(x);
        }
    }
}

// ─── Pre-built PWLs for SiLU and Exp (spec §0.9.4 / §0.9.5) ─────────────

/// PWL SiLU: 17 segments in [-7, 7]; below = 0, above = identity.
pub fn silu_pw() -> PiecewiseLinear {
    PiecewiseLinear::new(
        |x| x / (1.0 + (-x).exp()),
        -7.0,
        7.0,
        17,
        Saturation::Const(0.0),
        Saturation::Identity,
    )
}

/// PWL Exp: 11 segments in [-4, 1]; below = 0, above = e.
pub fn exp_pw() -> PiecewiseLinear {
    PiecewiseLinear::new(
        |x| x.exp(),
        -4.0,
        1.0,
        11,
        Saturation::Const(0.0),
        Saturation::Const(std::f32::consts::E),
    )
}
