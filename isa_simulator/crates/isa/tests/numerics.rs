//! TDD ladder for numerics — `isa_spec.md` v0.2 §0.9 (and the source-of-truth
//! `mamba/approximations.py`).
//!
//! Strict bit-equality against the Python reference is verified later with
//! captured `.npy` fixtures (Phase 9). These tests pin the *algorithm* and the
//! invariants that the spec calls out directly.

use emamba_isa::numerics::{
    exp_pw, range_normalization, relu, silu_pw, PiecewiseLinear, Saturation,
};

// ─── relu ────────────────────────────────────────────────────────────────

#[test]
fn test01_relu_basics() {
    assert_eq!(relu(1.5_f32).to_bits(), 1.5_f32.to_bits());
    assert_eq!(relu(-0.5_f32), 0.0_f32);
    assert_eq!(relu(0.0_f32), 0.0_f32);
    // Spec §0.9.1: NaN handling matches f32::max — max(NaN, 0) = 0.
    assert_eq!(relu(f32::NAN), 0.0_f32);
}

// ─── range_normalization ─────────────────────────────────────────────────

#[test]
fn test02_range_normalization_zero_centered_unit_range() {
    // x = [-1, 0, 1], gamma = [1,1,1], beta = [0,0,0]
    //   mean = 0, centered = x, range = 2, eps = 1e-5
    //   out[i] = 1.0 * x[i] / (2 + 1e-5) + 0.0
    let x = [-1.0_f32, 0.0, 1.0];
    let g = [1.0_f32; 3];
    let b = [0.0_f32; 3];
    let out = range_normalization(&x, &g, &b, 1e-5);
    let denom = 2.0_f32 + 1e-5;
    let expect = [-1.0 / denom, 0.0 / denom, 1.0 / denom];
    for (a, e) in out.iter().zip(expect.iter()) {
        assert_eq!(a.to_bits(), e.to_bits(), "got {a}, want {e}");
    }
}

#[test]
fn test03_range_normalization_applies_gamma_beta() {
    let x = [0.0_f32, 4.0];
    let g = [3.0_f32, 3.0];
    let b = [0.5_f32, 0.5];
    let out = range_normalization(&x, &g, &b, 1e-5);
    // mean = 2, centered = [-2, 2], range = 4
    // out = gamma * centered / (4 + 1e-5) + beta
    let denom = 4.0_f32 + 1e-5;
    assert_eq!(out[0].to_bits(), (3.0 * (-2.0_f32) / denom + 0.5).to_bits());
    assert_eq!(out[1].to_bits(), (3.0 * 2.0_f32 / denom + 0.5).to_bits());
}

// ─── PiecewiseLinear: structural invariants from spec §0.9.3 ────────────

#[test]
fn test04_pwl_has_exact_segment_count() {
    let pwl = PiecewiseLinear::new(
        |x: f32| x,
        -1.0,
        1.0,
        4,
        Saturation::Const(0.0),
        Saturation::Const(0.0),
    );
    assert_eq!(pwl.n_segments(), 4);
    assert_eq!(pwl.breakpoints().len(), 5);
    assert_eq!(pwl.slopes().len(), 4);
    assert_eq!(pwl.intercepts().len(), 4);
}

#[test]
fn test05_pwl_breakpoints_are_endpoints_inclusive() {
    let pwl = PiecewiseLinear::new(
        |x| x,
        -7.0,
        7.0,
        17,
        Saturation::Const(0.0),
        Saturation::Identity,
    );
    let bps = pwl.breakpoints();
    assert_eq!(bps[0], -7.0_f32, "linspace endpoint inclusive at lo");
    assert_eq!(bps[bps.len() - 1], 7.0_f32, "linspace endpoint inclusive at hi");
}

#[test]
fn test06_pwl_identity_target_produces_unit_slopes() {
    // f(x) = x → slopes all 1, intercepts all 0.
    let pwl = PiecewiseLinear::new(
        |x| x,
        -1.0,
        1.0,
        4,
        Saturation::Const(0.0),
        Saturation::Const(0.0),
    );
    for &s in pwl.slopes() {
        assert!((s - 1.0).abs() < 1e-6, "slope ≈ 1, got {s}");
    }
    for &b in pwl.intercepts() {
        assert!(b.abs() < 1e-6, "intercept ≈ 0, got {b}");
    }
}

#[test]
fn test07_pwl_apply_at_breakpoint_returns_target_value() {
    // For a linear target the PWL output must match the target at every
    // breakpoint (the segments interpolate exactly).
    let pwl = PiecewiseLinear::new(
        |x| 2.0 * x + 1.0,
        -3.0,
        3.0,
        6,
        Saturation::Const(0.0),
        Saturation::Const(0.0),
    );
    for &bp in pwl.breakpoints() {
        let want = 2.0 * bp + 1.0;
        let got = pwl.apply(bp);
        assert!(
            (got - want).abs() < 1e-5,
            "at bp={bp}: got {got}, want {want}"
        );
    }
}

// ─── silu_pw: saturation rules from spec §0.9.4 ─────────────────────────

#[test]
fn test08_silu_pw_below_minus7_clamps_to_zero() {
    let pwl = silu_pw();
    assert_eq!(pwl.apply(-7.5), 0.0_f32);
    assert_eq!(pwl.apply(-100.0), 0.0_f32);
}

#[test]
fn test09_silu_pw_above_7_is_identity() {
    let pwl = silu_pw();
    // Spec §0.9.4: above(x) = x
    assert_eq!(pwl.apply(7.5).to_bits(), 7.5_f32.to_bits());
    assert_eq!(pwl.apply(123.0).to_bits(), 123.0_f32.to_bits());
}

#[test]
fn test10_silu_pw_segment_count_is_17() {
    assert_eq!(silu_pw().n_segments(), 17);
}

// ─── exp_pw: saturation rules from spec §0.9.5 ──────────────────────────

#[test]
fn test11_exp_pw_below_minus4_clamps_to_zero() {
    let pwl = exp_pw();
    assert_eq!(pwl.apply(-5.0), 0.0_f32);
    assert_eq!(pwl.apply(-100.0), 0.0_f32);
}

#[test]
fn test12_exp_pw_above_1_clamps_to_e() {
    let pwl = exp_pw();
    let e_f32: f32 = std::f32::consts::E;
    assert_eq!(pwl.apply(1.5).to_bits(), e_f32.to_bits());
    assert_eq!(pwl.apply(100.0).to_bits(), e_f32.to_bits());
}

#[test]
fn test13_exp_pw_segment_count_is_11() {
    assert_eq!(exp_pw().n_segments(), 11);
}

// ─── PWL slice helper ───────────────────────────────────────────────────

#[test]
fn test14_pwl_apply_slice_matches_per_element() {
    let pwl = silu_pw();
    let xs = [-8.0_f32, -1.0, 0.0, 0.5, 8.0];
    let mut out = [0.0_f32; 5];
    pwl.apply_slice(&xs, &mut out);
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(out[i].to_bits(), pwl.apply(x).to_bits(), "i={i}");
    }
}
