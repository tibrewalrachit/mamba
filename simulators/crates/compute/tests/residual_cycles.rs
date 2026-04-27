//! Bonus test (plan §3 row 'ResidualAddUnit').
//! cycles = ceil(D / adder_width).

use emamba_compute::residual::ResidualAddUnit;

#[test]
fn residual_cycles_match_adder_width() {
    let u = ResidualAddUnit::new(16);
    // 20 / 16 = 1.25 → 2
    assert_eq!(u.cycles_for(20), 2);
    // 64 / 16 = 4
    assert_eq!(u.cycles_for(64), 4);
}

#[test]
fn wider_adder_is_faster() {
    let narrow = ResidualAddUnit::new(4);
    let wide = ResidualAddUnit::new(64);
    assert!(narrow.cycles_for(64) > wide.cycles_for(64));
}
