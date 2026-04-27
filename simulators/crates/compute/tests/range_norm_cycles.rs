//! Tests 11–12 — `compute::range_norm::cycles_*`
//! Pin the cycle-count formula from the paper §4.1, §5.3:
//!
//!     cycles = ceil(25 * D / units) + DIV_LATENCY
//!
//! Test 11 covers the baseline (single compute unit); test 12 pins behavior
//! at the paper's "10 units" sweet spot.

use emamba_compute::range_norm::RangeNormUnit;

#[test]
fn cycles_baseline_one_unit() {
    let u = RangeNormUnit::new(1, 4); // 1 compute unit, div_latency=4
    assert_eq!(u.cycles_for(20), 25 * 20 + 4);
    assert_eq!(u.cycles_for(20), 504);
}

#[test]
fn cycles_with_ten_units() {
    let u = RangeNormUnit::new(10, 4);
    // ceil(25 * 20 / 10) + 4 = ceil(50) + 4 = 54
    assert_eq!(u.cycles_for(20), 54);
}

#[test]
fn cycles_round_up_when_units_dont_divide_evenly() {
    let u = RangeNormUnit::new(7, 4);
    // ceil(25 * 20 / 7) = ceil(500 / 7) = ceil(71.42) = 72; + 4 = 76
    assert_eq!(u.cycles_for(20), 76);
}

#[test]
fn cycles_scale_linearly_with_d() {
    let u = RangeNormUnit::new(1, 0);
    assert_eq!(u.cycles_for(1), 25);
    assert_eq!(u.cycles_for(2), 50);
    assert_eq!(u.cycles_for(40), 1000);
}
