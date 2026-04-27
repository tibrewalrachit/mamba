//! Test 13 — `compute::conv1d::cycles_on_d20_k4`
//! cycles = D + (k - 1) + PIPELINE_FILL  (paper §4.3)

use emamba_compute::conv1d::Conv1DUnit;

#[test]
fn cycles_on_d20_k4() {
    let u = Conv1DUnit::new(4, 2); // K=4, pipeline_fill=2
    // 20 + (4 - 1) + 2 = 25
    assert_eq!(u.cycles_for(20), 25);
}

#[test]
fn cycles_scale_with_d() {
    let u = Conv1DUnit::new(4, 2);
    assert_eq!(u.cycles_for(40), 45);
    assert_eq!(u.cycles_for(100), 105);
}

#[test]
fn larger_kernel_costs_more() {
    let small = Conv1DUnit::new(4, 0);
    let big = Conv1DUnit::new(7, 0);
    assert_eq!(big.cycles_for(20) - small.cycles_for(20), 3);
}
