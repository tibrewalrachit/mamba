//! Tests 16-18 — `compute::pwl::*`, `compute::relu::*`
//! All three are fully-pipelined element-wise units with single-cycle LUT or
//! comparator latency, so cycles_for(D) = D + lut_latency-1 once filled.
//! Following the paper §4.5: 17-segment SiLU and 11-segment exp LUTs both
//! single-cycle; ReLU is a single-cycle compare.

use emamba_compute::{pwl::{PwlExpUnit, PwlSiluUnit}, relu::ReluUnit};

#[test]
fn pwl_silu_pipelined_throughput_one_per_cycle() {
    let u = PwlSiluUnit::new(17, 1);
    assert_eq!(u.segments(), 17);
    assert_eq!(u.cycles_for(20), 20);
}

#[test]
fn pwl_exp_same_shape_as_silu() {
    let u = PwlExpUnit::new(11, 1);
    assert_eq!(u.segments(), 11);
    assert_eq!(u.cycles_for(20), 20);
}

#[test]
fn relu_single_cycle_per_element() {
    let u = ReluUnit::new();
    assert_eq!(u.cycles_for(20), 20);
    assert_eq!(u.cycles_for(1), 1);
    assert_eq!(u.cycles_for(0), 0);
}

#[test]
fn higher_lut_latency_adds_pipeline_fill() {
    let u = PwlSiluUnit::new(17, 3);
    // pipelined throughput is still one per cycle; pipeline fill of (lat-1).
    assert_eq!(u.cycles_for(20), 20 + 2);
}
