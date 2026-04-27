//! Tests 14–15 — `compute::ssm::*`
//! SSM_STATE  cycles = ceil(D*N*E / mac_array_width) + READ_STATE + WRITE_STATE
//! SSM_OUTPUT cycles = ceil(D*N*E / mac_array_width) + PIPELINE_FILL
//! (paper §4.4, Figure 5)

use emamba_compute::ssm::{SsmOutputUnit, SsmStateUnit};

#[test]
fn ssm_state_cycles_proportional_to_dne_over_mac() {
    let u = SsmStateUnit::new(32, 1, 1); // mac_array_width=32, read=1, write=1
    // D=20, N=8, E=2 → 320 / 32 = 10; + 1 + 1 = 12
    assert_eq!(u.cycles_for(20, 8, 2), 12);
}

#[test]
fn ssm_state_rounds_up_when_mac_doesnt_divide() {
    let u = SsmStateUnit::new(32, 1, 1);
    // D=21, N=8, E=2 → 336 / 32 = 10.5 → 11; + 2 = 13
    assert_eq!(u.cycles_for(21, 8, 2), 13);
}

#[test]
fn ssm_output_same_shape_no_state_writeback() {
    let s = SsmStateUnit::new(32, 1, 1);
    let o = SsmOutputUnit::new(32, 2);
    // Both should yield the same MAC term; difference is the write-state vs
    // pipeline-fill epilogue. With (read=1, write=1) and (fill=2), the totals
    // happen to coincide on this configuration: 10+2 == 10+2. Verify the
    // formula independently.
    let mac = ((20u64 * 8 * 2) + 32 - 1) / 32; // = 10
    assert_eq!(s.cycles_for(20, 8, 2) as u64, mac + 1 + 1);
    assert_eq!(o.cycles_for(20, 8, 2) as u64, mac + 2);
}

#[test]
fn ssm_output_with_smaller_mac_is_slower() {
    let big = SsmOutputUnit::new(64, 0);
    let small = SsmOutputUnit::new(16, 0);
    assert!(small.cycles_for(20, 8, 2) > big.cycles_for(20, 8, 2));
}
