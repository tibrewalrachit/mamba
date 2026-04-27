//! Test 1 — `core::dtype::bytes_match_spec`
//! Asserts the per-element byte sizes implied by the eMamba paper:
//! Int8 = 1, Int24 = 3 (h_t state), Fp32 = 4 (FP baseline).

use emamba_core::Dtype;

#[test]
fn bytes_match_spec() {
    assert_eq!(Dtype::Int8.bytes(), 1);
    assert_eq!(Dtype::Int24.bytes(), 3);
    assert_eq!(Dtype::Fp32.bytes(), 4);
}
