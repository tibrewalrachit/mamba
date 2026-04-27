//! `core::dtype` — pin the byte sizes from `isa_spec.md` §0.3.

use emamba_core::Dtype;

#[test]
fn bytes_match_spec() {
    assert_eq!(Dtype::Int8.bytes(), 1);
    assert_eq!(Dtype::Int24.bytes(), 3);
    assert_eq!(Dtype::Fp32.bytes(), 4);
}

#[test]
fn dtype_bits_round_trip() {
    for d in [Dtype::Fp32, Dtype::Int8, Dtype::Int24] {
        assert_eq!(Dtype::from_bits(d.to_bits()), Some(d));
    }
    // Reserved codes round-trip to None.
    for code in 0b011u8..=0b111 {
        assert_eq!(Dtype::from_bits(code), None);
    }
}
