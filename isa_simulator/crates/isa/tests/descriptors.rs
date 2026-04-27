//! TDD ladder for the six descriptor structs in `isa_spec.md` v0.2 §0.7.
//!
//! Each test builds the descriptor's byte image *by hand* using the offsets
//! the spec mandates, then asserts that `from_bytes` round-trips through the
//! struct and `to_bytes` reproduces the same bytes. This way the tests
//! verify the spec, not the code.

use emamba_core::Dtype;
use emamba_isa::desc::{
    ConvDesc, DescError, DmaDesc, MacDesc, NormDesc, PwlDesc, SsmDesc,
};

// Helpers — encode primitive fields little-endian into a Vec<u8>.

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

// ─── §0.7.1 mac_desc_t — 20 bytes ───────────────────────────────────────

#[test]
fn test01_mac_desc_layout_matches_spec() {
    // From spec §0.12 example 1.
    let mut bytes = Vec::with_capacity(20);
    push_u32(&mut bytes, 0x000000); // src_a
    push_u32(&mut bytes, 0x004800); // src_b
    push_u32(&mut bytes, 0x401000); // dst
    push_u16(&mut bytes, 16); // m
    push_u16(&mut bytes, 16); // n
    push_u16(&mut bytes, 768); // k
    push_u8(&mut bytes, 0); // dtype = FP32
    push_u8(&mut bytes, 0); // reserved
    assert_eq!(bytes.len(), 20, "mac_desc_t is 20 bytes per spec §0.7.1");

    let d = MacDesc::from_bytes(&bytes).expect("decode");
    assert_eq!(d.src_a, 0x000000);
    assert_eq!(d.src_b, 0x004800);
    assert_eq!(d.dst, 0x401000);
    assert_eq!(d.m, 16);
    assert_eq!(d.n, 16);
    assert_eq!(d.k, 768);
    assert_eq!(d.dtype, Dtype::Fp32);

    assert_eq!(d.to_bytes().as_slice(), &bytes[..]);
}

#[test]
fn test02_mac_desc_short_input_errors() {
    let bytes = vec![0u8; 19];
    assert_eq!(MacDesc::from_bytes(&bytes), Err(DescError::ShortInput));
}

#[test]
fn test03_mac_desc_reserved_nonzero_errors() {
    let mut bytes = vec![0u8; 20];
    bytes[0] = 0x10; // src_a low byte (legal)
    bytes[14] = 0x01; // dtype = INT8 (legal decode in v0.2)
    bytes[15] = 0xff; // reserved (illegal)
    bytes[12] = 1; // m must be nonzero
    bytes[13] = 0;
    bytes[10] = 1; // n
    bytes[8] = 1; // k (low byte)
    assert_eq!(MacDesc::from_bytes(&bytes), Err(DescError::Malformed));
}

#[test]
fn test04_mac_desc_zero_dim_errors() {
    let mut bytes = vec![0u8; 20];
    // m = 0 (illegal per spec §0.7.1 — must be 1..=4096)
    assert_eq!(MacDesc::from_bytes(&bytes), Err(DescError::Malformed));
}

#[test]
fn test05_mac_desc_illegal_dtype_errors() {
    let mut bytes = Vec::with_capacity(20);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u16(&mut bytes, 1);
    push_u16(&mut bytes, 1);
    push_u16(&mut bytes, 1);
    push_u8(&mut bytes, 0b011); // reserved dtype
    push_u8(&mut bytes, 0);
    assert_eq!(MacDesc::from_bytes(&bytes), Err(DescError::IllegalDtype));
}

// ─── §0.7.2 norm_desc_t — 20 bytes ──────────────────────────────────────

#[test]
fn test06_norm_desc_layout_matches_spec() {
    let mut bytes = Vec::with_capacity(20);
    push_u32(&mut bytes, 0x400000); // src
    push_u32(&mut bytes, 0x401000); // dst
    push_u32(&mut bytes, 0x402000); // gamma
    push_u32(&mut bytes, 0x403000); // beta
    push_u16(&mut bytes, 768); // len
    push_u8(&mut bytes, 0); // dtype = FP32
    push_u8(&mut bytes, 0);
    assert_eq!(bytes.len(), 20);

    let d = NormDesc::from_bytes(&bytes).expect("decode");
    assert_eq!(d.src, 0x400000);
    assert_eq!(d.dst, 0x401000);
    assert_eq!(d.gamma, 0x402000);
    assert_eq!(d.beta, 0x403000);
    assert_eq!(d.len, 768);
    assert_eq!(d.dtype, Dtype::Fp32);
    assert_eq!(d.to_bytes().as_slice(), &bytes[..]);
}

// ─── §0.7.3 conv_desc_t — 20 bytes ──────────────────────────────────────

#[test]
fn test07_conv_desc_layout_matches_spec() {
    let mut bytes = Vec::with_capacity(20);
    push_u32(&mut bytes, 0x400000); // src
    push_u32(&mut bytes, 0x401000); // dst
    push_u32(&mut bytes, 0x000000); // weights
    push_u32(&mut bytes, 0x500000); // cache
    push_u16(&mut bytes, 1536); // len = ED
    push_u8(&mut bytes, 0); // dtype
    push_u8(&mut bytes, 4); // k_conv
    assert_eq!(bytes.len(), 20);

    let d = ConvDesc::from_bytes(&bytes).expect("decode");
    assert_eq!(d.src, 0x400000);
    assert_eq!(d.dst, 0x401000);
    assert_eq!(d.weights, 0x000000);
    assert_eq!(d.cache, 0x500000);
    assert_eq!(d.len, 1536);
    assert_eq!(d.dtype, Dtype::Fp32);
    assert_eq!(d.k_conv, 4);
    assert_eq!(d.to_bytes().as_slice(), &bytes[..]);
}

#[test]
fn test08_conv_desc_zero_kconv_errors() {
    let mut bytes = Vec::with_capacity(20);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u16(&mut bytes, 1);
    push_u8(&mut bytes, 0);
    push_u8(&mut bytes, 0); // k_conv = 0 → Malformed (spec: 1..=63)
    assert_eq!(ConvDesc::from_bytes(&bytes), Err(DescError::Malformed));
}

// ─── §0.7.4 ssm_desc_t — 16 bytes ───────────────────────────────────────

#[test]
fn test09_ssm_desc_layout_matches_spec() {
    let mut bytes = Vec::with_capacity(16);
    push_u32(&mut bytes, 0x400000); // ctx_in
    push_u32(&mut bytes, 0x500000); // state_h
    push_u32(&mut bytes, 0x420000); // ctx_out
    push_u16(&mut bytes, 1536); // ed
    push_u8(&mut bytes, 16); // n
    push_u8(&mut bytes, 0); // dtype
    assert_eq!(bytes.len(), 16);

    let d = SsmDesc::from_bytes(&bytes).expect("decode");
    assert_eq!(d.ctx_in, 0x400000);
    assert_eq!(d.state_h, 0x500000);
    assert_eq!(d.ctx_out, 0x420000);
    assert_eq!(d.ed, 1536);
    assert_eq!(d.n, 16);
    assert_eq!(d.dtype, Dtype::Fp32);
    assert_eq!(d.to_bytes().as_slice(), &bytes[..]);
}

// ─── §0.7.5 pwl_desc_t — 12 bytes ───────────────────────────────────────

#[test]
fn test10_pwl_desc_layout_matches_spec() {
    let mut bytes = Vec::with_capacity(12);
    push_u32(&mut bytes, 0x400000); // src
    push_u32(&mut bytes, 0x401000); // dst
    push_u16(&mut bytes, 1536); // len
    push_u8(&mut bytes, 0); // dtype
    push_u8(&mut bytes, 0); // reserved
    assert_eq!(bytes.len(), 12);

    let d = PwlDesc::from_bytes(&bytes).expect("decode");
    assert_eq!(d.src, 0x400000);
    assert_eq!(d.dst, 0x401000);
    assert_eq!(d.len, 1536);
    assert_eq!(d.dtype, Dtype::Fp32);
    assert_eq!(d.to_bytes().as_slice(), &bytes[..]);
}

// ─── §0.7.6 dma_desc_t — 16 bytes ───────────────────────────────────────

#[test]
fn test11_dma_desc_layout_matches_spec() {
    // From spec §0.12 example 3.
    let mut bytes = Vec::with_capacity(16);
    push_u64(&mut bytes, 0x10000000); // dram_addr
    push_u32(&mut bytes, 0x000000); // sram_addr
    push_u32(&mut bytes, 0x18000); // bytes
    assert_eq!(bytes.len(), 16);

    let d = DmaDesc::from_bytes(&bytes).expect("decode");
    assert_eq!(d.dram_addr, 0x10000000);
    assert_eq!(d.sram_addr, 0x000000);
    assert_eq!(d.bytes, 0x18000);
    assert_eq!(d.to_bytes().as_slice(), &bytes[..]);
}

#[test]
fn test12_dma_desc_zero_bytes_is_legal() {
    // §0.7.6: "bytes may be 0."
    let mut bytes = Vec::with_capacity(16);
    push_u64(&mut bytes, 0x10000000);
    push_u32(&mut bytes, 0x000000);
    push_u32(&mut bytes, 0); // bytes = 0 → no-op, legal
    let d = DmaDesc::from_bytes(&bytes).expect("zero-byte DMA is legal");
    assert_eq!(d.bytes, 0);
}
