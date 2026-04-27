//! TDD ladder for `RType` — every test pins a single contract from
//! `isa_spec.md` v0.2 §0.4 (R-type bit layout), §0.6 (funct7 selectors),
//! and §0.12 (worked encoding examples).
//!
//! Each test names the spec section it enforces. Numbered to keep ordering
//! stable when adding new ones.

use emamba_isa::insn::{funct7, RType, FUNCT3_XEMAMBA_V0_2, OPCODE_CUSTOM_0};

// --- spec §0.4 + §0.12 example 1: xemamba.mac x0, a0, x0 → 0x0205_000B ---

#[test]
fn test01_decode_mac_example() {
    let insn = RType::new(0x0205_000B);
    assert_eq!(insn.opcode, OPCODE_CUSTOM_0, "opcode = custom-0");
    assert_eq!(insn.funct3, FUNCT3_XEMAMBA_V0_2, "funct3 = 0 (xemamba v0.2)");
    assert_eq!(insn.funct7, funct7::MAC, "funct7 = 0x01 (mac)");
    assert_eq!(insn.rs1, 10, "rs1 = a0 (x10)");
    assert_eq!(insn.rs2, 0, "rs2 = x0 (unused)");
    assert_eq!(insn.rd, 0, "rd = x0 (status discarded)");
}

// --- spec §0.12 example 2: xemamba.silu x0, a1, x0 → 0x0A05_800B ---

#[test]
fn test02_decode_silu_example() {
    let insn = RType::new(0x0A05_800B);
    assert_eq!(insn.opcode, OPCODE_CUSTOM_0);
    assert_eq!(insn.funct7, funct7::SILU);
    assert_eq!(insn.rs1, 11, "rs1 = a1 (x11)");
    assert_eq!(insn.rs2, 0);
    assert_eq!(insn.rd, 0);
}

// --- spec §0.12 example 3: xemamba.dmal t0, a2, x0 → 0x0E06_028B ---

#[test]
fn test03_decode_dmal_example() {
    let insn = RType::new(0x0E06_028B);
    assert_eq!(insn.opcode, OPCODE_CUSTOM_0);
    assert_eq!(insn.funct7, funct7::DMAL);
    assert_eq!(insn.rs1, 12, "rs1 = a2 (x12)");
    assert_eq!(insn.rs2, 0);
    assert_eq!(insn.rd, 5, "rd = t0 (x5)");
}

// --- round-trip: encode(decode(x)) == x for every spec example ---

#[test]
fn test04_round_trip_all_spec_examples() {
    for &raw in &[0x0205_000Bu32, 0x0A05_800B, 0x0E06_028B] {
        assert_eq!(RType::new(raw).encode(), raw, "round-trip {:#x}", raw);
    }
}

// --- §0.4 field widths — every field masks to its spec width ---

#[test]
fn test05_field_widths_clamp_correctly() {
    // All-ones in all field positions; every field should saturate to its
    // spec width (funct7=7, rs/rd=5, funct3=3, opcode=7).
    let max = RType {
        funct7: 0xff,
        rs2: 0xff,
        rs1: 0xff,
        funct3: 0xff,
        rd: 0xff,
        opcode: 0xff,
    };
    let encoded = max.encode();
    let decoded = RType::new(encoded);
    assert_eq!(decoded.funct7, 0x7f);
    assert_eq!(decoded.rs2, 0x1f);
    assert_eq!(decoded.rs1, 0x1f);
    assert_eq!(decoded.funct3, 0x07);
    assert_eq!(decoded.rd, 0x1f);
    assert_eq!(decoded.opcode, 0x7f);
}

// --- §0.4: is_xemamba is true iff opcode is custom-0 ---

#[test]
fn test06_is_xemamba_predicate() {
    assert!(RType::new(0x0205_000B).is_xemamba());
    // Standard ADD: opcode 0b011_0011 = 0x33 → not xemamba.
    let add = RType::new(0x003100B3);
    assert_eq!(add.opcode, 0x33);
    assert!(!add.is_xemamba());
}

// --- spec §0.6: every named funct7 selector has a unique nonzero value in 1..=8 ---

#[test]
fn test07_funct7_selectors_unique_in_range() {
    let all = [
        funct7::MAC,
        funct7::NORM,
        funct7::CONV,
        funct7::SSM,
        funct7::SILU,
        funct7::EXP,
        funct7::DMAL,
        funct7::DMAS,
    ];
    for f in all {
        assert!(f >= 1 && f <= 8, "funct7 {} outside spec range 1..=8", f);
    }
    let mut sorted = all.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len(), "funct7 selectors must be unique");
}
