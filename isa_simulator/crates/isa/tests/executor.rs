//! TDD ladder for the OpExecutor — end-to-end RType decode → descriptor
//! fetch → semantics → bus update.

use emamba_core::Dtype;
use emamba_isa::desc::{DmaDesc, MacDesc, NormDesc, PwlDesc};
use emamba_isa::executor::{status, OpExecutor};
use emamba_isa::insn::{funct7, OPCODE_CUSTOM_0};
use emamba_isa::memory::{
    read_f32_slice, standard_bus, write_f32_slice, A_SRAM_BASE, DRAM_BASE, MemAccessSize,
    Memory, W_SRAM_BASE,
};
use emamba_isa::numerics::silu_pw;

/// Build an R-type word with funct3 = 0 (Xemamba v0.2 group).
fn enc(f7: u32, rs2: u32, rs1: u32, rd: u32) -> u32 {
    (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (rd << 7) | OPCODE_CUSTOM_0
}

/// Write a descriptor's bytes at the given DRAM address (descriptors live
/// in DRAM in v0.2 — that's the only region the host-side window covers).
fn write_desc(bus: &mut emamba_isa::memory::MemorySpace, addr: u32, bytes: &[u8]) {
    for (i, &b) in bytes.iter().enumerate() {
        assert!(bus.write_mem(addr + i as u32, MemAccessSize::Byte, b as u32));
    }
}

// ─── silu happy path ─────────────────────────────────────────────────────

#[test]
fn test01_silu_end_to_end() {
    let mut bus = standard_bus();

    // Place a 4-element f32 input at A-SRAM offset 0.
    let xs = [-1.0_f32, 0.0, 1.5, 100.0];
    let src = A_SRAM_BASE;
    let dst = A_SRAM_BASE + 0x1000;
    write_f32_slice(&mut bus, src, &xs).unwrap();

    // Place a PwlDesc in DRAM.
    let desc = PwlDesc {
        src,
        dst,
        len: xs.len() as u16,
        dtype: Dtype::Fp32,
    };
    let desc_addr = DRAM_BASE;
    write_desc(&mut bus, desc_addr, &desc.to_bytes());

    // Issue: xemamba.silu x0, <rs1=desc_addr in pretend reg>, x0
    // The executor takes rs1's *value* as a parameter (not a reg index).
    let insn = enc(funct7::SILU, 0, /*rs1*/ 1, /*rd*/ 0);
    let mut executor = OpExecutor::new(&mut bus);
    let status = executor.step(insn, /*rs1_value=*/ desc_addr as u64);
    assert_eq!(status, status::SUCCESS);

    // Verify: dst now holds silu_pw(xs) bit-for-bit.
    let pwl = silu_pw();
    let mut want = [0.0_f32; 4];
    pwl.apply_slice(&xs, &mut want);
    let mut got = [0.0_f32; 4];
    read_f32_slice(&mut bus, dst, &mut got).unwrap();
    for i in 0..4 {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "i={i}");
    }
}

// ─── exp happy path ──────────────────────────────────────────────────────

#[test]
fn test02_exp_end_to_end() {
    let mut bus = standard_bus();
    let xs = [-5.0_f32, -1.0, 0.0, 0.5, 2.0]; // hits below, in-range, above
    let src = A_SRAM_BASE;
    let dst = A_SRAM_BASE + 0x100;
    write_f32_slice(&mut bus, src, &xs).unwrap();

    let desc = PwlDesc {
        src,
        dst,
        len: xs.len() as u16,
        dtype: Dtype::Fp32,
    };
    let desc_addr = DRAM_BASE + 0x40;
    write_desc(&mut bus, desc_addr, &desc.to_bytes());

    let insn = enc(funct7::EXP, 0, 1, 0);
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(insn, desc_addr as u64), status::SUCCESS);

    let pwl = emamba_isa::numerics::exp_pw();
    let mut want = [0.0_f32; 5];
    pwl.apply_slice(&xs, &mut want);
    let mut got = [0.0_f32; 5];
    read_f32_slice(&mut bus, dst, &mut got).unwrap();
    for i in 0..5 {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "i={i}");
    }
}

// ─── DMA load happy path ─────────────────────────────────────────────────

#[test]
fn test03_dma_load_copies_dram_to_sram() {
    let mut bus = standard_bus();
    // Pre-populate 16 bytes in DRAM at the start.
    let payload: [u8; 16] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
    ];
    let src_dram = DRAM_BASE + 0x100;
    for (i, &b) in payload.iter().enumerate() {
        bus.write_mem(src_dram + i as u32, MemAccessSize::Byte, b as u32);
    }
    // DMA descriptor lives elsewhere in DRAM.
    let desc = DmaDesc {
        dram_addr: src_dram as u64,
        sram_addr: W_SRAM_BASE,
        bytes: payload.len() as u32,
    };
    let desc_addr = DRAM_BASE;
    write_desc(&mut bus, desc_addr, &desc.to_bytes());

    let insn = enc(funct7::DMAL, 0, 1, 0);
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(insn, desc_addr as u64), status::SUCCESS);

    // Verify W-SRAM was populated.
    for (i, &want) in payload.iter().enumerate() {
        let got = bus.read_mem(W_SRAM_BASE + i as u32, MemAccessSize::Byte).unwrap();
        assert_eq!(got, want as u32, "byte {i}");
    }
}

// ─── range_norm happy path ───────────────────────────────────────────────

#[test]
fn test04_norm_end_to_end() {
    let mut bus = standard_bus();
    let len = 4u16;
    let xs = [-1.0_f32, 0.0, 1.0, 2.0];
    let g = [1.0_f32; 4];
    let b = [0.0_f32; 4];
    let src = A_SRAM_BASE;
    let dst = A_SRAM_BASE + 0x100;
    let gamma = A_SRAM_BASE + 0x200;
    let beta = A_SRAM_BASE + 0x300;
    write_f32_slice(&mut bus, src, &xs).unwrap();
    write_f32_slice(&mut bus, gamma, &g).unwrap();
    write_f32_slice(&mut bus, beta, &b).unwrap();

    let desc = NormDesc {
        src,
        dst,
        gamma,
        beta,
        len,
        dtype: Dtype::Fp32,
    };
    let desc_addr = DRAM_BASE;
    write_desc(&mut bus, desc_addr, &desc.to_bytes());

    let insn = enc(funct7::NORM, 0, 1, 0);
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(insn, desc_addr as u64), status::SUCCESS);

    let want = emamba_isa::numerics::range_normalization(&xs, &g, &b, 1e-5);
    let mut got = [0.0_f32; 4];
    read_f32_slice(&mut bus, dst, &mut got).unwrap();
    for i in 0..4 {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "i={i}");
    }
}

// ─── mac happy path: tiny 2x2 · 2x2 = 2x2 ────────────────────────────────

#[test]
fn test05_mac_2x2_end_to_end() {
    let mut bus = standard_bus();
    // src_a = [[1, 2], [3, 4]]; src_b = [[5, 6], [7, 8]]
    // dst = src_a · src_b = [[19, 22], [43, 50]]
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let src_a = W_SRAM_BASE;
    let src_b = W_SRAM_BASE + 0x100;
    let dst = A_SRAM_BASE;
    write_f32_slice(&mut bus, src_a, &a).unwrap();
    write_f32_slice(&mut bus, src_b, &b).unwrap();

    let desc = MacDesc {
        src_a,
        src_b,
        dst,
        m: 2,
        n: 2,
        k: 2,
        dtype: Dtype::Fp32,
    };
    let desc_addr = DRAM_BASE;
    write_desc(&mut bus, desc_addr, &desc.to_bytes());

    let insn = enc(funct7::MAC, 0, 1, 0);
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(insn, desc_addr as u64), status::SUCCESS);

    let mut got = [0.0_f32; 4];
    read_f32_slice(&mut bus, dst, &mut got).unwrap();
    assert_eq!(got, [19.0, 22.0, 43.0, 50.0]);
}

// ─── exception paths ─────────────────────────────────────────────────────

#[test]
fn test06_unknown_funct7_returns_illegal_opcode() {
    let mut bus = standard_bus();
    let insn = enc(0x42, 0, 0, 0); // 0x42 ∉ {1..8}
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(insn, 0), status::ILLEGAL_OPCODE);
}

#[test]
fn test07_non_xemamba_opcode_returns_illegal_opcode() {
    let mut bus = standard_bus();
    // Standard ADD: opcode 0x33, not custom-0.
    let add = 0x003100B3u32;
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(add, 0), status::ILLEGAL_OPCODE);
}

#[test]
fn test08_descriptor_outside_bus_returns_descriptor_fault() {
    let mut bus = standard_bus();
    let insn = enc(funct7::SILU, 0, 1, 0);
    let mut executor = OpExecutor::new(&mut bus);
    // 4-aligned but unmapped — descriptor read should fault.
    assert_eq!(executor.step(insn, 0xFFFF_FF00), status::DESCRIPTOR_FAULT);
}

#[test]
fn test09_compute_op_pointing_outside_sram_returns_bus_fault() {
    let mut bus = standard_bus();
    let desc = PwlDesc {
        src: 0xDEAD_BEEF, // not in any SRAM region
        dst: A_SRAM_BASE,
        len: 4,
        dtype: Dtype::Fp32,
    };
    let desc_addr = DRAM_BASE;
    write_desc(&mut bus, desc_addr, &desc.to_bytes());
    let insn = enc(funct7::SILU, 0, 1, 0);
    let mut executor = OpExecutor::new(&mut bus);
    assert_eq!(executor.step(insn, desc_addr as u64), status::BUS_FAULT);
}
