//! TDD ladder for the bus / memory model — `isa_spec.md` v0.2 §0.5.

use emamba_isa::memory::{
    standard_bus, region_names, MemAccessSize, Memory, MemorySpace, MemorySpaceError,
    VecMemory, A_SRAM_BASE, A_SRAM_SIZE, DRAM_BASE, STATE_SRAM_BASE, STATE_SRAM_SIZE,
    W_SRAM_BASE, W_SRAM_SIZE,
};

// ─── VecMemory: byte / half / word access patterns ──────────────────────

#[test]
fn test01_vec_memory_byte_round_trip() {
    let mut mem = VecMemory::new(16);
    assert!(mem.write_mem(0, MemAccessSize::Byte, 0xAB));
    assert_eq!(mem.read_mem(0, MemAccessSize::Byte), Some(0xAB));
}

#[test]
fn test02_vec_memory_word_then_byte_reads() {
    let mut mem = VecMemory::new(16);
    assert!(mem.write_mem(0, MemAccessSize::Word, 0xDEAD_BEEF));
    assert_eq!(mem.read_mem(0, MemAccessSize::Word), Some(0xDEAD_BEEF));
    // Little-endian byte order: low byte at low addr.
    // 0xDEAD_BEEF.to_le_bytes() = [0xEF, 0xBE, 0xAD, 0xDE]
    assert_eq!(mem.read_mem(0, MemAccessSize::Byte), Some(0xEF));
    assert_eq!(mem.read_mem(1, MemAccessSize::Byte), Some(0xBE));
    assert_eq!(mem.read_mem(2, MemAccessSize::Byte), Some(0xAD));
    assert_eq!(mem.read_mem(3, MemAccessSize::Byte), Some(0xDE));
    assert_eq!(mem.read_mem(0, MemAccessSize::HalfWord), Some(0xBEEF));
    assert_eq!(mem.read_mem(2, MemAccessSize::HalfWord), Some(0xDEAD));
}

#[test]
fn test03_vec_memory_unaligned_access_returns_none() {
    let mut mem = VecMemory::new(16);
    // Word access must be 4-byte aligned; halfword must be 2-byte aligned.
    assert_eq!(mem.read_mem(1, MemAccessSize::Word), None);
    assert_eq!(mem.read_mem(2, MemAccessSize::Word), None);
    assert_eq!(mem.read_mem(1, MemAccessSize::HalfWord), None);
    assert!(!mem.write_mem(1, MemAccessSize::Word, 0));
}

#[test]
fn test04_vec_memory_out_of_bounds_returns_none() {
    let mut mem = VecMemory::new(16);
    assert_eq!(mem.read_mem(16, MemAccessSize::Byte), None);
    assert_eq!(mem.read_mem(13, MemAccessSize::Word), None); // 13 + 4 > 16
    assert!(!mem.write_mem(15, MemAccessSize::HalfWord, 0));
}

// ─── f32 slice helpers (used by every compute op) ───────────────────────

#[test]
fn test05_f32_slice_round_trip() {
    use emamba_isa::memory::{read_f32_slice, write_f32_slice};
    let mut mem = VecMemory::new(64);
    let data = [1.0_f32, -2.5, 3.1415927, 0.0, f32::INFINITY, -0.0];
    write_f32_slice(&mut mem, 0, &data).expect("write");
    let mut out = vec![0.0_f32; data.len()];
    read_f32_slice(&mut mem, 0, &mut out).expect("read");
    // Bit-exact compare via to_bits — 0.0 != -0.0 by value but distinct in bits.
    for (a, b) in data.iter().zip(out.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// ─── MemorySpace: routing across multiple regions ───────────────────────

#[test]
fn test06_memory_space_routes_to_region() {
    let mut bus = MemorySpace::new();
    let idx_a = bus
        .add_memory(0x1000, 0x100, Box::new(VecMemory::new(0x100)))
        .expect("add A");
    let idx_b = bus
        .add_memory(0x2000, 0x100, Box::new(VecMemory::new(0x100)))
        .expect("add B");
    assert_ne!(idx_a, idx_b);

    bus.write_mem(0x1010, MemAccessSize::Word, 0xAAAA_AAAA);
    bus.write_mem(0x2010, MemAccessSize::Word, 0xBBBB_BBBB);
    assert_eq!(bus.read_mem(0x1010, MemAccessSize::Word), Some(0xAAAA_AAAA));
    assert_eq!(bus.read_mem(0x2010, MemAccessSize::Word), Some(0xBBBB_BBBB));
}

#[test]
fn test07_memory_space_overlap_rejected() {
    let mut bus = MemorySpace::new();
    bus.add_memory(0x1000, 0x100, Box::new(VecMemory::new(0x100)))
        .unwrap();
    let err = bus
        .add_memory(0x1080, 0x100, Box::new(VecMemory::new(0x100)))
        .unwrap_err();
    assert_eq!(err, MemorySpaceError::RegionOverlap);
}

#[test]
fn test08_memory_space_unmapped_access_returns_none() {
    let mut bus = MemorySpace::new();
    bus.add_memory(0x1000, 0x100, Box::new(VecMemory::new(0x100)))
        .unwrap();
    // Address outside any region — BusFault domain in spec §0.10.
    assert_eq!(bus.read_mem(0x5000, MemAccessSize::Word), None);
    assert!(!bus.write_mem(0x5000, MemAccessSize::Word, 0));
}

// ─── spec §0.5 standard bus map ─────────────────────────────────────────

#[test]
fn test09_standard_bus_has_three_srams_and_dram() {
    let bus = standard_bus();
    let names = region_names(&bus);
    // Order is deterministic so test names — that is the spec contract.
    assert_eq!(names, vec!["W-SRAM", "A-SRAM", "STATE-SRAM", "DRAM"]);
}

#[test]
fn test10_standard_bus_addresses_match_spec() {
    // Constants from spec §0.5.1 + §0.5.2.
    assert_eq!(W_SRAM_BASE, 0x0000_0000);
    assert_eq!(W_SRAM_SIZE, 4 * 1024 * 1024); // 4 MB
    assert_eq!(A_SRAM_BASE, 0x0040_0000);
    assert_eq!(A_SRAM_SIZE, 1 * 1024 * 1024); // 1 MB
    assert_eq!(STATE_SRAM_BASE, 0x0050_0000);
    assert_eq!(STATE_SRAM_SIZE, 2 * 1024 * 1024); // 2 MB
    assert_eq!(DRAM_BASE, 0x1000_0000);
    // 23-bit SRAM space invariant: STATE-SRAM top fits in 23 bits.
    assert!(STATE_SRAM_BASE + STATE_SRAM_SIZE - 1 <= 0x007F_FFFF);
}

#[test]
fn test11_standard_bus_round_trip_in_each_sram() {
    let mut bus = standard_bus();
    // Write a u32 to each SRAM region, read it back.
    for &base in &[W_SRAM_BASE, A_SRAM_BASE, STATE_SRAM_BASE, DRAM_BASE] {
        bus.write_mem(base, MemAccessSize::Word, 0xCAFE_BABE);
        assert_eq!(
            bus.read_mem(base, MemAccessSize::Word),
            Some(0xCAFE_BABE),
            "round-trip at {:#x}",
            base
        );
    }
}
