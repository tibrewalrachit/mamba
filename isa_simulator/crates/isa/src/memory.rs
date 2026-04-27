//! Bus / memory model — Xemamba spec §0.5.
//!
//! Mirrors rrs's split: a `Memory` trait with byte/half/word access, a
//! `VecMemory` backing store, and a `MemorySpace` bus that registers regions
//! and routes accesses by address. Region naming and the standard mamba-130m
//! bus map come from the spec.

/// Width of a single memory access. Matches RV32 sub-word access sizes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemAccessSize {
    Byte,
    HalfWord,
    Word,
}

impl MemAccessSize {
    pub fn bytes(self) -> u32 {
        match self {
            Self::Byte => 1,
            Self::HalfWord => 2,
            Self::Word => 4,
        }
    }
}

/// Anything addressable on the accelerator bus or the host-side window.
///
/// The trait is byte-addressable (returns `u32` because that's the widest
/// access size; half/byte reads are zero-extended into the low bits).
/// `read_mem` / `write_mem` return `None` / `false` on misalignment or
/// out-of-region access — the executor lifts that into the corresponding
/// spec §0.10 status code.
pub trait Memory {
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32>;
    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool;
}

// ─── VecMemory: contiguous byte-addressable backing store ───────────────

pub struct VecMemory {
    bytes: Vec<u8>,
}

impl VecMemory {
    pub fn new(size: usize) -> Self {
        Self {
            bytes: vec![0u8; size],
        }
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

impl Memory for VecMemory {
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32> {
        let n = size.bytes();
        if addr % n != 0 {
            return None;
        }
        let lo = addr as usize;
        let hi = lo.checked_add(n as usize)?;
        if hi > self.bytes.len() {
            return None;
        }
        let mut buf = [0u8; 4];
        buf[..n as usize].copy_from_slice(&self.bytes[lo..hi]);
        Some(u32::from_le_bytes(buf))
    }

    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool {
        let n = size.bytes();
        if addr % n != 0 {
            return false;
        }
        let lo = addr as usize;
        let hi = match lo.checked_add(n as usize) {
            Some(h) => h,
            None => return false,
        };
        if hi > self.bytes.len() {
            return false;
        }
        let src = store_data.to_le_bytes();
        self.bytes[lo..hi].copy_from_slice(&src[..n as usize]);
        true
    }
}

// ─── f32 slice helpers ──────────────────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SliceError {
    /// Address is not 4-byte aligned (spec §0.5.3 misalignment for FP32).
    Misaligned,
    /// Read or write extends past the region (spec §0.10 OutOfBounds).
    OutOfBounds,
}

pub fn read_f32_slice<M: Memory>(
    mem: &mut M,
    addr: u32,
    out: &mut [f32],
) -> Result<(), SliceError> {
    if addr % 4 != 0 {
        return Err(SliceError::Misaligned);
    }
    let mut a = addr;
    for slot in out.iter_mut() {
        let bits = mem.read_mem(a, MemAccessSize::Word).ok_or(SliceError::OutOfBounds)?;
        *slot = f32::from_bits(bits);
        a = a.checked_add(4).ok_or(SliceError::OutOfBounds)?;
    }
    Ok(())
}

pub fn write_f32_slice<M: Memory>(
    mem: &mut M,
    addr: u32,
    data: &[f32],
) -> Result<(), SliceError> {
    if addr % 4 != 0 {
        return Err(SliceError::Misaligned);
    }
    let mut a = addr;
    for &x in data {
        if !mem.write_mem(a, MemAccessSize::Word, x.to_bits()) {
            return Err(SliceError::OutOfBounds);
        }
        a = a.checked_add(4).ok_or(SliceError::OutOfBounds)?;
    }
    Ok(())
}

// ─── MemorySpace: register regions, route by address ────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemorySpaceError {
    RegionOverlap,
}

struct Region {
    base: u32,
    size: u32,
    name: &'static str,
    mem: Box<dyn Memory>,
}

pub struct MemorySpace {
    regions: Vec<Region>,
}

impl MemorySpace {
    pub fn new() -> Self {
        Self { regions: Vec::new() }
    }

    pub fn add_memory(
        &mut self,
        base: u32,
        size: u32,
        mem: Box<dyn Memory>,
    ) -> Result<usize, MemorySpaceError> {
        self.add_named_memory(base, size, "", mem)
    }

    pub fn add_named_memory(
        &mut self,
        base: u32,
        size: u32,
        name: &'static str,
        mem: Box<dyn Memory>,
    ) -> Result<usize, MemorySpaceError> {
        let new_end = base.saturating_add(size);
        for r in &self.regions {
            let r_end = r.base.saturating_add(r.size);
            if base < r_end && r.base < new_end {
                return Err(MemorySpaceError::RegionOverlap);
            }
        }
        self.regions.push(Region { base, size, name, mem });
        Ok(self.regions.len() - 1)
    }

    fn route_mut(&mut self, addr: u32, n: u32) -> Option<(&mut dyn Memory, u32)> {
        for r in &mut self.regions {
            let r_end = r.base.saturating_add(r.size);
            if addr >= r.base && addr.checked_add(n)? <= r_end {
                let off = addr - r.base;
                return Some((r.mem.as_mut(), off));
            }
        }
        None
    }
}

impl Default for MemorySpace {
    fn default() -> Self {
        Self::new()
    }
}

impl Memory for MemorySpace {
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32> {
        let (mem, off) = self.route_mut(addr, size.bytes())?;
        mem.read_mem(off, size)
    }

    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool {
        match self.route_mut(addr, size.bytes()) {
            Some((mem, off)) => mem.write_mem(off, size, store_data),
            None => false,
        }
    }
}

/// Region names in the order they were registered. Used by tests and the
/// disassembler to label addresses.
pub fn region_names(bus: &MemorySpace) -> Vec<&'static str> {
    bus.regions.iter().map(|r| r.name).collect()
}

// ─── spec §0.5.1 + §0.5.2 standard bus map (mamba-130m sizing) ──────────

pub const W_SRAM_BASE: u32 = 0x0000_0000;
pub const W_SRAM_SIZE: u32 = 4 * 1024 * 1024;
pub const A_SRAM_BASE: u32 = 0x0040_0000;
pub const A_SRAM_SIZE: u32 = 1 * 1024 * 1024;
pub const STATE_SRAM_BASE: u32 = 0x0050_0000;
pub const STATE_SRAM_SIZE: u32 = 2 * 1024 * 1024;
pub const DRAM_BASE: u32 = 0x1000_0000;
pub const DRAM_SIZE: u32 = 512 * 1024 * 1024;

/// Build the four-region bus that matches `isa_spec.md` §0.5 for mamba-130m.
/// Regions are added in the order: W-SRAM, A-SRAM, STATE-SRAM, DRAM.
pub fn standard_bus() -> MemorySpace {
    let mut bus = MemorySpace::new();
    bus.add_named_memory(
        W_SRAM_BASE,
        W_SRAM_SIZE,
        "W-SRAM",
        Box::new(VecMemory::new(W_SRAM_SIZE as usize)),
    )
    .expect("W-SRAM");
    bus.add_named_memory(
        A_SRAM_BASE,
        A_SRAM_SIZE,
        "A-SRAM",
        Box::new(VecMemory::new(A_SRAM_SIZE as usize)),
    )
    .expect("A-SRAM");
    bus.add_named_memory(
        STATE_SRAM_BASE,
        STATE_SRAM_SIZE,
        "STATE-SRAM",
        Box::new(VecMemory::new(STATE_SRAM_SIZE as usize)),
    )
    .expect("STATE-SRAM");
    bus.add_named_memory(
        DRAM_BASE,
        DRAM_SIZE,
        "DRAM",
        Box::new(VecMemory::new(DRAM_SIZE as usize)),
    )
    .expect("DRAM");
    bus
}
