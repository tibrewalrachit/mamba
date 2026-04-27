//! `BankedScratchpad` — on-chip SRAM with per-bank read + write ports.
//!
//! Mirrors the role of ramulator2's per-bank state in DDR4
//! (`ramulator2/src/dram/impl/DDR4.cpp`), but flatter: no row state, just
//! port-busy-until cycles. A request consumes `ceil(bytes / port_width)` cycles
//! and is rejected if the port is still busy at `now`.
//!
//! Read and write ports are independent: a write at t=0 followed by a read on
//! the same bank at t=1 succeeds because the read port wasn't busy.

use emamba_core::Cycle;

#[derive(Clone, Debug)]
pub struct BankedScratchpad {
    banks: u32,
    bytes_per_bank: u64,
    port_width_bytes: u32,
    read_busy_until: Vec<Cycle>,
    write_busy_until: Vec<Cycle>,
}

impl BankedScratchpad {
    pub fn new(banks: u32, bytes_per_bank: u64, port_width_bytes: u32) -> Self {
        assert!(banks >= 1);
        assert!(port_width_bytes >= 1);
        Self {
            banks,
            bytes_per_bank,
            port_width_bytes,
            read_busy_until: vec![0; banks as usize],
            write_busy_until: vec![0; banks as usize],
        }
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.bytes_per_bank * self.banks as u64
    }

    pub fn banks(&self) -> u32 {
        self.banks
    }

    pub fn try_read(&mut self, bank: u32, bytes: u32, now: Cycle) -> Option<Cycle> {
        let idx = self.bank_index(bank);
        if self.read_busy_until[idx] > now {
            return None;
        }
        let done = now + self.cycles_for_bytes(bytes);
        self.read_busy_until[idx] = done;
        Some(done)
    }

    pub fn try_write(&mut self, bank: u32, bytes: u32, now: Cycle) -> Option<Cycle> {
        let idx = self.bank_index(bank);
        if self.write_busy_until[idx] > now {
            return None;
        }
        let done = now + self.cycles_for_bytes(bytes);
        self.write_busy_until[idx] = done;
        Some(done)
    }

    fn cycles_for_bytes(&self, bytes: u32) -> Cycle {
        let pw = self.port_width_bytes as u64;
        ((bytes as u64 + pw - 1) / pw) as Cycle
    }

    fn bank_index(&self, bank: u32) -> usize {
        assert!(
            bank < self.banks,
            "bank {} out of range (banks={})",
            bank,
            self.banks
        );
        bank as usize
    }
}
