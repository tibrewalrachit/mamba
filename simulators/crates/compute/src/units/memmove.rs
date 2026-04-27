//! `MemmoveStage` — LOAD/STORE handler.
//!
//! Computes transfer latency from the SimpleDram formula:
//!   cycles = dram_latency + ceil(bytes / bandwidth)
//! MemmoveStage has a single slot (no pipelining), so back-to-back
//! LOAD/STORE ops naturally serialize through it.

use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct MemmoveStage {
    slot: PipelineSlot,
    dram_bw_bytes_per_cycle: u32,
    dram_latency_cycles: u32,
}

impl MemmoveStage {
    pub fn new() -> Self {
        Self::with_dram(8, 80)
    }

    pub fn with_dram(bw: u32, lat: u32) -> Self {
        assert!(bw >= 1);
        Self {
            slot: PipelineSlot::new(),
            dram_bw_bytes_per_cycle: bw,
            dram_latency_cycles: lat,
        }
    }

    pub fn cycles_for(&self, bytes: u32) -> u32 {
        let bw = self.dram_bw_bytes_per_cycle as u64;
        let bw_cycles = (bytes as u64 + bw - 1) / bw;
        self.dram_latency_cycles as u64 as u32 + bw_cycles as u32
    }
}

impl Default for MemmoveStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Tick for MemmoveStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for MemmoveStage {
    fn name(&self) -> &str {
        "Memmove"
    }
    fn accepts(&self, op: &Op) -> bool {
        matches!(op.kind, OpKind::Load | OpKind::Store)
    }
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let cycles = match op.params {
            OpParams::Memmove { bytes } => self.cycles_for(bytes),
            _ => return Err(op),
        };
        self.slot.enqueue(op, now, cycles)
    }
    fn ready(&self, now: Cycle) -> bool {
        self.slot.ready(now)
    }
    fn drain(&mut self, now: Cycle) -> Option<Op> {
        self.slot.drain(now)
    }
    fn ops_completed(&self) -> u64 {
        self.slot.ops_completed()
    }
    fn busy_cycles(&self) -> u64 {
        self.slot.busy_cycles()
    }
}
