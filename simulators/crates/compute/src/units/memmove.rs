//! `MemmoveStage` — placeholder LOAD/STORE handler. Phase 4's chip routes
//! LOAD and STORE here as 0-cycle ops; Phase 6 replaces this with actual DRAM
//! enqueue/drain coupling.

use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, Tick};

pub struct MemmoveStage {
    slot: PipelineSlot,
}

impl MemmoveStage {
    pub fn new() -> Self {
        Self {
            slot: PipelineSlot::new(),
        }
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
        // Placeholder — Phase 6 wires this to SimpleDram and inherits its
        // latency + bandwidth. For now, 1 cycle so the slot mechanism still
        // exercises the in-flight path.
        self.slot.enqueue(op, now, 1)
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
