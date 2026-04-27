use crate::relu::ReluUnit;
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct ReluStage {
    formula: ReluUnit,
    slot: PipelineSlot,
}

impl ReluStage {
    pub fn new() -> Self {
        Self {
            formula: ReluUnit::new(),
            slot: PipelineSlot::new(),
        }
    }
}

impl Default for ReluStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Tick for ReluStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for ReluStage {
    fn name(&self) -> &str {
        "Relu"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::Relu
    }
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let cycles = match op.params {
            OpParams::Elementwise { d } => self.formula.cycles_for(d),
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
