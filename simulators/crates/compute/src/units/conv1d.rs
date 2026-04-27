use crate::conv1d::Conv1DUnit;
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct Conv1DStage {
    formula: Conv1DUnit,
    slot: PipelineSlot,
}

impl Conv1DStage {
    pub fn new(kernel_size: u32, pipeline_fill: u32) -> Self {
        Self {
            formula: Conv1DUnit::new(kernel_size, pipeline_fill),
            slot: PipelineSlot::new(),
        }
    }
}

impl Tick for Conv1DStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for Conv1DStage {
    fn name(&self) -> &str {
        "Conv1D"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::Conv1D
    }
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let cycles = match op.params {
            OpParams::Conv1D { d, .. } => self.formula.cycles_for(d),
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
