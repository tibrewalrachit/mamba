use crate::residual::ResidualAddUnit;
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct ResidualAddStage {
    formula: ResidualAddUnit,
    slot: PipelineSlot,
}

impl ResidualAddStage {
    pub fn new(adder_width: u32) -> Self {
        Self {
            formula: ResidualAddUnit::new(adder_width),
            slot: PipelineSlot::new(),
        }
    }
}

impl Tick for ResidualAddStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for ResidualAddStage {
    fn name(&self) -> &str {
        "ResidualAdd"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::ResidualAdd
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
