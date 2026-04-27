use crate::linear::LinearProjUnit;
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct LinearProjStage {
    formula: LinearProjUnit,
    slot: PipelineSlot,
}

impl LinearProjStage {
    pub fn new(mac_width: u32, pipeline_fill: u32) -> Self {
        Self {
            formula: LinearProjUnit::new(mac_width, pipeline_fill),
            slot: PipelineSlot::new(),
        }
    }
}

impl Tick for LinearProjStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for LinearProjStage {
    fn name(&self) -> &str {
        "LinearProj"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::LinearProj
    }
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let cycles = match op.params {
            OpParams::Linear { in_dim, out_dim } => self.formula.cycles_for(in_dim, out_dim),
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
