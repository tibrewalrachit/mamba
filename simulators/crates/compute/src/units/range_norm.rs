use crate::range_norm::RangeNormUnit;
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct RangeNormStage {
    formula: RangeNormUnit,
    slot: PipelineSlot,
}

impl RangeNormStage {
    pub fn new(units: u32, div_latency: u32) -> Self {
        Self {
            formula: RangeNormUnit::new(units, div_latency),
            slot: PipelineSlot::new(),
        }
    }
}

impl Tick for RangeNormStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for RangeNormStage {
    fn name(&self) -> &str {
        "RangeNorm"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::RangeNorm
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
