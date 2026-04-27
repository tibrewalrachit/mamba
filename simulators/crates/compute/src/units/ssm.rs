use crate::ssm::{SsmOutputUnit, SsmStateUnit};
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

pub struct SsmStateStage {
    formula: SsmStateUnit,
    slot: PipelineSlot,
}

pub struct SsmOutputStage {
    formula: SsmOutputUnit,
    slot: PipelineSlot,
}

impl SsmStateStage {
    pub fn new(mac: u32, read: u32, write: u32) -> Self {
        Self {
            formula: SsmStateUnit::new(mac, read, write),
            slot: PipelineSlot::new(),
        }
    }
}
impl SsmOutputStage {
    pub fn new(mac: u32, fill: u32) -> Self {
        Self {
            formula: SsmOutputUnit::new(mac, fill),
            slot: PipelineSlot::new(),
        }
    }
}

impl Tick for SsmStateStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}
impl Tick for SsmOutputStage {
    fn tick(&mut self, now: Cycle) {
        self.slot.tick(now);
    }
}

impl ComputeUnit for SsmStateStage {
    fn name(&self) -> &str {
        "SsmState"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::SsmStateUpdate
    }
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let cycles = match op.params {
            OpParams::Ssm { d, n, e } => self.formula.cycles_for(d, n, e),
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

impl ComputeUnit for SsmOutputStage {
    fn name(&self) -> &str {
        "SsmOutput"
    }
    fn accepts(&self, op: &Op) -> bool {
        op.kind == OpKind::SsmOutput
    }
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let cycles = match op.params {
            OpParams::Ssm { d, n, e } => self.formula.cycles_for(d, n, e),
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
