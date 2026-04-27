use crate::pwl::{PwlExpUnit, PwlSiluUnit};
use crate::unit::{ComputeUnit, PipelineSlot};
use emamba_core::{Cycle, Op, OpKind, OpParams, Tick};

macro_rules! pwl_stage {
    ($Stage:ident, $Formula:ident, $kind:expr, $name:expr) => {
        pub struct $Stage {
            formula: $Formula,
            slot: PipelineSlot,
        }
        impl $Stage {
            pub fn new(segments: u32, lut_latency: u32) -> Self {
                Self {
                    formula: $Formula::new(segments, lut_latency),
                    slot: PipelineSlot::new(),
                }
            }
        }
        impl Tick for $Stage {
            fn tick(&mut self, now: Cycle) {
                self.slot.tick(now);
            }
        }
        impl ComputeUnit for $Stage {
            fn name(&self) -> &str {
                $name
            }
            fn accepts(&self, op: &Op) -> bool {
                op.kind == $kind
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
    };
}

pwl_stage!(PwlSiluStage, PwlSiluUnit, OpKind::PwlSilu, "PwlSilu");
pwl_stage!(PwlExpStage, PwlExpUnit, OpKind::PwlExp, "PwlExp");
