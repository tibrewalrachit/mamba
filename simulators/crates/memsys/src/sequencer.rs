//! `Sequencer` — routes ops to compute units, manages ready/valid handshake.
//!
//! Mirrors ramulator2's `IDramController::send` + scheduler interaction. The
//! frontend `submit`s ops; the sequencer dispatches to whichever unit's
//! `accepts()` returns true, bouncing the op back as `Err` if the target is
//! not `ready()` (back-pressure).
//!
//! Per-tick:
//!   1. drain finished ops from each unit, fire callbacks, count completed.
//!   2. tick each unit to advance their internal state.

use emamba_compute::unit::ComputeUnit;
use emamba_core::{Cycle, Op, Tick};

pub struct Sequencer {
    units: Vec<Box<dyn ComputeUnit>>,
    completed: u64,
}

impl Sequencer {
    pub fn new() -> Self {
        Self {
            units: Vec::new(),
            completed: 0,
        }
    }

    pub fn add_unit(&mut self, unit: Box<dyn ComputeUnit>) {
        self.units.push(unit);
    }

    pub fn submit(&mut self, op: Op, now: Cycle) -> Result<(), Op> {
        let mut op = op;
        // Find the first unit that accepts this op.
        for u in self.units.iter_mut() {
            if u.accepts(&op) {
                if !u.ready(now) {
                    return Err(op);
                }
                return u.enqueue(op, now);
            }
        }
        // No unit claimed it.
        op.scratch[3] = -1; // hint for debugging
        Err(op)
    }

    pub fn completed_ops(&self) -> u64 {
        self.completed
    }

    pub fn unit_by_name(&self, name: &str) -> Option<&dyn ComputeUnit> {
        self.units
            .iter()
            .find(|u| u.name() == name)
            .map(|b| &**b)
    }

    pub fn units(&self) -> impl Iterator<Item = &dyn ComputeUnit> {
        self.units.iter().map(|b| &**b)
    }

    pub fn all_idle(&self, now: Cycle) -> bool {
        self.units.iter().all(|u| u.ready(now))
    }

    pub fn write_stats(&self, sink: &mut dyn emamba_core::stats::StatSink, total_cycles: Cycle) {
        for u in &self.units {
            let name = u.name();
            sink.record_u64(&["units", name, "ops"], u.ops_completed());
            sink.record_u64(&["units", name, "busy_cycles"], u.busy_cycles());
            sink.record_u64(
                &["units", name, "idle_cycles"],
                total_cycles.saturating_sub(u.busy_cycles()),
            );
        }
    }
}

impl Default for Sequencer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tick for Sequencer {
    fn tick(&mut self, now: Cycle) {
        for u in self.units.iter_mut() {
            u.tick(now);
            if let Some(mut op) = u.drain(now) {
                self.completed += 1;
                if let Some(cb) = op.callback.take() {
                    cb(&op);
                }
            }
        }
    }
}
