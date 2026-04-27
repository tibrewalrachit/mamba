//! `ComputeUnit` trait + `PipelineSlot` helper.
//!
//! Mirrors ramulator2's `IDramController` interface. Each compute stage owns
//! one `PipelineSlot` that holds the currently in-flight op and its
//! `done_by` cycle. `tick(now)` completes the op once `now >= done_by`,
//! making it visible to `drain(now)`.
//!
//! This is intentionally simple — one op per stage, no internal pipelining
//! beyond the cycles_for formula. The accelerator's per-token pipeline is
//! achieved at the *sequencer* level, by overlapping different stages.

use emamba_core::{Cycle, Op, OpKind, Tick};

pub trait ComputeUnit: Tick {
    fn name(&self) -> &str;

    /// True iff this unit is the one that should service this op kind.
    fn accepts(&self, op: &Op) -> bool;

    /// Enqueue an op for execution. Err returns the op if the unit is busy.
    fn enqueue(&mut self, op: Op, now: Cycle) -> Result<(), Op>;

    /// True iff a new op can be enqueued at `now` (no in-flight op).
    fn ready(&self, now: Cycle) -> bool;

    /// Pop any ops whose `done_by <= now`. At most one per call (single-slot).
    fn drain(&mut self, now: Cycle) -> Option<Op>;

    /// Stats accessors.
    fn ops_completed(&self) -> u64;
    fn busy_cycles(&self) -> u64;
}

/// State shared across all stage implementations.
pub struct PipelineSlot {
    in_flight: Option<(Op, Cycle, Cycle)>, // (op, started_at, done_by)
    ops_completed: u64,
    busy_cycles: u64,
    last_tick: Cycle,
}

impl PipelineSlot {
    pub fn new() -> Self {
        Self {
            in_flight: None,
            ops_completed: 0,
            busy_cycles: 0,
            last_tick: 0,
        }
    }

    pub fn ready(&self, _now: Cycle) -> bool {
        self.in_flight.is_none()
    }

    pub fn enqueue(&mut self, op: Op, now: Cycle, cycles: u32) -> Result<(), Op> {
        if self.in_flight.is_some() {
            return Err(op);
        }
        self.in_flight = Some((op, now, now + cycles as Cycle));
        Ok(())
    }

    pub fn tick(&mut self, now: Cycle) {
        if let Some((_, started_at, _)) = self.in_flight.as_ref() {
            // Accumulate busy cycles for the slice from last_tick → now.
            let prev = self.last_tick.max(*started_at);
            if now > prev {
                self.busy_cycles += now - prev;
            }
        }
        self.last_tick = now;
    }

    pub fn drain(&mut self, now: Cycle) -> Option<Op> {
        let should_pop = matches!(&self.in_flight, Some((_, _, done_by)) if *done_by <= now);
        if should_pop {
            let (op, _, _) = self.in_flight.take().expect("checked above");
            self.ops_completed += 1;
            Some(op)
        } else {
            None
        }
    }

    pub fn ops_completed(&self) -> u64 {
        self.ops_completed
    }

    pub fn busy_cycles(&self) -> u64 {
        self.busy_cycles
    }
}

/// Convenience: dispatch an op kind to its expected unit category — used by
/// `Sequencer` to route. Each unit's `accepts()` is the source of truth.
pub fn unit_kind_label(kind: OpKind) -> &'static str {
    match kind {
        OpKind::RangeNorm => "RangeNorm",
        OpKind::Conv1D => "Conv1D",
        OpKind::SsmStateUpdate => "SsmState",
        OpKind::SsmOutput => "SsmOutput",
        OpKind::PwlSilu => "PwlSilu",
        OpKind::PwlExp => "PwlExp",
        OpKind::Relu => "Relu",
        OpKind::LinearProj => "LinearProj",
        OpKind::ResidualAdd => "ResidualAdd",
        OpKind::Load | OpKind::Store => "MemMove",
    }
}
