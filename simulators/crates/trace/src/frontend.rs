//! `OpTraceFrontend` — analog of ramulator2's `LoadStoreTrace` frontend
//! (`ramulator2/src/frontend/impl/processor/`). Reads a pre-parsed op list and
//! emits one op per `tick()` via the `DispatchHandle` set by `connect()`.
//!
//! Back-pressure: if the dispatcher returns `Err(op)`, the op is held and
//! retried next cycle. Mirrors ramulator2's `send()` returning false when a
//! controller's input buffer is full.

use crate::parser::{parse_trace, ParseError};
use emamba_core::{Cycle, Op, Tick};
use std::collections::VecDeque;

pub type DispatchHandle = Box<dyn FnMut(Op) -> Result<(), Op>>;

pub struct OpTraceFrontend {
    pending: VecDeque<Op>,
    dispatch: Option<DispatchHandle>,
    held: Option<Op>,
    finished: bool,
}

impl OpTraceFrontend {
    pub fn from_trace_str(trace: &str) -> Result<Self, ParseError> {
        let ops = parse_trace(trace)?;
        Ok(Self {
            pending: ops.into_iter().collect(),
            dispatch: None,
            held: None,
            finished: false,
        })
    }

    pub fn connect(&mut self, dispatch: DispatchHandle) {
        self.dispatch = Some(dispatch);
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn ops_remaining(&self) -> usize {
        self.pending.len() + self.held.as_ref().map(|_| 1).unwrap_or(0)
    }

    /// Pull-mode: take the next op directly. Used by `AcceleratorChip`'s
    /// run loop, which drives ops itself rather than via the
    /// `DispatchHandle` callback. Returns `None` when the queue is empty.
    pub fn try_next(&mut self) -> Option<Op> {
        self.held.take().or_else(|| self.pending.pop_front())
    }

    /// Pull-mode back-pressure: return an op so it's tried again next cycle.
    pub fn return_op(&mut self, op: Op) {
        self.held = Some(op);
    }
}

impl Tick for OpTraceFrontend {
    fn tick(&mut self, _now: Cycle) {
        if self.finished {
            return;
        }

        // Latch finished at the start of a tick when there's nothing left to
        // dispatch. That way the tick that drains the last op leaves the
        // frontend in `!is_finished()`, and the next tick flips the flag —
        // matching the contract in test 10.
        if self.pending.is_empty() && self.held.is_none() {
            self.finished = true;
            return;
        }

        let mut next_op = self.held.take().or_else(|| self.pending.pop_front());

        if let (Some(dispatch), Some(op)) = (self.dispatch.as_mut(), next_op.take()) {
            match dispatch(op) {
                Ok(()) => {}
                Err(returned) => {
                    self.held = Some(returned);
                }
            }
        } else if let Some(op) = next_op {
            // No dispatcher connected — hold so the test can observe.
            self.held = Some(op);
        }
    }
}
