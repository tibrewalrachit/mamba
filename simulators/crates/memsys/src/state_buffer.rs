//! `StateBuffer` — tracks SSM state liveness between SsmState and SsmOutput.
//!
//! SSM_STATE writes to the state buffer during execution; the slot is
//! held until the op completes so that a concurrent SSM_OUTPUT cannot
//! read a partially-updated state. The chip wires `lock()` / `release()`
//! around SsmStateStage completion.

use emamba_core::Cycle;

pub struct StateBuffer {
    capacity_bytes: u32,
    locked_until: Option<Cycle>,
}

impl StateBuffer {
    pub fn new(capacity_bytes: u32) -> Self {
        Self {
            capacity_bytes,
            locked_until: None,
        }
    }

    pub fn capacity_bytes(&self) -> u32 {
        self.capacity_bytes
    }

    /// Lock the buffer until `until` (exclusive). Overwrites any prior lock.
    pub fn lock(&mut self, until: Cycle) {
        self.locked_until = Some(until);
    }

    /// Returns true when no lock is held or the lock has expired.
    pub fn is_available(&self, now: Cycle) -> bool {
        match self.locked_until {
            None => true,
            Some(until) => now >= until,
        }
    }

    /// Explicitly release the lock (called on SSM_STATE completion).
    pub fn release(&mut self) {
        self.locked_until = None;
    }
}
