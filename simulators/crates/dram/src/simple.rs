//! `SimpleDram` — v1 off-chip memory model.
//!
//! Each enqueued request is a (bytes, done_by) entry; `drain(now)` returns the
//! ids of every request whose `done_by <= now`. No banking, no row buffer —
//! pure latency + bandwidth. This is the swap-point for a future DDR4 model
//! (see plan §10 open-question 1).

use emamba_core::Cycle;
use std::collections::VecDeque;

#[derive(Debug)]
pub enum DramError {
    QueueFull,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    id: u64,
    done_by: Cycle,
}

#[derive(Debug)]
pub struct SimpleDram {
    bandwidth_bytes_per_cycle: u32,
    latency_cycles: u32,
    queue_depth: u32,
    in_flight: VecDeque<Entry>,
    next_id: u64,
}

impl SimpleDram {
    pub fn new(bandwidth_bytes_per_cycle: u32, latency_cycles: u32, queue_depth: u32) -> Self {
        assert!(bandwidth_bytes_per_cycle >= 1);
        Self {
            bandwidth_bytes_per_cycle,
            latency_cycles,
            queue_depth,
            in_flight: VecDeque::new(),
            next_id: 0,
        }
    }

    pub fn occupancy(&self) -> u32 {
        self.in_flight.len() as u32
    }

    pub fn enqueue(&mut self, bytes: u32, now: Cycle) -> Result<u64, DramError> {
        if self.in_flight.len() as u32 >= self.queue_depth {
            return Err(DramError::QueueFull);
        }
        let bw = self.bandwidth_bytes_per_cycle as u64;
        let bw_cycles = (bytes as u64 + bw - 1) / bw;
        let done_by = now + self.latency_cycles as Cycle + bw_cycles;
        let id = self.next_id;
        self.next_id += 1;
        self.in_flight.push_back(Entry { id, done_by });
        Ok(id)
    }

    pub fn drain(&mut self, now: Cycle) -> Vec<u64> {
        let mut out = Vec::new();
        while let Some(front) = self.in_flight.front() {
            if front.done_by > now {
                break;
            }
            out.push(front.id);
            self.in_flight.pop_front();
        }
        out
    }
}
