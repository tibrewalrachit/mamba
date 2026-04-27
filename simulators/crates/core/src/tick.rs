//! Cycle-driven simulation primitives. Mirrors ramulator2's `Clocked<T>` (see
//! `ramulator2_rust_design.md` §7) — a Tick is a per-cycle hook; the caller
//! advances `now`. Per-component `clock_ratio()` lets the top-level loop
//! synchronize sub-domains via lcm/step (RD §7).

pub type Cycle = u64;

pub trait Tick {
    fn tick(&mut self, now: Cycle);

    fn clock_ratio(&self) -> u32 {
        1
    }
}
