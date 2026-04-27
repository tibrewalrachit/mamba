//! Test 3 — `core::tick::null_tick_advances_clock`
//! The Tick trait does not own time; the caller passes `now`. A null impl
//! must accept ticks at any cycle and not blow up. Default clock_ratio == 1.

use emamba_core::{Cycle, Tick};

struct Null {
    last_seen: Cycle,
}

impl Tick for Null {
    fn tick(&mut self, now: Cycle) {
        self.last_seen = now;
    }
}

#[test]
fn null_tick_advances_clock() {
    let mut n = Null { last_seen: 0 };
    for cycle in 0..5 {
        n.tick(cycle);
    }
    assert_eq!(n.last_seen, 4);
}

#[test]
fn default_clock_ratio_is_one() {
    let n = Null { last_seen: 0 };
    assert_eq!(n.clock_ratio(), 1);
}
