//! `ReluUnit` — paper §4.5.
//! Replaces softplus in the SSM discretization. A single-cycle comparator,
//! fully pipelined: cycles_for(D) = D.

#[derive(Clone, Debug, Default)]
pub struct ReluUnit;

impl ReluUnit {
    pub fn new() -> Self {
        Self
    }

    pub fn cycles_for(&self, d: u32) -> u32 {
        d
    }
}
