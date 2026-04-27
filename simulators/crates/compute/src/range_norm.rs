//! `RangeNormUnit` — paper §4.1, §5.3.
//!
//! Range normalization replaces LayerNorm: instead of computing variance, the
//! hardware compares input min/max. Latency is dominated by `25 * D` cycles
//! per element, divided across `compute_units` parallel min/max comparators,
//! plus a final fixed-cost division.
//!
//! `cycles_for(D) = ceil(25 * D / units) + DIV_LATENCY`
//!
//! `compute_units = 10` is the paper's MARS-frame configuration; varying it is
//! the area/latency knob from Figure 8.

#[derive(Clone, Debug)]
pub struct RangeNormUnit {
    units: u32,
    div_latency: u32,
}

impl RangeNormUnit {
    pub fn new(units: u32, div_latency: u32) -> Self {
        assert!(units >= 1, "RangeNormUnit needs at least one compute unit");
        Self { units, div_latency }
    }

    pub fn cycles_for(&self, d: u32) -> u32 {
        let raw = 25u64 * d as u64;
        let parallel = (raw + self.units as u64 - 1) / self.units as u64; // ceil-div
        parallel as u32 + self.div_latency
    }

    pub fn units(&self) -> u32 {
        self.units
    }
}
