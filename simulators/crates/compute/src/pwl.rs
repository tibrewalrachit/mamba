//! Piecewise-linear approximation units — paper §4.5 (and the Python
//! reference in `approximations.py`).
//!
//! SiLU: 17 segments in [-7, 7]; below -7 → 0, above 7 → identity.
//! Exp:  11 segments in [-4, 1]; below -4 → 0, above 1 → e¹.
//!
//! Hardware is a small LUT + multiplier + adder, fully pipelined: one element
//! per cycle once the lut_latency-1 fill cycles are paid.
//!
//! `cycles_for(D) = D + (lut_latency - 1)`

#[derive(Clone, Debug)]
pub struct PwlSiluUnit {
    segments: u32,
    lut_latency: u32,
}

#[derive(Clone, Debug)]
pub struct PwlExpUnit {
    segments: u32,
    lut_latency: u32,
}

impl PwlSiluUnit {
    pub fn new(segments: u32, lut_latency: u32) -> Self {
        assert!(lut_latency >= 1, "lut_latency must be >= 1");
        Self {
            segments,
            lut_latency,
        }
    }

    pub fn segments(&self) -> u32 {
        self.segments
    }

    pub fn cycles_for(&self, d: u32) -> u32 {
        d + (self.lut_latency - 1)
    }
}

impl PwlExpUnit {
    pub fn new(segments: u32, lut_latency: u32) -> Self {
        assert!(lut_latency >= 1, "lut_latency must be >= 1");
        Self {
            segments,
            lut_latency,
        }
    }

    pub fn segments(&self) -> u32 {
        self.segments
    }

    pub fn cycles_for(&self, d: u32) -> u32 {
        d + (self.lut_latency - 1)
    }
}
