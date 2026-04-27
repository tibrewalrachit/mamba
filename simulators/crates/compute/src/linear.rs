//! `LinearProjUnit` — paper §4.6.
//! Matrix-vector multiply for the projection layers. Throughput is
//! `mac_width` MAC ops per cycle, plus a pipeline-fill epilogue.
//!
//! `cycles_for(in_dim, out_dim) = ceil(in_dim * out_dim / mac_width) + PIPELINE_FILL`

#[derive(Clone, Debug)]
pub struct LinearProjUnit {
    mac_width: u32,
    pipeline_fill: u32,
}

impl LinearProjUnit {
    pub fn new(mac_width: u32, pipeline_fill: u32) -> Self {
        assert!(mac_width >= 1, "mac_width must be >= 1");
        Self {
            mac_width,
            pipeline_fill,
        }
    }

    pub fn cycles_for(&self, in_dim: u32, out_dim: u32) -> u32 {
        let work = in_dim as u64 * out_dim as u64;
        let mac_cycles = (work + self.mac_width as u64 - 1) / self.mac_width as u64;
        mac_cycles as u32 + self.pipeline_fill
    }
}
