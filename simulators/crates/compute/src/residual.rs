//! `ResidualAddUnit` — paper §4.4.
//! Element-wise adder for the residual skip connection.
//!
//! `cycles_for(D) = ceil(D / adder_width)`

#[derive(Clone, Debug)]
pub struct ResidualAddUnit {
    adder_width: u32,
}

impl ResidualAddUnit {
    pub fn new(adder_width: u32) -> Self {
        assert!(adder_width >= 1, "adder_width must be >= 1");
        Self { adder_width }
    }

    pub fn cycles_for(&self, d: u32) -> u32 {
        (d + self.adder_width - 1) / self.adder_width
    }
}
