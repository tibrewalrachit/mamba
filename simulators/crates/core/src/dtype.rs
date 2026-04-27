//! Element dtypes used by the eMamba accelerator.
//!
//! Cycle counting only depends on byte-width. Real arithmetic is out of scope.
//! Int24 is the SSM-state bitwidth (paper §4.6).

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Dtype {
    Int8,
    Int24,
    Fp32,
}

impl Dtype {
    pub const fn bytes(self) -> u32 {
        match self {
            Dtype::Int8 => 1,
            Dtype::Int24 => 3,
            Dtype::Fp32 => 4,
        }
    }
}
