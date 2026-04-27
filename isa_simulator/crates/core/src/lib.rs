//! emamba-core: domain-free primitives shared by every other crate.
//!
//! Today this is just `Dtype`. The ISA spec (`isa_spec.md` §0.3) defines the
//! 3-bit dtype field; this enum is its Rust mirror.

/// Element type tag carried in the `dtype` field of every compute op.
///
/// Discriminant values match the on-the-wire 3-bit code from `isa_spec.md` §0.3:
/// FP32 = 0b000, INT8 = 0b001, INT24 = 0b010.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    Fp32 = 0b000,
    Int8 = 0b001,
    Int24 = 0b010,
}

impl Dtype {
    /// Bytes per element. Matches the table in `isa_spec.md` §0.3.
    pub fn bytes(self) -> usize {
        match self {
            Self::Int8 => 1,
            Self::Int24 => 3,
            Self::Fp32 => 4,
        }
    }

    /// Decode a 3-bit dtype field. Returns `None` for reserved codes
    /// (the executor turns this into `OpException::IllegalDtype`).
    pub fn from_bits(bits: u8) -> Option<Self> {
        match bits & 0b111 {
            0b000 => Some(Self::Fp32),
            0b001 => Some(Self::Int8),
            0b010 => Some(Self::Int24),
            _ => None,
        }
    }

    /// Encode as a 3-bit field for op construction / disassembly.
    pub fn to_bits(self) -> u8 {
        self as u8
    }
}
