//! 32-bit RISC-V R-type instruction decoder for the Xemamba custom-0 extension.
//!
//! See `isa_spec.md` §0.4 for the bit layout and §0.6 for the funct3/funct7
//! → mnemonic table.

/// RISC-V `custom-0` major opcode value, per the RV base opcode map
/// (Unprivileged ISA Manual, "RV32/64G Instruction Set Listings"):
/// `inst[6:5]=00`, `inst[4:2]=010`, `inst[1:0]=11`.
pub const OPCODE_CUSTOM_0: u32 = 0b000_1011;

/// Minor-opcode group used by every Xemamba v0.2 instruction (`funct3 = 0`).
pub const FUNCT3_XEMAMBA_V0_2: u32 = 0b000;

/// Per-op `funct7` selectors (spec §0.6).
pub mod funct7 {
    pub const MAC: u32 = 0x01;
    pub const NORM: u32 = 0x02;
    pub const CONV: u32 = 0x03;
    pub const SSM: u32 = 0x04;
    pub const SILU: u32 = 0x05;
    pub const EXP: u32 = 0x06;
    pub const DMAL: u32 = 0x07;
    pub const DMAS: u32 = 0x08;
}

/// Decoded R-type instruction word — five register/funct fields plus the
/// 7-bit major opcode. No semantic interpretation; matches the bit layout
/// in `isa_spec.md` §0.4.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RType {
    pub funct7: u32,
    pub rs2: u32,
    pub rs1: u32,
    pub funct3: u32,
    pub rd: u32,
    pub opcode: u32,
}

impl RType {
    /// Decode a 32-bit instruction word into its six R-type fields.
    pub fn new(insn: u32) -> Self {
        Self {
            funct7: (insn >> 25) & 0x7f,
            rs2: (insn >> 20) & 0x1f,
            rs1: (insn >> 15) & 0x1f,
            funct3: (insn >> 12) & 0x07,
            rd: (insn >> 7) & 0x1f,
            opcode: insn & 0x7f,
        }
    }

    /// Round-trip helper: re-encode the fields into a 32-bit word.
    /// Used by tests and the disassembler.
    pub fn encode(self) -> u32 {
        ((self.funct7 & 0x7f) << 25)
            | ((self.rs2 & 0x1f) << 20)
            | ((self.rs1 & 0x1f) << 15)
            | ((self.funct3 & 0x07) << 12)
            | ((self.rd & 0x1f) << 7)
            | (self.opcode & 0x7f)
    }

    /// True if the major opcode is `custom-0`. The dispatcher should
    /// reject (with `IllegalOpcode`) any instruction for which this is false.
    pub fn is_xemamba(self) -> bool {
        self.opcode == OPCODE_CUSTOM_0
    }
}
