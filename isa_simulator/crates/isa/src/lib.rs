//! emamba-isa — Xemamba RISC-V custom-0 extension functional simulator library.
//!
//! Implements the encoding and dispatch layer of the Xemamba ISA spec
//! (`isa_simulator/isa_spec.md` v0.2). Decoded instructions are R-type words
//! under the RISC-V `custom-0` opcode (`0b0001011`); a `funct3 / funct7` pair
//! selects the operation, and `rs1` carries a host-memory pointer to a
//! per-op descriptor (see spec §0.4–§0.7).

pub mod desc;
pub mod insn;
