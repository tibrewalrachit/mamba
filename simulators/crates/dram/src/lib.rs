//! emamba-dram: off-chip DRAM model.
//!
//! Plan §10 open-question 1: v1 ships a `SimpleDram` with fixed latency +
//! bandwidth, queue-bounded back-pressure. The `Dram` trait is shaped so a
//! future Ddr4Dram (or a ramulator2 FFI shim) drops in without touching
//! `memsys`. RD §11 split between `&self` queries and `&mut self` updates so
//! a v1.5 `par_iter_mut` per-channel parallelism is a swap, not a refactor.

pub mod simple;

pub use simple::SimpleDram;
