//! emamba-memsys: on-chip memory + sequencer + accelerator chip top-level.

pub mod chip;
pub mod scratchpad;
pub mod sequencer;
pub mod state_buffer;

pub use chip::{AcceleratorChip, Simulation};
pub use scratchpad::BankedScratchpad;
pub use sequencer::Sequencer;
pub use state_buffer::StateBuffer;
