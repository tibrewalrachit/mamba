//! `AcceleratorChip` — top-level wiring for the eMamba accelerator.
//!
//! Owns the sequencer, scratchpad, and (Phase 6) DRAM. `default_mars()` builds
//! the MARS-frame configuration from the paper §5: 10-unit range-norm,
//! 4-bank scratchpad, 32-wide MAC array, etc.
//!
//! `Simulation` is the runnable form: it pulls ops from the frontend, submits
//! to the sequencer, and ticks until both the frontend and the sequencer
//! drain. Mirrors the cycle loop in `ramulator2/src/main.cpp:81-112`.

use crate::scratchpad::BankedScratchpad;
use crate::sequencer::Sequencer;
use emamba_compute::units::{
    Conv1DStage, LinearProjStage, MemmoveStage, PwlExpStage, PwlSiluStage,
    RangeNormStage, ReluStage, ResidualAddStage, SsmOutputStage, SsmStateStage,
};
use emamba_core::{Cycle, Op, Tick};
use emamba_trace::frontend::OpTraceFrontend;

pub struct AcceleratorChip {
    sequencer: Sequencer,
    scratchpad: BankedScratchpad,
}

impl AcceleratorChip {
    /// Pre-wired MARS-frame chip: paper §5 defaults.
    pub fn default_mars() -> Self {
        let mut sequencer = Sequencer::new();
        sequencer.add_unit(Box::new(RangeNormStage::new(10, 4)));
        sequencer.add_unit(Box::new(Conv1DStage::new(4, 2)));
        sequencer.add_unit(Box::new(SsmStateStage::new(32, 1, 1)));
        sequencer.add_unit(Box::new(SsmOutputStage::new(32, 2)));
        sequencer.add_unit(Box::new(PwlSiluStage::new(17, 1)));
        sequencer.add_unit(Box::new(PwlExpStage::new(11, 1)));
        sequencer.add_unit(Box::new(ReluStage::new()));
        sequencer.add_unit(Box::new(LinearProjStage::new(64, 2)));
        sequencer.add_unit(Box::new(ResidualAddStage::new(16)));
        sequencer.add_unit(Box::new(MemmoveStage::new()));

        let scratchpad = BankedScratchpad::new(/* banks */ 4, /* bytes_per_bank */ 16384, /* port_width */ 16);

        Self {
            sequencer,
            scratchpad,
        }
    }

    pub fn sequencer(&self) -> &Sequencer {
        &self.sequencer
    }

    pub fn scratchpad(&self) -> &BankedScratchpad {
        &self.scratchpad
    }

    pub fn into_simulation(self, frontend: OpTraceFrontend) -> Simulation {
        Simulation::new(self, frontend, None)
    }

    pub fn into_simulation_with_op_decorator(
        self,
        frontend: OpTraceFrontend,
        decorator: Box<dyn FnMut(Op) -> Op>,
    ) -> Simulation {
        Simulation::new(self, frontend, Some(decorator))
    }
}

pub struct Simulation {
    chip: AcceleratorChip,
    frontend: OpTraceFrontend,
    now: Cycle,
    op_decorator: Option<Box<dyn FnMut(Op) -> Op>>,
    completed_at_close: u64,
}

impl Simulation {
    fn new(
        chip: AcceleratorChip,
        frontend: OpTraceFrontend,
        decorator: Option<Box<dyn FnMut(Op) -> Op>>,
    ) -> Self {
        Self {
            chip,
            frontend,
            now: 0,
            op_decorator: decorator,
            completed_at_close: 0,
        }
    }

    pub fn total_cycles(&self) -> Cycle {
        self.now
    }

    pub fn completed_ops(&self) -> u64 {
        self.chip.sequencer.completed_ops()
    }

    pub fn chip(&self) -> &AcceleratorChip {
        &self.chip
    }

    pub fn write_stats(
        &self,
        sink: &mut dyn emamba_core::stats::StatSink,
        trace_path: &str,
    ) {
        sink.record_u64(&["simulator", "total_cycles"], self.now);
        sink.record_u64(&["simulator", "ops_completed"], self.completed_ops());
        sink.record_str(&["simulator", "trace_path"], trace_path);
        self.chip.sequencer.write_stats(sink, self.now);
    }

    /// Drive the simulator until both the frontend and the sequencer drain.
    /// Conservative cycle bound prevents infinite loops on bad input.
    pub fn run_to_completion(&mut self) {
        const MAX_CYCLES: Cycle = 10_000_000;

        while self.now < MAX_CYCLES {
            // 1. Try to pull the next op from the frontend and submit it.
            if let Some(op) = self.frontend.try_next() {
                let decorated = match self.op_decorator.as_mut() {
                    Some(f) => f(op),
                    None => op,
                };
                match self.chip.sequencer.submit(decorated, self.now) {
                    Ok(()) => {}
                    Err(returned) => {
                        // Sequencer rejected — back-pressure, retry next cycle.
                        self.frontend.return_op(returned);
                    }
                }
            }

            // 2. Tick the chip's sequencer.
            self.chip.sequencer.tick(self.now);

            // 3. Termination: frontend drained AND no in-flight ops.
            let frontend_done = self.frontend.ops_remaining() == 0;
            let chip_done = self.chip.sequencer.all_idle(self.now);
            if frontend_done && chip_done {
                self.completed_at_close = self.chip.sequencer.completed_ops();
                return;
            }

            self.now += 1;
        }

        panic!(
            "simulation did not converge within {} cycles ({} ops completed)",
            MAX_CYCLES,
            self.chip.sequencer.completed_ops()
        );
    }
}
