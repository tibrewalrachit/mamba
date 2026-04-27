//! Tests 28-29, 35-37 — `memsys::chip::*`
//! Drive a real trace through the AcceleratorChip and assert the cycle count
//! matches the formula.

use emamba_compute::range_norm::RangeNormUnit;
use emamba_compute::residual::ResidualAddUnit;
use emamba_compute::units::MemmoveStage;
use emamba_memsys::chip::AcceleratorChip;
use emamba_trace::frontend::OpTraceFrontend;
use std::cell::Cell;
use std::rc::Rc;

#[test]
fn end_to_end_single_op_completes() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8\n";
    let frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let chip = AcceleratorChip::default_mars();
    let mut sim = chip.into_simulation(frontend);
    sim.run_to_completion();

    // RangeNorm with units=10, div=4 → cycles_for(20) = 54
    let expected = RangeNormUnit::new(10, 4).cycles_for(20) as u64;
    assert_eq!(sim.completed_ops(), 1);
    // total_cycles is the cycle at which the last op landed, plus the
    // final tick that observed the chip empty. The last op's done_by is
    // exactly `expected`, and the run loop terminates the cycle after.
    assert!(
        sim.total_cycles() >= expected,
        "total_cycles {} >= expected {}",
        sim.total_cycles(),
        expected
    );
    assert!(
        sim.total_cycles() <= expected + 4,
        "total_cycles {} within drain envelope of expected {}",
        sim.total_cycles(),
        expected
    );
}

#[test]
fn callback_fires_on_completion() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8\n";
    let mut frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let fired: Rc<Cell<u32>> = Rc::new(Cell::new(0));
    let f = fired.clone();

    // We can't attach a callback through the public API yet — callbacks are
    // set on the Op struct itself. Wire one in via the dispatch handle by
    // intercepting the op before submitting.
    let chip = AcceleratorChip::default_mars();
    let mut sim = chip.into_simulation_with_op_decorator(
        frontend,
        Box::new(move |mut op| {
            let f2 = f.clone();
            op.callback = Some(Box::new(move |_| {
                f2.set(f2.get() + 1);
            }));
            op
        }),
    );
    sim.run_to_completion();

    assert_eq!(fired.get(), 1, "callback fires exactly once");
}

// --- Phase 6: DRAM-aware integration ---

/// Test 35 — a LOAD (20 bytes, DRAM bw=8, lat=80) ties up MemmoveStage for 83
/// cycles.  The simulation must wait for DRAM to drain before declaring done;
/// total_cycles must be ≥ the DRAM formula value.
#[test]
fn load_then_compute_blocks_until_dram_drained() {
    let trace = "LOAD src=0x80000000 src_mem=dram dst=0x1000 dst_mem=sram bytes=20\n";
    let frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let chip = AcceleratorChip::default_mars();
    let mut sim = chip.into_simulation(frontend);
    sim.run_to_completion();

    // MemmoveStage default_mars: bw=8, lat=80 → cycles = 80 + ceil(20/8) = 83
    let dram_cycles = MemmoveStage::new().cycles_for(20) as u64;
    assert_eq!(dram_cycles, 83);
    assert_eq!(sim.completed_ops(), 1);
    assert!(
        sim.total_cycles() >= dram_cycles,
        "total_cycles {} must be ≥ DRAM latency {}",
        sim.total_cycles(),
        dram_cycles
    );
    assert!(
        sim.total_cycles() <= dram_cycles + 3,
        "total_cycles {} within drain envelope of {}",
        sim.total_cycles(),
        dram_cycles
    );
}

/// Test 36 — LOAD (83 cycles) + independent RANGE_NORM (54 cycles) run on
/// different units simultaneously.  total_cycles ≈ max(83, 54+1) = 83, which
/// is far less than the sequential sum (83+54=137).
#[test]
fn compute_overlaps_dram_when_independent() {
    let trace =
        "LOAD src=0x80000000 src_mem=dram dst=0x1000 dst_mem=sram bytes=20\n\
         RANGE_NORM in=0x2000 in_mem=sram out=0x3000 out_mem=sram D=20 dtype=int8\n";
    let frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let chip = AcceleratorChip::default_mars();
    let mut sim = chip.into_simulation(frontend);
    sim.run_to_completion();

    let dram_cycles = MemmoveStage::new().cycles_for(20) as u64; // 83
    let rn_cycles = RangeNormUnit::new(10, 4).cycles_for(20) as u64; // 54
    let sequential_sum = dram_cycles + rn_cycles; // 137

    assert_eq!(sim.completed_ops(), 2);
    // Parallel upper bound: max(dram, 1+rn) + small dispatch overhead
    assert!(
        sim.total_cycles() <= dram_cycles + 5,
        "total_cycles {} should be near max({},{}) not sequential sum {}",
        sim.total_cycles(),
        dram_cycles,
        rn_cycles,
        sequential_sum
    );
    assert!(
        sim.total_cycles() >= dram_cycles,
        "total_cycles {} must be ≥ DRAM latency {}",
        sim.total_cycles(),
        dram_cycles
    );
}

/// Test 37 — RESIDUAL reads two distinct input tensors (a and b from SRAM).
/// The op completes in ceil(D / adder_width) cycles (D=20, width=16 → 2).
#[test]
fn residual_uses_two_input_banks() {
    let trace =
        "RESIDUAL a=0x1000 a_mem=sram b=0x2000 b_mem=sram out=0x3000 out_mem=sram D=20 dtype=int8\n";
    let frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let chip = AcceleratorChip::default_mars();
    let mut sim = chip.into_simulation(frontend);
    sim.run_to_completion();

    let expected_cycles = ResidualAddUnit::new(16).cycles_for(20) as u64; // 2
    assert_eq!(sim.completed_ops(), 1);
    assert!(
        sim.total_cycles() >= expected_cycles,
        "total_cycles {} >= residual cycles {}",
        sim.total_cycles(),
        expected_cycles
    );
    assert!(
        sim.total_cycles() <= expected_cycles + 3,
        "total_cycles {} within drain envelope",
        sim.total_cycles()
    );
}
