//! Tests 28-29 — `memsys::chip::*`
//! Drive a real trace through the AcceleratorChip and assert the cycle count
//! matches the formula.

use emamba_compute::range_norm::RangeNormUnit;
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
