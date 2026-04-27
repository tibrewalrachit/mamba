//! Test 10 — `trace::frontend::tick_emits_one_op_per_cycle`
//! `OpTraceFrontend` initialized with a single-op trace emits exactly one Op
//! on its first tick, sets `is_finished()` true on the second tick, and the
//! emitted Op is routed through the `DispatchHandle` configured via `connect`.

use emamba_core::{Op, Tick};
use emamba_trace::frontend::OpTraceFrontend;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn tick_emits_one_op_per_cycle() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8\n";
    let mut frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let captured: Rc<RefCell<Vec<Op>>> = Rc::new(RefCell::new(Vec::new()));
    let cap = captured.clone();
    frontend.connect(Box::new(move |op| {
        cap.borrow_mut().push(op);
        Ok(())
    }));

    assert!(!frontend.is_finished(), "fresh frontend not yet finished");

    frontend.tick(0);
    assert_eq!(captured.borrow().len(), 1, "first tick emits one op");
    assert!(
        !frontend.is_finished(),
        "still not finished after emitting last op — finish flag flips on next tick"
    );

    frontend.tick(1);
    assert_eq!(captured.borrow().len(), 1, "no more ops dispatched");
    assert!(frontend.is_finished(), "is_finished after exhausted");
}

#[test]
fn back_pressure_holds_op_until_dispatch_succeeds() {
    let trace = "\
RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8
RANGE_NORM in=0x3000 in_mem=sram out=0x4000 out_mem=sram D=20 dtype=int8
";
    let mut frontend = OpTraceFrontend::from_trace_str(trace).expect("parse");

    let captured: Rc<RefCell<Vec<Op>>> = Rc::new(RefCell::new(Vec::new()));
    let counter: Rc<RefCell<u32>> = Rc::new(RefCell::new(0));
    let cap = captured.clone();
    let cnt = counter.clone();
    frontend.connect(Box::new(move |op| {
        let mut n = cnt.borrow_mut();
        *n += 1;
        if *n == 2 {
            // Reject the second dispatch attempt to simulate downstream busy.
            return Err(op);
        }
        cap.borrow_mut().push(op);
        Ok(())
    }));

    frontend.tick(0); // emits op 1
    assert_eq!(captured.borrow().len(), 1);
    assert_eq!(*counter.borrow(), 1);

    frontend.tick(1); // tries op 2, rejected
    assert_eq!(captured.borrow().len(), 1, "op 2 was rejected");
    assert_eq!(*counter.borrow(), 2);
    assert!(!frontend.is_finished(), "op 2 still pending");

    frontend.tick(2); // retry succeeds
    assert_eq!(captured.borrow().len(), 2);
    assert_eq!(*counter.borrow(), 3);
}
