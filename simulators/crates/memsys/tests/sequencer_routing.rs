//! Tests 25-27 — `memsys::sequencer::*`
//! Sequencer takes ops from the frontend and routes each to the unit whose
//! `accepts()` returns true. Holds an op when the target unit is busy
//! (back-pressure). Pipelines independent ops through different units.

use emamba_compute::units::{Conv1DStage, RangeNormStage};
use emamba_compute::unit::ComputeUnit;
use emamba_core::{Op, OpKind, OpParams, Tick};
use emamba_memsys::sequencer::Sequencer;

fn make_op(id: u64, kind: OpKind, params: OpParams) -> Op {
    Op {
        id,
        layer: 0,
        token: 0,
        kind,
        inputs: vec![],
        outputs: vec![],
        params,
        scratch: [0; 4],
        callback: None,
    }
}

#[test]
fn routes_op_to_correct_unit() {
    let mut seq = Sequencer::new();
    seq.add_unit(Box::new(RangeNormStage::new(10, 4)));
    seq.add_unit(Box::new(Conv1DStage::new(4, 2)));

    let op = make_op(0, OpKind::RangeNorm, OpParams::Elementwise { d: 20 });
    seq.submit(op, 0).expect("range_norm op routed");

    // Tick until done.
    for t in 0..200 {
        seq.tick(t);
        if seq.completed_ops() == 1 {
            break;
        }
    }
    assert_eq!(seq.completed_ops(), 1, "op completes");

    // Conv1D unit should have been untouched.
    assert_eq!(seq.unit_by_name("Conv1D").unwrap().ops_completed(), 0);
    assert_eq!(seq.unit_by_name("RangeNorm").unwrap().ops_completed(), 1);
}

#[test]
fn back_pressure_when_unit_busy() {
    let mut seq = Sequencer::new();
    seq.add_unit(Box::new(RangeNormStage::new(10, 4)));

    let op_a = make_op(0, OpKind::RangeNorm, OpParams::Elementwise { d: 20 });
    let op_b = make_op(1, OpKind::RangeNorm, OpParams::Elementwise { d: 20 });

    seq.submit(op_a, 0).expect("first ok");
    let bounced = seq.submit(op_b, 0);
    assert!(bounced.is_err(), "second op back-pressured (unit busy)");
}

#[test]
fn ready_valid_handshake_pipelines_two_ops() {
    let mut seq = Sequencer::new();
    seq.add_unit(Box::new(RangeNormStage::new(10, 4)));
    seq.add_unit(Box::new(Conv1DStage::new(4, 2)));

    // Two ops to *different* units submitted same cycle should both be
    // accepted (no contention).
    seq.submit(make_op(0, OpKind::RangeNorm, OpParams::Elementwise { d: 20 }), 0)
        .expect("range_norm");
    seq.submit(
        make_op(1, OpKind::Conv1D, OpParams::Conv1D { d: 20, k: 4 }),
        0,
    )
    .expect("conv1d");

    for t in 0..200 {
        seq.tick(t);
        if seq.completed_ops() == 2 {
            break;
        }
    }
    assert_eq!(seq.completed_ops(), 2);
}
