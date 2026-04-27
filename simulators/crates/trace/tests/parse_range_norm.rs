//! Test 6 — `trace::parse::range_norm_one_line`
//! One `RANGE_NORM` line parses into an `Op` with the right shape and dtype.

use emamba_core::{Dtype, MemSpace, OpKind, OpParams};
use emamba_trace::parser::parse_trace;

#[test]
fn range_norm_one_line() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8\n";
    let ops = parse_trace(trace).expect("parse must succeed");

    assert_eq!(ops.len(), 1);
    let op = &ops[0];
    assert_eq!(op.kind, OpKind::RangeNorm);
    assert_eq!(op.layer, 0);
    assert_eq!(op.token, 0);

    assert_eq!(op.inputs.len(), 1);
    let inp = &op.inputs[0];
    assert_eq!(inp.base, 0x1000);
    assert_eq!(inp.space, MemSpace::Sram);
    assert_eq!(inp.shape, vec![20]);
    assert_eq!(inp.dtype, Dtype::Int8);

    assert_eq!(op.outputs.len(), 1);
    let out = &op.outputs[0];
    assert_eq!(out.base, 0x2000);
    assert_eq!(out.space, MemSpace::Sram);
    assert_eq!(out.shape, vec![20]);
    assert_eq!(out.dtype, Dtype::Int8);

    matches!(op.params, OpParams::Elementwise { .. });
}
