//! Test 9 — `trace::parse::all_op_kinds_roundtrip`
//! Every op kind in the grammar parses, and the resulting OpKind tag matches.
//! Stricter than a fuzz roundtrip but covers the grammar surface explicitly.

use emamba_core::{MemSpace, OpKind, OpParams};
use emamba_trace::parser::parse_trace;

#[test]
fn all_op_kinds_parse() {
    let trace = "\
LAYER 0 TOKEN 0
LOAD       src=0x80000000 src_mem=dram dst=0x1000 dst_mem=sram bytes=20
RANGE_NORM in=0x1000 in_mem=sram out=0x1100 out_mem=sram D=20 dtype=int8
LINEAR     in=0x1100 in_mem=sram out=0x1200 out_mem=sram in_dim=20 out_dim=40 dtype=int8
CONV1D     in=0x1200 in_mem=sram out=0x1300 out_mem=sram D=20 K=4 dtype=int8
PWL_SILU   in=0x1300 in_mem=sram out=0x1400 out_mem=sram D=20 dtype=int8
SSM_STATE  in=0x1400 in_mem=sram state=0xA000 state_mem=sram D=20 N=8 E=2 dtype=int8 state_dtype=int24
SSM_OUTPUT state=0xA000 state_mem=sram in=0x1400 in_mem=sram out=0x1500 out_mem=sram D=20 N=8 E=2 dtype=int8
PWL_EXP    in=0x1500 in_mem=sram out=0x1600 out_mem=sram D=20 dtype=int8
RELU       in=0x1600 in_mem=sram out=0x1700 out_mem=sram D=20 dtype=int8
RESIDUAL   a=0x1700 a_mem=sram b=0x1000 b_mem=sram out=0x1800 out_mem=sram D=20 dtype=int8
STORE      src=0x1800 src_mem=sram dst=0x80001000 dst_mem=dram bytes=20
";
    let ops = parse_trace(trace).expect("parse");

    let kinds: Vec<OpKind> = ops.iter().map(|o| o.kind).collect();
    assert_eq!(
        kinds,
        vec![
            OpKind::Load,
            OpKind::RangeNorm,
            OpKind::LinearProj,
            OpKind::Conv1D,
            OpKind::PwlSilu,
            OpKind::SsmStateUpdate,
            OpKind::SsmOutput,
            OpKind::PwlExp,
            OpKind::Relu,
            OpKind::ResidualAdd,
            OpKind::Store,
        ]
    );

    // ids are dense and start at 0
    for (i, op) in ops.iter().enumerate() {
        assert_eq!(op.id, i as u64);
    }

    // sanity-check a few params
    let load = ops.iter().find(|o| o.kind == OpKind::Load).unwrap();
    assert!(matches!(load.params, OpParams::Memmove { bytes: 20 }));
    assert_eq!(load.inputs[0].space, MemSpace::Dram);
    assert_eq!(load.outputs[0].space, MemSpace::Sram);

    let ssm = ops.iter().find(|o| o.kind == OpKind::SsmStateUpdate).unwrap();
    assert!(matches!(
        ssm.params,
        OpParams::Ssm { d: 20, n: 8, e: 2 }
    ));

    let lin = ops.iter().find(|o| o.kind == OpKind::LinearProj).unwrap();
    assert!(matches!(
        lin.params,
        OpParams::Linear {
            in_dim: 20,
            out_dim: 40
        }
    ));
}
