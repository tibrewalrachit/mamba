//! Test 8 — `trace::parse::header_sets_layer_token`
//! `LAYER N TOKEN M` propagates onto subsequent ops until the next header.

use emamba_trace::parser::parse_trace;

#[test]
fn header_propagates_to_following_ops() {
    let trace = "\
LAYER 3 TOKEN 7
RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8
RANGE_NORM in=0x3000 in_mem=sram out=0x4000 out_mem=sram D=20 dtype=int8
LAYER 4 TOKEN 0
RANGE_NORM in=0x5000 in_mem=sram out=0x6000 out_mem=sram D=20 dtype=int8
";
    let ops = parse_trace(trace).expect("parse");
    assert_eq!(ops.len(), 3);
    assert_eq!((ops[0].layer, ops[0].token), (3, 7));
    assert_eq!((ops[1].layer, ops[1].token), (3, 7));
    assert_eq!((ops[2].layer, ops[2].token), (4, 0));
}

#[test]
fn ops_before_any_header_default_to_zero() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8\n";
    let ops = parse_trace(trace).expect("parse");
    assert_eq!((ops[0].layer, ops[0].token), (0, 0));
}

#[test]
fn comments_and_blank_lines_skipped() {
    let trace = "\
# eMamba demo trace
\n
   # leading whitespace and comment

LAYER 0 TOKEN 0
RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8 # inline comment
";
    let ops = parse_trace(trace).expect("parse");
    assert_eq!(ops.len(), 1);
}
