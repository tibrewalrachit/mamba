//! Test 7 — `trace::parse::missing_required_key_errors`
//! A missing required key produces `ParseError::MissingKey` with the line number.

use emamba_trace::parser::{parse_trace, ParseError};

#[test]
fn missing_dtype_errors_with_line_number() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20\n";
    let err = parse_trace(trace).expect_err("must error");
    match err {
        ParseError::MissingKey { line, op, key } => {
            assert_eq!(line, 1);
            assert_eq!(op, "RANGE_NORM");
            assert_eq!(key, "dtype");
        }
        other => panic!("wrong error variant: {:?}", other),
    }
}

#[test]
fn missing_key_on_later_line_reports_correct_line() {
    let trace = "\
# header comment
LAYER 0 TOKEN 0
RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=int8
RANGE_NORM in=0x3000 in_mem=sram out=0x4000 out_mem=sram dtype=int8
";
    let err = parse_trace(trace).expect_err("must error");
    match err {
        ParseError::MissingKey { line, op, key } => {
            assert_eq!(line, 4);
            assert_eq!(op, "RANGE_NORM");
            assert_eq!(key, "D");
        }
        other => panic!("wrong error variant: {:?}", other),
    }
}

#[test]
fn unknown_op_kind_errors() {
    let trace = "FOOBAR in=0x1000 D=20\n";
    let err = parse_trace(trace).expect_err("must error");
    matches!(err, ParseError::UnknownOp { .. });
}

#[test]
fn bad_dtype_errors() {
    let trace = "RANGE_NORM in=0x1000 in_mem=sram out=0x2000 out_mem=sram D=20 dtype=fp16\n";
    let err = parse_trace(trace).expect_err("must error");
    matches!(err, ParseError::BadValue { .. });
}
