//! Op-trace parser. Plan §4 grammar.
//!
//! Each non-comment, non-blank, non-header line is one op:
//!     OP_KIND key1=val1 key2=val2 ...
//! Header line `LAYER N TOKEN M` sets the (layer, token) for following ops.
//! Addresses are hex (`0x...`); `*_mem` fields are `sram` or `dram`; `dtype`
//! is `int8` | `int24` | `fp32`.

use emamba_core::{Addr, Dtype, MemSpace, Op, OpKind, OpParams, TensorDesc};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ParseError {
    UnknownOp { line: usize, kind: String },
    MissingKey { line: usize, op: String, key: &'static str },
    BadValue { line: usize, key: String, value: String },
    BadHeader { line: usize, raw: String },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownOp { line, kind } => {
                write!(f, "line {}: unknown op kind `{}`", line, kind)
            }
            ParseError::MissingKey { line, op, key } => {
                write!(f, "line {}: op {} missing required key `{}`", line, op, key)
            }
            ParseError::BadValue { line, key, value } => {
                write!(f, "line {}: bad value for key `{}`: `{}`", line, key, value)
            }
            ParseError::BadHeader { line, raw } => {
                write!(f, "line {}: malformed header: `{}`", line, raw)
            }
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_trace(input: &str) -> Result<Vec<Op>, ParseError> {
    let mut ops = Vec::new();
    let mut layer: u32 = 0;
    let mut token: u32 = 0;
    let mut next_id: u64 = 0;

    for (idx, raw) in input.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = strip_comment(raw).trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let head = parts.next().expect("non-empty after trim");

        if head == "LAYER" {
            let l = parts.next().ok_or_else(|| ParseError::BadHeader {
                line: line_no,
                raw: raw.to_string(),
            })?;
            let tok_kw = parts.next().ok_or_else(|| ParseError::BadHeader {
                line: line_no,
                raw: raw.to_string(),
            })?;
            let t = parts.next().ok_or_else(|| ParseError::BadHeader {
                line: line_no,
                raw: raw.to_string(),
            })?;
            if tok_kw != "TOKEN" {
                return Err(ParseError::BadHeader {
                    line: line_no,
                    raw: raw.to_string(),
                });
            }
            layer = l.parse().map_err(|_| ParseError::BadHeader {
                line: line_no,
                raw: raw.to_string(),
            })?;
            token = t.parse().map_err(|_| ParseError::BadHeader {
                line: line_no,
                raw: raw.to_string(),
            })?;
            continue;
        }

        let kv = collect_kv(parts, line_no, head)?;
        let op = build_op(head, &kv, line_no, layer, token, next_id)?;
        next_id += 1;
        ops.push(op);
    }

    Ok(ops)
}

fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

fn collect_kv<'a>(
    parts: impl Iterator<Item = &'a str>,
    line_no: usize,
    op: &str,
) -> Result<HashMap<&'a str, &'a str>, ParseError> {
    let mut out = HashMap::new();
    for tok in parts {
        let mut split = tok.splitn(2, '=');
        let k = split.next().ok_or_else(|| ParseError::BadValue {
            line: line_no,
            key: op.to_string(),
            value: tok.to_string(),
        })?;
        let v = split.next().ok_or_else(|| ParseError::BadValue {
            line: line_no,
            key: k.to_string(),
            value: tok.to_string(),
        })?;
        out.insert(k, v);
    }
    Ok(out)
}

fn build_op(
    kind_str: &str,
    kv: &HashMap<&str, &str>,
    line: usize,
    layer: u32,
    token: u32,
    id: u64,
) -> Result<Op, ParseError> {
    macro_rules! req {
        ($key:expr) => {{
            kv.get($key).copied().ok_or_else(|| ParseError::MissingKey {
                line,
                op: kind_str.to_string(),
                key: $key,
            })?
        }};
    }

    let parse_addr = |s: &str, key: &str| -> Result<Addr, ParseError> {
        let s = s.trim_start_matches("0x");
        u64::from_str_radix(s, 16).map_err(|_| ParseError::BadValue {
            line,
            key: key.to_string(),
            value: s.to_string(),
        })
    };
    let parse_u32 = |s: &str, key: &str| -> Result<u32, ParseError> {
        s.parse().map_err(|_| ParseError::BadValue {
            line,
            key: key.to_string(),
            value: s.to_string(),
        })
    };
    let parse_dtype = |s: &str, key: &str| -> Result<Dtype, ParseError> {
        match s {
            "int8" => Ok(Dtype::Int8),
            "int24" => Ok(Dtype::Int24),
            "fp32" => Ok(Dtype::Fp32),
            _ => Err(ParseError::BadValue {
                line,
                key: key.to_string(),
                value: s.to_string(),
            }),
        }
    };
    let parse_mem = |s: &str, key: &str| -> Result<MemSpace, ParseError> {
        match s {
            "sram" => Ok(MemSpace::Sram),
            "dram" => Ok(MemSpace::Dram),
            _ => Err(ParseError::BadValue {
                line,
                key: key.to_string(),
                value: s.to_string(),
            }),
        }
    };

    let kind = match kind_str {
        "RANGE_NORM" => OpKind::RangeNorm,
        "CONV1D" => OpKind::Conv1D,
        "SSM_STATE" => OpKind::SsmStateUpdate,
        "SSM_OUTPUT" => OpKind::SsmOutput,
        "PWL_SILU" => OpKind::PwlSilu,
        "PWL_EXP" => OpKind::PwlExp,
        "RELU" => OpKind::Relu,
        "LINEAR" => OpKind::LinearProj,
        "RESIDUAL" => OpKind::ResidualAdd,
        "LOAD" => OpKind::Load,
        "STORE" => OpKind::Store,
        other => {
            return Err(ParseError::UnknownOp {
                line,
                kind: other.to_string(),
            })
        }
    };

    let (inputs, outputs, params) = match kind {
        OpKind::RangeNorm
        | OpKind::PwlSilu
        | OpKind::PwlExp
        | OpKind::Relu => {
            let d = parse_u32(req!("D"), "D")?;
            let dtype = parse_dtype(req!("dtype"), "dtype")?;
            let inp = TensorDesc {
                base: parse_addr(req!("in"), "in")?,
                space: parse_mem(req!("in_mem"), "in_mem")?,
                shape: vec![d],
                dtype,
            };
            let out = TensorDesc {
                base: parse_addr(req!("out"), "out")?,
                space: parse_mem(req!("out_mem"), "out_mem")?,
                shape: vec![d],
                dtype,
            };
            (vec![inp], vec![out], OpParams::Elementwise { d })
        }
        OpKind::Conv1D => {
            let d = parse_u32(req!("D"), "D")?;
            let k = parse_u32(req!("K"), "K")?;
            let dtype = parse_dtype(req!("dtype"), "dtype")?;
            let inp = TensorDesc {
                base: parse_addr(req!("in"), "in")?,
                space: parse_mem(req!("in_mem"), "in_mem")?,
                shape: vec![d],
                dtype,
            };
            let out = TensorDesc {
                base: parse_addr(req!("out"), "out")?,
                space: parse_mem(req!("out_mem"), "out_mem")?,
                shape: vec![d],
                dtype,
            };
            (vec![inp], vec![out], OpParams::Conv1D { d, k })
        }
        OpKind::SsmStateUpdate => {
            let d = parse_u32(req!("D"), "D")?;
            let n = parse_u32(req!("N"), "N")?;
            let e = parse_u32(req!("E"), "E")?;
            let dtype = parse_dtype(req!("dtype"), "dtype")?;
            let state_dtype = parse_dtype(req!("state_dtype"), "state_dtype")?;
            let inp = TensorDesc {
                base: parse_addr(req!("in"), "in")?,
                space: parse_mem(req!("in_mem"), "in_mem")?,
                shape: vec![d],
                dtype,
            };
            let state = TensorDesc {
                base: parse_addr(req!("state"), "state")?,
                space: parse_mem(req!("state_mem"), "state_mem")?,
                shape: vec![d, n],
                dtype: state_dtype,
            };
            // SSM_STATE rewrites the state tensor in-place; expose it as both
            // input (we read h_{t-1}) and output (we write h_t).
            (
                vec![inp, state.clone()],
                vec![state],
                OpParams::Ssm { d, n, e },
            )
        }
        OpKind::SsmOutput => {
            let d = parse_u32(req!("D"), "D")?;
            let n = parse_u32(req!("N"), "N")?;
            let e = parse_u32(req!("E"), "E")?;
            let dtype = parse_dtype(req!("dtype"), "dtype")?;
            let state = TensorDesc {
                base: parse_addr(req!("state"), "state")?,
                space: parse_mem(req!("state_mem"), "state_mem")?,
                shape: vec![d, n],
                dtype: Dtype::Int24,
            };
            let inp = TensorDesc {
                base: parse_addr(req!("in"), "in")?,
                space: parse_mem(req!("in_mem"), "in_mem")?,
                shape: vec![d],
                dtype,
            };
            let out = TensorDesc {
                base: parse_addr(req!("out"), "out")?,
                space: parse_mem(req!("out_mem"), "out_mem")?,
                shape: vec![d],
                dtype,
            };
            (vec![state, inp], vec![out], OpParams::Ssm { d, n, e })
        }
        OpKind::LinearProj => {
            let in_dim = parse_u32(req!("in_dim"), "in_dim")?;
            let out_dim = parse_u32(req!("out_dim"), "out_dim")?;
            let dtype = parse_dtype(req!("dtype"), "dtype")?;
            let inp = TensorDesc {
                base: parse_addr(req!("in"), "in")?,
                space: parse_mem(req!("in_mem"), "in_mem")?,
                shape: vec![in_dim],
                dtype,
            };
            let out = TensorDesc {
                base: parse_addr(req!("out"), "out")?,
                space: parse_mem(req!("out_mem"), "out_mem")?,
                shape: vec![out_dim],
                dtype,
            };
            (vec![inp], vec![out], OpParams::Linear { in_dim, out_dim })
        }
        OpKind::ResidualAdd => {
            let d = parse_u32(req!("D"), "D")?;
            let dtype = parse_dtype(req!("dtype"), "dtype")?;
            let a = TensorDesc {
                base: parse_addr(req!("a"), "a")?,
                space: parse_mem(req!("a_mem"), "a_mem")?,
                shape: vec![d],
                dtype,
            };
            let b = TensorDesc {
                base: parse_addr(req!("b"), "b")?,
                space: parse_mem(req!("b_mem"), "b_mem")?,
                shape: vec![d],
                dtype,
            };
            let out = TensorDesc {
                base: parse_addr(req!("out"), "out")?,
                space: parse_mem(req!("out_mem"), "out_mem")?,
                shape: vec![d],
                dtype,
            };
            (vec![a, b], vec![out], OpParams::Elementwise { d })
        }
        OpKind::Load => {
            let bytes = parse_u32(req!("bytes"), "bytes")?;
            let src = TensorDesc {
                base: parse_addr(req!("src"), "src")?,
                space: parse_mem(req!("src_mem"), "src_mem")?,
                shape: vec![bytes],
                dtype: Dtype::Int8,
            };
            let dst = TensorDesc {
                base: parse_addr(req!("dst"), "dst")?,
                space: parse_mem(req!("dst_mem"), "dst_mem")?,
                shape: vec![bytes],
                dtype: Dtype::Int8,
            };
            (vec![src], vec![dst], OpParams::Memmove { bytes })
        }
        OpKind::Store => {
            let bytes = parse_u32(req!("bytes"), "bytes")?;
            let src = TensorDesc {
                base: parse_addr(req!("src"), "src")?,
                space: parse_mem(req!("src_mem"), "src_mem")?,
                shape: vec![bytes],
                dtype: Dtype::Int8,
            };
            let dst = TensorDesc {
                base: parse_addr(req!("dst"), "dst")?,
                space: parse_mem(req!("dst_mem"), "dst_mem")?,
                shape: vec![bytes],
                dtype: Dtype::Int8,
            };
            (vec![src], vec![dst], OpParams::Memmove { bytes })
        }
    };

    Ok(Op {
        id,
        layer,
        token,
        kind,
        inputs,
        outputs,
        params,
        scratch: [0; 4],
        callback: None,
    })
}
