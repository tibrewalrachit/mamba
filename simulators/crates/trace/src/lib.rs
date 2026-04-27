//! emamba-trace: op-trace frontend.
//!
//! Format spec lives in the plan §4. One op per line, leading whitespace
//! ignored, `#` for comments. Header lines `LAYER N TOKEN M` set the
//! layer/token for subsequent ops.

pub mod frontend;
pub mod parser;
