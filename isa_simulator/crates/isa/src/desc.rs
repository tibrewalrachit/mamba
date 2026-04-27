//! Per-op descriptor structs (Xemamba spec §0.7).
//!
//! Each descriptor is the in-memory record that the host points at via `rs1`
//! when issuing a Xemamba instruction. Layouts mirror the C struct definitions
//! in the spec exactly: fields are little-endian, packed, no padding inserted.
//! `from_bytes` validates the spec's range constraints and returns a typed
//! `DescError` on violation; `to_bytes` is the inverse and is used by tests
//! and the upcoming assembler.

use emamba_core::Dtype;

/// Errors a descriptor parser can raise (subset of spec §0.10 — only the
/// conditions visible at descriptor-decode time, not at execute time).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DescError {
    /// Input slice is shorter than the descriptor's spec-mandated size.
    ShortInput,
    /// Reserved bits were non-zero, or a `..=4096`/`..=65535` field was zero.
    Malformed,
    /// `dtype` field is a reserved code (0b011..0b111).
    IllegalDtype,
}

// ─── helpers ────────────────────────────────────────────────────────────

#[inline]
fn read_u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn read_u64_le(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        b[off], b[off + 1], b[off + 2], b[off + 3],
        b[off + 4], b[off + 5], b[off + 6], b[off + 7],
    ])
}

#[inline]
fn read_u16_le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn decode_dtype(byte: u8) -> Result<Dtype, DescError> {
    Dtype::from_bits(byte).ok_or(DescError::IllegalDtype)
}

// ─── §0.7.1 mac_desc_t (20 bytes) ───────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MacDesc {
    pub src_a: u32,
    pub src_b: u32,
    pub dst: u32,
    pub m: u16,
    pub n: u16,
    pub k: u16,
    pub dtype: Dtype,
}

impl MacDesc {
    pub const SIZE: usize = 20;

    pub fn from_bytes(b: &[u8]) -> Result<Self, DescError> {
        if b.len() < Self::SIZE {
            return Err(DescError::ShortInput);
        }
        // mac_desc_t (§0.7.1):
        //   off  0: src_a (u32, 4)
        //   off  4: src_b (u32, 4)
        //   off  8: dst   (u32, 4)
        //   off 12: m     (u16, 2)
        //   off 14: n     (u16, 2)
        //   off 16: k     (u16, 2)
        //   off 18: dtype (u8,  1)
        //   off 19: reserved (u8, 1)
        let src_a = read_u32_le(b, 0);
        let src_b = read_u32_le(b, 4);
        let dst = read_u32_le(b, 8);
        let m = read_u16_le(b, 12);
        let n = read_u16_le(b, 14);
        let k = read_u16_le(b, 16);
        let dtype_byte = b[18];
        let reserved = b[19];
        if dtype_byte & 0xf8 != 0 || reserved != 0 {
            return Err(DescError::Malformed);
        }
        if m == 0 || n == 0 || k == 0 {
            return Err(DescError::Malformed);
        }
        Ok(Self {
            src_a,
            src_b,
            dst,
            m,
            n,
            k,
            dtype: decode_dtype(dtype_byte)?,
        })
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::SIZE);
        b.extend_from_slice(&self.src_a.to_le_bytes());
        b.extend_from_slice(&self.src_b.to_le_bytes());
        b.extend_from_slice(&self.dst.to_le_bytes());
        b.extend_from_slice(&self.m.to_le_bytes());
        b.extend_from_slice(&self.n.to_le_bytes());
        b.extend_from_slice(&self.k.to_le_bytes());
        b.push(self.dtype.to_bits());
        b.push(0);
        b
    }
}

// ─── §0.7.2 norm_desc_t (20 bytes) ──────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NormDesc {
    pub src: u32,
    pub dst: u32,
    pub gamma: u32,
    pub beta: u32,
    pub len: u16,
    pub dtype: Dtype,
}

impl NormDesc {
    pub const SIZE: usize = 20;

    pub fn from_bytes(b: &[u8]) -> Result<Self, DescError> {
        if b.len() < Self::SIZE {
            return Err(DescError::ShortInput);
        }
        // off  0: src,  off  4: dst, off  8: gamma, off 12: beta,
        // off 16: len (u16), off 18: dtype, off 19: reserved
        let src = read_u32_le(b, 0);
        let dst = read_u32_le(b, 4);
        let gamma = read_u32_le(b, 8);
        let beta = read_u32_le(b, 12);
        let len = read_u16_le(b, 16);
        let dtype_byte = b[18];
        let reserved = b[19];
        if dtype_byte & 0xf8 != 0 || reserved != 0 {
            return Err(DescError::Malformed);
        }
        if len == 0 {
            return Err(DescError::Malformed);
        }
        Ok(Self {
            src,
            dst,
            gamma,
            beta,
            len,
            dtype: decode_dtype(dtype_byte)?,
        })
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::SIZE);
        b.extend_from_slice(&self.src.to_le_bytes());
        b.extend_from_slice(&self.dst.to_le_bytes());
        b.extend_from_slice(&self.gamma.to_le_bytes());
        b.extend_from_slice(&self.beta.to_le_bytes());
        b.extend_from_slice(&self.len.to_le_bytes());
        b.push(self.dtype.to_bits());
        b.push(0);
        b
    }
}

// ─── §0.7.3 conv_desc_t (20 bytes) ──────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ConvDesc {
    pub src: u32,
    pub dst: u32,
    pub weights: u32,
    pub cache: u32,
    pub len: u16,
    pub dtype: Dtype,
    pub k_conv: u8,
}

impl ConvDesc {
    pub const SIZE: usize = 20;

    pub fn from_bytes(b: &[u8]) -> Result<Self, DescError> {
        if b.len() < Self::SIZE {
            return Err(DescError::ShortInput);
        }
        // off  0: src, off  4: dst, off  8: weights, off 12: cache,
        // off 16: len (u16), off 18: dtype (u8), off 19: k_conv (u8)
        let src = read_u32_le(b, 0);
        let dst = read_u32_le(b, 4);
        let weights = read_u32_le(b, 8);
        let cache = read_u32_le(b, 12);
        let len = read_u16_le(b, 16);
        let dtype_byte = b[18];
        let k_conv = b[19];
        if dtype_byte & 0xf8 != 0 {
            return Err(DescError::Malformed);
        }
        // §0.7.3: k_conv ∈ 1..=63 (6-bit field).
        if k_conv == 0 || k_conv > 63 {
            return Err(DescError::Malformed);
        }
        if len == 0 {
            return Err(DescError::Malformed);
        }
        Ok(Self {
            src,
            dst,
            weights,
            cache,
            len,
            dtype: decode_dtype(dtype_byte)?,
            k_conv,
        })
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::SIZE);
        b.extend_from_slice(&self.src.to_le_bytes());
        b.extend_from_slice(&self.dst.to_le_bytes());
        b.extend_from_slice(&self.weights.to_le_bytes());
        b.extend_from_slice(&self.cache.to_le_bytes());
        b.extend_from_slice(&self.len.to_le_bytes());
        b.push(self.dtype.to_bits());
        b.push(self.k_conv);
        b
    }
}

// ─── §0.7.4 ssm_desc_t (16 bytes) ───────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SsmDesc {
    pub ctx_in: u32,
    pub state_h: u32,
    pub ctx_out: u32,
    pub ed: u16,
    pub n: u8,
    pub dtype: Dtype,
}

impl SsmDesc {
    pub const SIZE: usize = 16;

    pub fn from_bytes(b: &[u8]) -> Result<Self, DescError> {
        if b.len() < Self::SIZE {
            return Err(DescError::ShortInput);
        }
        // off  0: ctx_in, off  4: state_h, off  8: ctx_out,
        // off 12: ed (u16), off 14: n (u8), off 15: dtype (u8)
        let ctx_in = read_u32_le(b, 0);
        let state_h = read_u32_le(b, 4);
        let ctx_out = read_u32_le(b, 8);
        let ed = read_u16_le(b, 12);
        let n = b[14];
        let dtype_byte = b[15];
        if dtype_byte & 0xf8 != 0 {
            return Err(DescError::Malformed);
        }
        if ed == 0 || n == 0 {
            return Err(DescError::Malformed);
        }
        Ok(Self {
            ctx_in,
            state_h,
            ctx_out,
            ed,
            n,
            dtype: decode_dtype(dtype_byte)?,
        })
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::SIZE);
        b.extend_from_slice(&self.ctx_in.to_le_bytes());
        b.extend_from_slice(&self.state_h.to_le_bytes());
        b.extend_from_slice(&self.ctx_out.to_le_bytes());
        b.extend_from_slice(&self.ed.to_le_bytes());
        b.push(self.n);
        b.push(self.dtype.to_bits());
        b
    }
}

// ─── §0.7.5 pwl_desc_t (12 bytes) ───────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PwlDesc {
    pub src: u32,
    pub dst: u32,
    pub len: u16,
    pub dtype: Dtype,
}

impl PwlDesc {
    pub const SIZE: usize = 12;

    pub fn from_bytes(b: &[u8]) -> Result<Self, DescError> {
        if b.len() < Self::SIZE {
            return Err(DescError::ShortInput);
        }
        // off 0: src, off 4: dst, off 8: len (u16), off 10: dtype, off 11: reserved.
        let src = read_u32_le(b, 0);
        let dst = read_u32_le(b, 4);
        let len = read_u16_le(b, 8);
        let dtype_byte = b[10];
        let reserved = b[11];
        if dtype_byte & 0xf8 != 0 || reserved != 0 {
            return Err(DescError::Malformed);
        }
        if len == 0 {
            return Err(DescError::Malformed);
        }
        Ok(Self {
            src,
            dst,
            len,
            dtype: decode_dtype(dtype_byte)?,
        })
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::SIZE);
        b.extend_from_slice(&self.src.to_le_bytes());
        b.extend_from_slice(&self.dst.to_le_bytes());
        b.extend_from_slice(&self.len.to_le_bytes());
        b.push(self.dtype.to_bits());
        b.push(0);
        b
    }
}

// ─── §0.7.6 dma_desc_t (16 bytes) ───────────────────────────────────────

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DmaDesc {
    pub dram_addr: u64,
    pub sram_addr: u32,
    pub bytes: u32,
}

impl DmaDesc {
    pub const SIZE: usize = 16;

    pub fn from_bytes(b: &[u8]) -> Result<Self, DescError> {
        if b.len() < Self::SIZE {
            return Err(DescError::ShortInput);
        }
        // off 0: dram_addr (u64), off 8: sram_addr (u32), off 12: bytes (u32).
        // bytes may be 0 per §0.7.6 — no Malformed check.
        Ok(Self {
            dram_addr: read_u64_le(b, 0),
            sram_addr: read_u32_le(b, 8),
            bytes: read_u32_le(b, 12),
        })
    }

    pub fn to_bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(Self::SIZE);
        b.extend_from_slice(&self.dram_addr.to_le_bytes());
        b.extend_from_slice(&self.sram_addr.to_le_bytes());
        b.extend_from_slice(&self.bytes.to_le_bytes());
        b
    }
}
