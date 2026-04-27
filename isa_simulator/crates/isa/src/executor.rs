//! Functional executor for Xemamba instructions.
//!
//! `OpExecutor` consumes one R-type instruction word + the value of `rs1`
//! (the descriptor pointer) and runs the spec §0.8 semantics against a bus.
//! The result is a u32 status word matching the spec §0.10 codes — this is
//! what the host writes back to `rd`.
//!
//! The executor is generic over `M: Memory`, mirroring rrs's
//! `InstructionExecutor<'a, M>` pattern.

use emamba_core::Dtype;

use crate::desc::{ConvDesc, DescError, DmaDesc, MacDesc, NormDesc, PwlDesc, SsmDesc};
use crate::insn::{funct7, RType, FUNCT3_XEMAMBA_V0_2};
use crate::memory::{
    read_f32_slice, write_f32_slice, MemAccessSize, Memory, SliceError, A_SRAM_BASE,
    A_SRAM_SIZE, DRAM_BASE, DRAM_SIZE, STATE_SRAM_BASE, STATE_SRAM_SIZE, W_SRAM_BASE,
    W_SRAM_SIZE,
};
use crate::numerics::{exp_pw, range_normalization, relu, silu_pw};

/// Status codes returned in `rd` after every Xemamba instruction (spec §0.10).
pub mod status {
    pub const SUCCESS: u32 = 0x00;
    pub const ILLEGAL_OPCODE: u32 = 0x01;
    pub const ILLEGAL_DTYPE: u32 = 0x02;
    pub const MISALIGNED: u32 = 0x03;
    pub const BUS_FAULT: u32 = 0x04;
    pub const MALFORMED_OP: u32 = 0x05;
    pub const OUT_OF_BOUNDS: u32 = 0x06;
    pub const DESCRIPTOR_FAULT: u32 = 0x07;
}

pub struct OpExecutor<'a, M: Memory> {
    pub mem: &'a mut M,
}

impl<'a, M: Memory> OpExecutor<'a, M> {
    pub fn new(mem: &'a mut M) -> Self {
        Self { mem }
    }

    /// Decode + execute one instruction word.
    pub fn step(&mut self, insn: u32, rs1_value: u64) -> u32 {
        let r = RType::new(insn);
        if !r.is_xemamba() || r.funct3 != FUNCT3_XEMAMBA_V0_2 {
            return status::ILLEGAL_OPCODE;
        }
        match r.funct7 {
            funct7::MAC => self.exec_mac(rs1_value),
            funct7::NORM => self.exec_norm(rs1_value),
            funct7::CONV => self.exec_conv(rs1_value),
            funct7::SSM => self.exec_ssm(rs1_value),
            funct7::SILU => self.exec_pwl(rs1_value, Pwl::Silu),
            funct7::EXP => self.exec_pwl(rs1_value, Pwl::Exp),
            funct7::DMAL => self.exec_dma(rs1_value, DmaDir::Load),
            funct7::DMAS => self.exec_dma(rs1_value, DmaDir::Store),
            _ => status::ILLEGAL_OPCODE,
        }
    }

    fn read_descriptor(&mut self, addr: u64, len: usize) -> Result<Vec<u8>, u32> {
        if addr % 4 != 0 {
            return Err(status::MISALIGNED);
        }
        let mut buf = vec![0u8; len];
        for (i, slot) in buf.iter_mut().enumerate() {
            let a = (addr as u32).wrapping_add(i as u32);
            let v = self
                .mem
                .read_mem(a, MemAccessSize::Byte)
                .ok_or(status::DESCRIPTOR_FAULT)?;
            *slot = v as u8;
        }
        Ok(buf)
    }

    // ─── MAC_TILE ────────────────────────────────────────────────────────

    fn exec_mac(&mut self, rs1_value: u64) -> u32 {
        let bytes = match self.read_descriptor(rs1_value, MacDesc::SIZE) {
            Ok(b) => b,
            Err(s) => return s,
        };
        let d = match MacDesc::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => return desc_err_to_status(e),
        };
        if d.dtype != Dtype::Fp32 {
            return status::ILLEGAL_DTYPE;
        }
        if let Err(s) = check_sram(d.src_a, (d.m as u32) * (d.k as u32) * 4) {
            return s;
        }
        if let Err(s) = check_sram(d.src_b, (d.k as u32) * (d.n as u32) * 4) {
            return s;
        }
        if let Err(s) = check_sram(d.dst, (d.m as u32) * (d.n as u32) * 4) {
            return s;
        }
        let m = d.m as usize;
        let n = d.n as usize;
        let k = d.k as usize;
        let mut a = vec![0.0_f32; m * k];
        let mut b = vec![0.0_f32; k * n];
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.src_a, &mut a)) {
            return s;
        }
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.src_b, &mut b)) {
            return s;
        }
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for r in 0..k {
                    acc += a[i * k + r] * b[r * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        slice_err_to_status(write_f32_slice(self.mem, d.dst, &out))
            .err()
            .unwrap_or(status::SUCCESS)
    }

    // ─── NORM ────────────────────────────────────────────────────────────

    fn exec_norm(&mut self, rs1_value: u64) -> u32 {
        let bytes = match self.read_descriptor(rs1_value, NormDesc::SIZE) {
            Ok(b) => b,
            Err(s) => return s,
        };
        let d = match NormDesc::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => return desc_err_to_status(e),
        };
        if d.dtype != Dtype::Fp32 {
            return status::ILLEGAL_DTYPE;
        }
        let n_bytes = (d.len as u32) * 4;
        for &addr in &[d.src, d.dst, d.gamma, d.beta] {
            if let Err(s) = check_sram(addr, n_bytes) {
                return s;
            }
        }
        let n = d.len as usize;
        let mut x = vec![0.0_f32; n];
        let mut g = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.src, &mut x)) {
            return s;
        }
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.gamma, &mut g)) {
            return s;
        }
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.beta, &mut b)) {
            return s;
        }
        let out = range_normalization(&x, &g, &b, 1e-5_f32);
        slice_err_to_status(write_f32_slice(self.mem, d.dst, &out))
            .err()
            .unwrap_or(status::SUCCESS)
    }

    // ─── CONV_STEP ───────────────────────────────────────────────────────

    fn exec_conv(&mut self, rs1_value: u64) -> u32 {
        let bytes = match self.read_descriptor(rs1_value, ConvDesc::SIZE) {
            Ok(b) => b,
            Err(s) => return s,
        };
        let d = match ConvDesc::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => return desc_err_to_status(e),
        };
        if d.dtype != Dtype::Fp32 {
            return status::ILLEGAL_DTYPE;
        }
        let len = d.len as usize;
        let k_conv = d.k_conv as usize;
        for (addr, n_elems) in [
            (d.src, len),
            (d.dst, len),
            (d.weights, k_conv * len),
            (d.cache, (k_conv - 1) * len),
        ] {
            if let Err(s) = check_sram(addr, (n_elems * 4) as u32) {
                return s;
            }
        }
        let mut src = vec![0.0_f32; len];
        let mut weights = vec![0.0_f32; k_conv * len];
        let mut cache = vec![0.0_f32; (k_conv - 1) * len];
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.src, &mut src)) {
            return s;
        }
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.weights, &mut weights)) {
            return s;
        }
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.cache, &mut cache)) {
            return s;
        }
        let mut out = vec![0.0_f32; len];
        for c in 0..len {
            let mut acc = weights[0 * len + c] * src[c];
            for t in 1..k_conv {
                acc += weights[t * len + c] * cache[(t - 1) * len + c];
            }
            out[c] = acc;
        }
        // Shift cache: new_cache[0] = src; new_cache[t] = old_cache[t-1] for t in 1..k_conv-1.
        let mut new_cache = vec![0.0_f32; (k_conv - 1) * len];
        for c in 0..len {
            new_cache[0 * len + c] = src[c];
            for t in 1..(k_conv - 1) {
                new_cache[t * len + c] = cache[(t - 1) * len + c];
            }
        }
        if let Err(s) = slice_err_to_status(write_f32_slice(self.mem, d.dst, &out)) {
            return s;
        }
        slice_err_to_status(write_f32_slice(self.mem, d.cache, &new_cache))
            .err()
            .unwrap_or(status::SUCCESS)
    }

    // ─── SSM_STEP ────────────────────────────────────────────────────────

    fn exec_ssm(&mut self, rs1_value: u64) -> u32 {
        let bytes = match self.read_descriptor(rs1_value, SsmDesc::SIZE) {
            Ok(b) => b,
            Err(s) => return s,
        };
        let d = match SsmDesc::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => return desc_err_to_status(e),
        };
        if d.dtype != Dtype::Fp32 {
            return status::ILLEGAL_DTYPE;
        }
        let ed = d.ed as usize;
        let n = d.n as usize;
        let ctx_in_bytes = ((4 * ed + 2 * ed * n) * 4) as u32;
        let ctx_out_bytes = (ed * 4) as u32;
        let state_bytes = ((ed * n) * 4) as u32;
        if let Err(s) = check_sram(d.ctx_in, ctx_in_bytes) {
            return s;
        }
        if let Err(s) = check_sram(d.ctx_out, ctx_out_bytes) {
            return s;
        }
        if let Err(s) = check_sram(d.state_h, state_bytes) {
            return s;
        }

        // Read packed ctx_in (offsets per spec §0.7.4): x_t, delta, B, C, A_diag, D_skip.
        let mut buf = vec![0.0_f32; 4 * ed + 2 * ed * n];
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.ctx_in, &mut buf)) {
            return s;
        }
        let off_xt = 0;
        let off_delta = ed;
        let off_b = 2 * ed;
        let off_c = 2 * ed + ed * n;
        let off_a = 2 * ed + 2 * ed * n;
        let off_d = 3 * ed + 2 * ed * n;

        let mut h = vec![0.0_f32; ed * n];
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.state_h, &mut h)) {
            return s;
        }

        let exp = exp_pw();
        let mut delta_relu = vec![0.0_f32; ed];
        for i in 0..ed {
            delta_relu[i] = relu(buf[off_delta + i]);
        }
        for i in 0..ed {
            let dr = delta_relu[i];
            let xi = buf[off_xt + i];
            for j in 0..n {
                let a_diag = buf[off_a + i * n + j];
                let d_bar = exp.apply(dr * a_diag);
                let b_bar = dr * buf[off_b + i * n + j] * xi;
                let h_old = h[i * n + j];
                h[i * n + j] = d_bar * h_old + b_bar;
            }
        }
        let mut y_t = vec![0.0_f32; ed];
        for i in 0..ed {
            let mut acc = buf[off_d + i] * buf[off_xt + i];
            for j in 0..n {
                acc += buf[off_c + i * n + j] * h[i * n + j];
            }
            y_t[i] = acc;
        }
        if let Err(s) = slice_err_to_status(write_f32_slice(self.mem, d.state_h, &h)) {
            return s;
        }
        slice_err_to_status(write_f32_slice(self.mem, d.ctx_out, &y_t))
            .err()
            .unwrap_or(status::SUCCESS)
    }

    // ─── SILU / EXP ──────────────────────────────────────────────────────

    fn exec_pwl(&mut self, rs1_value: u64, kind: Pwl) -> u32 {
        let bytes = match self.read_descriptor(rs1_value, PwlDesc::SIZE) {
            Ok(b) => b,
            Err(s) => return s,
        };
        let d = match PwlDesc::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => return desc_err_to_status(e),
        };
        if d.dtype != Dtype::Fp32 {
            return status::ILLEGAL_DTYPE;
        }
        let n_bytes = (d.len as u32) * 4;
        if let Err(s) = check_sram(d.src, n_bytes) {
            return s;
        }
        if let Err(s) = check_sram(d.dst, n_bytes) {
            return s;
        }
        let mut xs = vec![0.0_f32; d.len as usize];
        if let Err(s) = slice_err_to_status(read_f32_slice(self.mem, d.src, &mut xs)) {
            return s;
        }
        let pwl = match kind {
            Pwl::Silu => silu_pw(),
            Pwl::Exp => exp_pw(),
        };
        let mut out = vec![0.0_f32; d.len as usize];
        pwl.apply_slice(&xs, &mut out);
        slice_err_to_status(write_f32_slice(self.mem, d.dst, &out))
            .err()
            .unwrap_or(status::SUCCESS)
    }

    // ─── DMA load/store ──────────────────────────────────────────────────

    fn exec_dma(&mut self, rs1_value: u64, dir: DmaDir) -> u32 {
        let bytes = match self.read_descriptor(rs1_value, DmaDesc::SIZE) {
            Ok(b) => b,
            Err(s) => return s,
        };
        let d = match DmaDesc::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => return desc_err_to_status(e),
        };
        if d.bytes == 0 {
            return status::SUCCESS;
        }
        // DMA endpoints: SRAM side bounded check, DRAM side bounded check.
        if let Err(s) = check_sram(d.sram_addr, d.bytes) {
            return s;
        }
        if let Err(s) = check_dram(d.dram_addr as u32, d.bytes) {
            return s;
        }
        for i in 0..d.bytes {
            let (from, to) = match dir {
                DmaDir::Load => (d.dram_addr as u32 + i, d.sram_addr + i),
                DmaDir::Store => (d.sram_addr + i, d.dram_addr as u32 + i),
            };
            let v = match self.mem.read_mem(from, MemAccessSize::Byte) {
                Some(v) => v,
                None => return status::OUT_OF_BOUNDS,
            };
            if !self.mem.write_mem(to, MemAccessSize::Byte, v) {
                return status::OUT_OF_BOUNDS;
            }
        }
        status::SUCCESS
    }
}

#[derive(Copy, Clone)]
enum Pwl {
    Silu,
    Exp,
}

#[derive(Copy, Clone)]
enum DmaDir {
    Load,
    Store,
}

// ─── helpers ────────────────────────────────────────────────────────────

fn desc_err_to_status(e: DescError) -> u32 {
    match e {
        DescError::ShortInput => status::DESCRIPTOR_FAULT,
        DescError::Malformed => status::MALFORMED_OP,
        DescError::IllegalDtype => status::ILLEGAL_DTYPE,
    }
}

fn slice_err_to_status<T>(r: Result<T, SliceError>) -> Result<T, u32> {
    r.map_err(|e| match e {
        SliceError::Misaligned => status::MISALIGNED,
        SliceError::OutOfBounds => status::OUT_OF_BOUNDS,
    })
}

/// Confirm `[addr, addr + bytes)` lies entirely inside one of the three
/// SRAM regions (spec §0.5.3 — compute ops MUST address SRAM only).
fn check_sram(addr: u32, bytes: u32) -> Result<(), u32> {
    let end = addr.checked_add(bytes).ok_or(status::OUT_OF_BOUNDS)?;
    let in_w = addr >= W_SRAM_BASE && end <= W_SRAM_BASE + W_SRAM_SIZE;
    let in_a = addr >= A_SRAM_BASE && end <= A_SRAM_BASE + A_SRAM_SIZE;
    let in_s = addr >= STATE_SRAM_BASE && end <= STATE_SRAM_BASE + STATE_SRAM_SIZE;
    if in_w || in_a || in_s {
        Ok(())
    } else {
        Err(status::BUS_FAULT)
    }
}

fn check_dram(addr: u32, bytes: u32) -> Result<(), u32> {
    let end = addr.checked_add(bytes).ok_or(status::OUT_OF_BOUNDS)?;
    if addr >= DRAM_BASE && end <= DRAM_BASE + DRAM_SIZE {
        Ok(())
    } else {
        Err(status::BUS_FAULT)
    }
}
