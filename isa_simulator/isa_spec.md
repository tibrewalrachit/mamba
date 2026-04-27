# Xemamba — eMamba RISC-V custom extension, v0.2

Status: draft (v0.2)
Companion to: `architecture.md` (the *why*); this document is the *what*.
Supersedes: v0.1 (128-bit fixed-width encoding). v0.2 reframes eMamba as a
RISC-V non-standard vendor extension named **Xemamba** so the accelerator can
attach to any open RISC-V core (Rocket, BOOM, CVA6, etc.) — not only as a
RoCC coprocessor, but also as a cache-coherent peripheral driven by ordinary
RV32/RV64 code.

## 0.1 Scope

Xemamba is a vendor-defined non-standard RISC-V extension (per the naming
rules in *RISC-V Unprivileged ISA Manual*, Chapter 15 — non-standard
extensions use the `X` prefix). It defines:

- a single major opcode in the RISC-V *custom-0* slot,
- 8 instructions (one per eMamba functional block),
- the per-instruction calling convention (which x-registers carry which
  operands, what the accelerator reads from memory),
- the in-memory descriptor layout that each instruction's `rs1` points at,
- a 23-bit on-chip SRAM bus map and a 32-bit DRAM-side address window,
- the four normative numeric primitives (range-norm, ReLU, PWL-SiLU, PWL-Exp)
  inherited from `mamba/approximations.py`,
- exception conditions (mapped onto a status word in `rd`).

The companion `architecture.md` motivates these choices from the eMamba paper
and the workload analysis in `mamba/workload/results/report.json`. Read it for
context; nothing in this document depends on it for correctness.

This is **v0.2**: FP32-only, decode-only single token, no fences yet (memory
ordering against the host hart is left to a future revision). INT8/INT24
dtype codes are reserved for future revisions and decode-only today.

## 0.2 Programmer's model

- **Host model.** Xemamba instructions execute on the same hart as ordinary
  RISC-V code. Each Xemamba instruction is a single 32-bit word in the
  instruction stream — it occupies one slot in the host's fetch/decode
  pipeline like any R-type instruction. From the RISC-V manual's perspective,
  the accelerator is a functional unit that completes the instruction
  (synchronously in v0.2; asynchronously with `xemamba.fence` in v0.3+).
- **Operands.** Every Xemamba instruction has the same shape:
  `<mnemonic> rd, rs1, rs2`. Conventionally `rs1` holds a pointer (XLEN-wide,
  byte address in the host's address space) to an in-memory **descriptor**
  struct that carries the bulky per-op parameters (tile dimensions, dtype,
  buffer addresses on the accelerator-side bus). `rs2` is reserved for
  per-op auxiliary inputs (currently always zero / `x0`). `rd` receives a
  status word — `0` on success, otherwise a non-zero exception code (§0.10).
- **Architectural state.** Two pieces:
  1. The host's standard register file and PC (managed by the RISC-V core).
  2. The bytes held in the accelerator-side bus regions (§0.5). The
     accelerator has *no* general-purpose registers — its only addressable
     state is the SRAM banks and DRAM weight backing store.
- **Endianness.** RISC-V default little-endian for both instructions and
  data. F32 = IEEE-754 binary32 (`f32::to_le_bytes`); descriptor structs are
  packed little-endian.
- **XLEN.** Xemamba is XLEN-agnostic: the same encoding runs on RV32 and
  RV64 hosts. Pointers in `rs1` are XLEN bits; descriptor pointer fields
  follow the host's pointer width (use `uint32_t` on RV32, `uint64_t` on
  RV64 for the host-side `dram_addr` field; bus-side SRAM addresses are
  always 32-bit aliases, of which 23 bits are significant).
- **Concurrency.** v0.2 specifies functional semantics only. Each
  instruction completes atomically before the next host instruction
  retires. A future revision will add an asynchronous mode plus an
  explicit fence.

## 0.3 Data types — `dtype` field (3 bits, in descriptor)

| value   | mnemonic | bytes/element | status                                 |
|---------|----------|---------------|----------------------------------------|
| `0b000` | `FP32`   | 4             | implemented (v0.2)                     |
| `0b001` | `INT8`   | 1             | reserved; encode/decode only in v0.2   |
| `0b010` | `INT24`  | 3             | reserved (h_t state); decode only      |
| others  | —        | —             | reserved; raise `IllegalDtype` (§0.10) |

The Rust mirror is `emamba_core::Dtype` (`crates/core/src/lib.rs`).

## 0.4 Encoding — R-type with custom-0 opcode

Every Xemamba instruction is an ordinary RISC-V **R-type** word. The major
opcode is **`custom-0` = `0b0001011`** (= `0x0B`), per the RISC-V base
opcode map (Unprivileged ISA Manual, "RV32/64G Instruction Set Listings"
chapter — `inst[6:5]=00`, `inst[4:2]=010`, `inst[1:0]=11`).

```
 31      25 24    20 19    15 14    12 11     7 6        0
┌──────────┬────────┬────────┬────────┬────────┬───────────┐
│  funct7  │   rs2  │   rs1  │ funct3 │   rd   │  opcode   │
│  (7 b)   │  (5 b) │  (5 b) │  (3 b) │  (5 b) │ 0001011   │
└──────────┴────────┴────────┴────────┴────────┴───────────┘
```

Field meanings *for Xemamba* (overlays the standard R-type semantics):

| field   | width | use in v0.2                                              |
|---------|-------|----------------------------------------------------------|
| opcode  | 7     | `0b0001011` — custom-0; identifies the extension          |
| funct3  | 3     | major Xemamba opcode group (see §0.6)                     |
| funct7  | 7     | minor Xemamba opcode within the group                     |
| rs1     | 5     | x-reg holding pointer to descriptor (or 0 if op has none) |
| rs2     | 5     | x-reg holding aux input (always `x0` in v0.2)             |
| rd      | 5     | x-reg receiving status word (0 = success)                  |

`custom-1` (`0b0101011`), `custom-2` (`0b1011011`, also reserved for RV128),
and `custom-3` (`0b1111011`, also reserved for RV128) are unused by Xemamba
v0.2; the choice of `custom-0` follows the convention to leave the higher
custom slots free for future or coexisting vendor extensions.

## 0.5 Address space

Xemamba splits addresses into two windows:

1. **Host-side (RISC-V) addresses** — XLEN-wide pointers used in host code
   and in the `dram_addr` field of DMA descriptors. These are ordinary
   memory addresses serviced by the host's load/store unit and the
   accelerator's DMA engine.
2. **Accelerator-side bus addresses** — 32-bit aliases (23 significant
   bits) into the on-chip SRAM banks. These appear in compute-op
   descriptors and have no meaning to the host CPU.

### 0.5.1 Accelerator bus map (mamba-130m sizing)

| range                          | name        | size  | purpose                           |
|--------------------------------|-------------|-------|-----------------------------------|
| `0x0000_0000 – 0x003F_FFFF`    | W-SRAM      | 4 MB  | weight tile (ping-pong)           |
| `0x0040_0000 – 0x004F_FFFF`    | A-SRAM      | 1 MB  | activations (x, y, gate, Δ, B, C) |
| `0x0050_0000 – 0x006F_FFFF`    | STATE-SRAM  | 2 MB  | recurrent h_t + conv-cache        |

All on-chip-SRAM addresses fit in **23 bits** (max `0x6F_FFFF`).

### 0.5.2 Host-side window

| range                           | purpose                                            |
|---------------------------------|----------------------------------------------------|
| `0x1000_0000 – 0x2FFF_FFFF`*    | DRAM weight backing store (DMA endpoint)            |
| `0x8000_0000 – 0x8000_0007`*    | SimCtrl MMIO (write to halt; for the simulator)    |

*For the functional simulator. On a real SoC the DRAM window is wherever
the SoC's interconnect maps DDR; the SimCtrl device is omitted.*

### 0.5.3 Access rules

- Compute-op descriptors (`mac_tile`, `norm`, `conv_step`, `ssm_step`,
  `silu`, `exp`) MUST address SRAM only. A bus address outside W/A/STATE-SRAM
  raises `BusFault`.
- DMA-op descriptors cross between DRAM and any one SRAM region. They
  MUST NOT cross between two SRAMs in v0.2 (use two DMA ops).
- Address alignment: SRAM byte addresses must be aligned to `dtype.bytes()`.
  Misaligned access raises `Misaligned`.
- A read or write that begins in-region but extends past the region
  boundary raises `OutOfBounds`.

## 0.6 Instructions (`funct3` × `funct7`)

`funct3` selects the op group; `funct7` selects the specific op. All eight
v0.2 ops live under `funct3 = 0b000`. Future revisions may use other
`funct3` values for INT8/INT24 paths (see §0.13).

| funct3  | funct7   | mnemonic       | descriptor format | block consumed       |
|---------|----------|----------------|-------------------|----------------------|
| `0b000` | `0x01`   | `xemamba.mac`  | `mac_desc_t`       | MAC array            |
| `0b000` | `0x02`   | `xemamba.norm` | `norm_desc_t`      | Range-Norm unit      |
| `0b000` | `0x03`   | `xemamba.conv` | `conv_desc_t`      | Conv1D engine        |
| `0b000` | `0x04`   | `xemamba.ssm`  | `ssm_desc_t`       | SSM Scan + PWL-Exp   |
| `0b000` | `0x05`   | `xemamba.silu` | `pwl_desc_t`       | PWL-SiLU             |
| `0b000` | `0x06`   | `xemamba.exp`  | `pwl_desc_t`       | PWL-Exp              |
| `0b000` | `0x07`   | `xemamba.dmal` | `dma_desc_t`       | DMA load (DRAM→SRAM) |
| `0b000` | `0x08`   | `xemamba.dmas` | `dma_desc_t`       | DMA store (SRAM→DRAM)|

`funct3 = 0b000`, `funct7 ∉ {1..8}` raises `IllegalOpcode`.
`funct3 ≠ 0b000` raises `IllegalOpcode` in v0.2.

### 0.6.1 Calling convention (per-op uniform)

Every instruction follows the same shape:

```
xemamba.<mnemonic>  rd, rs1, rs2
```

- **`rs1`** — XLEN-wide host-memory pointer to the descriptor for this op.
  Must be aligned to 4 bytes. If the op has no descriptor (none in v0.2),
  pass `x0`.
- **`rs2`** — reserved; pass `x0` in v0.2. Future revisions may use it for
  a small immediate or a second descriptor.
- **`rd`** — receives the status word: `0` on success, otherwise a non-zero
  exception code from §0.10. Pass `x0` to discard.

The accelerator reads the descriptor from `rs1` via the host's memory
hierarchy (same coherence domain as the calling thread), executes the op,
and writes the status to `rd` before returning control. v0.2 instructions
are synchronous: the next host instruction observes the op's effects.

## 0.7 Descriptor layouts

All descriptors are `#[repr(C)]`-style packed structs, little-endian, with
the alignment indicated. Bus addresses are 32-bit unsigned (`u32`), of
which the low 23 bits index SRAM (§0.5). Element counts are `u16` unless
noted. `dtype` is the 3-bit code from §0.3, stored in a `u8` (high 5 bits
must be zero — non-zero high bits raise `MalformedOp`).

### 0.7.1 `mac_desc_t` — `xemamba.mac`

Computes `dst[m × n] = src_a[m × k] · src_b[k × n]`. All three operand
tiles use the same `dtype`, row-major.

```c
struct mac_desc_t {
    uint32_t src_a;   // SRAM byte address of src_a tile (m·k elements)
    uint32_t src_b;   // SRAM byte address of src_b tile (k·n elements)
    uint32_t dst;     // SRAM byte address of dst tile (m·n elements)
    uint16_t m;       // rows of src_a / dst, 1..=4096
    uint16_t n;       // cols of src_b / dst, 1..=4096
    uint16_t k;       // reduction dimension, 1..=4096
    uint8_t  dtype;   // §0.3
    uint8_t  reserved; // must be 0
};
// total: 20 bytes, 4-byte aligned
```

### 0.7.2 `norm_desc_t` — `xemamba.norm`

Range-normalize: `dst[i] = γ[i] · (src[i] − μ) / (max − min + ε) + β[i]`,
`ε = 1e-5`. See §0.9 for the canonical algorithm.

```c
struct norm_desc_t {
    uint32_t src;     // SRAM byte address (len elements)
    uint32_t dst;     // SRAM byte address (len elements)
    uint32_t gamma;   // SRAM byte address (len elements)
    uint32_t beta;    // SRAM byte address (len elements)
    uint16_t len;     // 1..=65535
    uint8_t  dtype;
    uint8_t  reserved;
};
// total: 20 bytes
```

### 0.7.3 `conv_desc_t` — `xemamba.conv`

One single-token step of depthwise causal Conv1D, channel count `len`,
kernel size `k_conv`. Reads `cache[(k_conv-1) · len]` (older taps), shifts
in `src[len]` (newest tap), writes `dst[len]` of per-channel dot products.
Cache updated in place.

```c
struct conv_desc_t {
    uint32_t src;       // newest x_t (len elements)
    uint32_t dst;       // conv output (len elements)
    uint32_t weights;   // weights (k_conv·len elements; weights[t·len+c] = tap t of chan c)
    uint32_t cache;     // conv-cache ((k_conv-1)·len elements; updated in place)
    uint16_t len;       // channel count (= ED), 1..=65535
    uint8_t  dtype;
    uint8_t  k_conv;    // kernel size (eMamba uses 4); 1..=63
};
// total: 20 bytes
```

### 0.7.4 `ssm_desc_t` — `xemamba.ssm`

Fused per-token SSM recurrence (architecture.md §2). Inputs are in a
**packed activation context** at `ctx_in`; output `y_t` written to
`ctx_out`. The state `state_h` lives in STATE-SRAM and is updated in place.

```c
struct ssm_desc_t {
    uint32_t ctx_in;    // packed input context (layout below)
    uint32_t state_h;   // h_t for this layer (ed·n elements)
    uint32_t ctx_out;   // output buffer (y_t[ed])
    uint16_t ed;        // channel count (= ED)
    uint8_t  n;         // state size (= N), 1..=255
    uint8_t  dtype;
};
// total: 16 bytes
```

`ctx_in` layout (offsets in elements, dtype `dtype`):

```
offset             field    size
0                  x_t      ed
ed                 delta    ed
2·ed               B        ed·n
2·ed + ed·n        C        ed·n
2·ed + 2·ed·n      A_diag   ed·n   (static; preloaded)
3·ed + 2·ed·n      D_skip   ed     (static; preloaded)
total              4·ed + 2·ed·n
```

### 0.7.5 `pwl_desc_t` — `xemamba.silu`, `xemamba.exp`

Element-wise PWL: `dst[i] = pwl(src[i])` for `i in 0..len`.

```c
struct pwl_desc_t {
    uint32_t src;    // SRAM byte address (len elements)
    uint32_t dst;    // SRAM byte address (len elements)
    uint16_t len;
    uint8_t  dtype;
    uint8_t  reserved;
};
// total: 12 bytes
```

### 0.7.6 `dma_desc_t` — `xemamba.dmal`, `xemamba.dmas`

Byte-granular copy between DRAM and an SRAM region. `dma_load` copies
DRAM → SRAM; `dma_store` copies SRAM → DRAM. `dtype` is ignored.

```c
struct dma_desc_t {
    uint64_t dram_addr;  // host-side byte address (XLEN-wide; on RV32, high 32 bits = 0)
    uint32_t sram_addr;  // accelerator-side byte address
    uint32_t bytes;      // transfer length in bytes; may be 0
};
// total: 16 bytes, 8-byte aligned (for the u64)
```

## 0.8 Per-op semantics

Identical to v0.1; reproduced here for self-containedness. Every op is
logically: (1) the accelerator reads the descriptor from `rs1`, (2)
performs the algorithm below, (3) writes the status word to `rd`.

### `xemamba.mac`

```
read mac_desc_t from M[rs1]
inputs:  src_a[m, k], src_b[k, n]   (SRAM)
output:  dst[m, n]                  (SRAM)
algorithm:
    for i in 0..m, j in 0..n:
        acc = 0.0_f32
        for r in 0..k:
            acc += src_a[i, r] * src_b[r, j]
        dst[i, j] = acc
```

Accumulation is left-to-right over `r` for golden-trace determinism.

### `xemamba.norm`

```
read norm_desc_t from M[rs1]
inputs:  src[len], gamma[len], beta[len]
output:  dst[len]
constants: eps = 1e-5_f32
algorithm (numerics::range_normalization):
    mu       = sum(src) / len
    centered = src - mu
    rng      = max(centered) - min(centered)
    dst      = gamma * centered / (rng + eps) + beta
```

### `xemamba.conv`

```
read conv_desc_t from M[rs1]
inputs:  src[len], weights[k_conv·len], cache[(k_conv-1)·len]
output:  dst[len]
side effect: cache updated in place — oldest tap dropped, src becomes newest.
algorithm (depthwise causal, groups = len):
    for c in 0..len:
        acc = weights[0·len + c] * src[c]                       // newest tap
        for t in 1..k_conv:
            acc += weights[t·len + c] * cache[(t-1)·len + c]
        dst[c] = acc
    // shift cache: cache := [src, cache[0..k_conv-2]] per channel
    for t in (1..k_conv-1).rev():
        cache[t·len + ..] = cache[(t-1)·len + ..]
    cache[0·len + ..] = src[..]
```

### `xemamba.ssm`

```
read ssm_desc_t from M[rs1]
inputs (from ctx_in): x_t[ed], delta[ed], B[ed,n], C[ed,n], A_diag[ed,n], D_skip[ed]
state (read/written): h[ed,n]                       (at state_h)
output (to ctx_out):  y_t[ed]
algorithm:
    delta_relu = relu(delta)                         // softplus → ReLU
    for i in 0..ed, j in 0..n:
        d_bar  = exp_pw(delta_relu[i] * A_diag[i,j])  // PWL-Exp inline
        b_bar  = delta_relu[i] * B[i,j] * x_t[i]
        h[i,j] = d_bar * h[i,j] + b_bar
    for i in 0..ed:
        acc = D_skip[i] * x_t[i]
        for j in 0..n:
            acc += C[i,j] * h[i,j]
        y_t[i] = acc
```

### `xemamba.silu` / `xemamba.exp`

```
read pwl_desc_t from M[rs1]
dst[0..len] = silu_pw(src[0..len])    (or exp_pw; §0.9)
```

### `xemamba.dmal` / `xemamba.dmas`

```
read dma_desc_t from M[rs1]
dmal: for b in 0..bytes: SRAM[sram_addr + b] = DRAM[dram_addr + b]
dmas: for b in 0..bytes: DRAM[dram_addr + b] = SRAM[sram_addr + b]
```

## 0.9 Numeric approximations (normative)

Unchanged from v0.1. A conformant implementation MUST produce
**bit-identical FP32 outputs** for bit-identical FP32 inputs, when compared
against `mamba/approximations.py`.

### 0.9.1 `relu(x) = max(x, 0.0_f32)`

NaN handling matches `f32::max` (`max(NaN, 0)` returns 0).

### 0.9.2 `range_normalization`

Per `xemamba.norm` semantics. `eps = 1e-5_f32`. Mean and min/max computed
in FP32 left-to-right; no Kahan compensation.

### 0.9.3 PWL construction (shared by SiLU and Exp)

```
breakpoints = linspace(lo, hi, n + 1)         // n+1 evenly spaced f32 values
values      = f(breakpoints)                  // n+1 evaluations, f32
slopes      = (values[1..] - values[..n]) / (breakpoints[1..] - breakpoints[..n])
intercepts  = values[..n] - slopes * breakpoints[..n]
```

Lookup of `pwl(x)`:

```
if x < lo:    return below_value(x)
if x > hi:    return above_value(x)
otherwise:    let i = clamp(partition_point(breakpoints, |bp| bp <= x) - 1, 0, n - 1)
              return slopes[i] * x + intercepts[i]
```

`partition_point(.., |bp| bp <= x) - 1` is the Rust analog of NumPy's
`searchsorted(side='right') - 1`.

### 0.9.4 `silu_pw`

```
target   silu(x) = x / (1 + exp(-x))    (in f32)
domain   lo = -7.0, hi = 7.0
segments 17
below(x) = 0.0
above(x) = x          (linear identity)
```

### 0.9.5 `exp_pw`

```
target   exp(x)
domain   lo = -4.0, hi = 1.0
segments 11
below(x) = 0.0
above(x) = e ≈ 2.7182817_f32   (= np.float32(np.e))
```

## 0.10 Exceptions

Xemamba reports exceptions through the `rd` status word. A non-zero status
indicates the op did not modify any architectural state (the descriptor is
treated as invalid; SRAM is unchanged). Status codes:

| code   | name             | raised when                                                |
|--------|------------------|------------------------------------------------------------|
| `0x00` | success          | op completed; results are committed                        |
| `0x01` | `IllegalOpcode`  | `funct3 ≠ 0` or `funct7` outside `{1..8}`                   |
| `0x02` | `IllegalDtype`   | `dtype ∈ {0b011..=0b111}`, or INT8/INT24 in v0.2            |
| `0x03` | `Misaligned`     | descriptor pointer or any SRAM address not aligned          |
| `0x04` | `BusFault`       | SRAM address outside any region in §0.5                     |
| `0x05` | `MalformedOp`    | reserved bits non-zero, or zero-valued `m`/`n`/`k`/`len`    |
| `0x06` | `OutOfBounds`    | a read/write extends past its region                        |
| `0x07` | `DescriptorFault`| reading the descriptor itself faulted (host-side)            |

A future revision MAY also raise standard RISC-V `IllegalInstruction`
traps for the same conditions when running in a "strict" mode.

## 0.11 Halt

In the functional simulator only: writing any value to host-side address
`0x8000_0000` halts the simulator after the current op completes. This is
not a Xemamba instruction — it is an ordinary `sw`/`sd` to the SimCtrl
MMIO region. On real hardware the host issues a software interrupt or
returns to its caller in the usual RISC-V manner.

## 0.12 Encoding examples

### Example 1: `xemamba.mac  x0, a0, x0`

A `mac_tile` whose descriptor pointer is in `a0` (= `x10`), discarding
status. Field values:

- `funct7 = 0x01`   (mac_tile)
- `rs2    = 0`      (`x0`, unused)
- `rs1    = 10`     (`a0`)
- `funct3 = 0b000`
- `rd     = 0`      (`x0`, discard status)
- `opcode = 0b0001011`  (custom-0)

Bit assembly:
```
inst = (0x01 << 25) | (0 << 20) | (10 << 15) | (0b000 << 12) | (0 << 7) | 0b0001011
     = 0x0205_000B
```

Decode check:
- bits[31:25] = `0000001` → funct7 = 1 ✓
- bits[24:20] = `00000`   → rs2 = 0 ✓
- bits[19:15] = `01010`   → rs1 = 10 ✓
- bits[14:12] = `000`     → funct3 = 0 ✓
- bits[11:7]  = `00000`   → rd = 0 ✓
- bits[6:0]   = `0001011` → custom-0 ✓

The descriptor pointed at by `a0` follows the §0.7.1 layout:

```c
mac_desc_t d = {
    .src_a = 0x000000,    // W-SRAM base
    .src_b = 0x004800,    // W-SRAM offset
    .dst   = 0x401000,    // A-SRAM
    .m     = 16,
    .n     = 16,
    .k     = 768,
    .dtype = 0,           // FP32
    .reserved = 0,
};
```

### Example 2: `xemamba.silu  x0, a1, x0`

A SiLU op with descriptor pointer in `a1` (= `x11`):

```
inst = (0x05 << 25) | (0 << 20) | (11 << 15) | (0b000 << 12) | (0 << 7) | 0b0001011
     = 0x0A05_800B
```

Descriptor (`pwl_desc_t`):

```c
pwl_desc_t d = { .src = 0x400000, .dst = 0x401000, .len = 1536, .dtype = 0, .reserved = 0 };
```

### Example 3: `xemamba.dmal  t0, a2, x0`

A DMA load with descriptor in `a2` (= `x12`), status into `t0` (= `x5`):

```
inst = (0x07 << 25) | (0 << 20) | (12 << 15) | (0b000 << 12) | (5 << 7) | 0b0001011
     = 0x0E06_028B
```

Descriptor (`dma_desc_t`):

```c
dma_desc_t d = {
    .dram_addr = 0x10000000,   // DRAM base
    .sram_addr = 0x000000,     // W-SRAM base
    .bytes     = 0x18000,      // 96 KB
};
```

After the instruction retires, `t0` holds `0` on success or one of the
codes from §0.10 on failure.

## 0.13 Versioning and reservation rules

This is **v0.2** (the first RISC-V-compatible revision). Future revisions
append a changelog at the bottom of this document and obey:

1. **Funct7 allocations are append-only.** A revision MAY assign a new
   `funct7` value in the `funct3=0` group; it MUST NOT reuse an
   already-allocated `funct7` for a different operation.
2. **`funct3` slots may grow.** Future revisions may allocate `funct3=001`
   (e.g., for INT8 paths) or higher; they MUST NOT redefine `funct3=000`
   semantics.
3. **Descriptor structs are append-only.** Reserved bytes/bits MAY take on
   meaning in a future revision. Existing fields keep their offsets and
   widths.

Decoders SHOULD report the spec version their executor implements, so a
loader can refuse to run a stream that requires unimplemented features.

---

## Changelog

- **v0.2** — reframe as RISC-V `Xemamba` non-standard extension. Encoding
  becomes 32-bit R-type with custom-0 opcode (`0b0001011`), `funct3` /
  `funct7` selectors, and per-op descriptor structs read from memory via
  `rs1`. Numerics, address-space rules, and per-op semantics unchanged
  from v0.1. Removes the v0.1 128-bit fixed-width encoding entirely.
- **v0.1** — initial release. 8 opcodes in a 128-bit fixed-width encoding,
  FP32-only, single-token decode. Superseded by v0.2.
