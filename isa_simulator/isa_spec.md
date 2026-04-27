# eMamba ISA — v0.1 specification

Status: draft (v0.1)
Companion to: `architecture.md` (the *why*); this document is the *what*.

## 0.1 Scope

This document is the architectural-state-machine contract for the eMamba
accelerator at the level a functional simulator (or assembler, or compiler
backend) needs to be conformant. It defines:

- the programmer's model (state, control flow, halt),
- a 3-bit data-type space and the only one currently realized (`FP32`),
- the 128-bit fixed-width instruction encoding,
- the 8-opcode ISA and one bit-table per format,
- per-op semantics in pseudocode,
- the four normative numeric approximations (range-norm, ReLU, PWL-SiLU,
  PWL-Exp) inherited from `approximations.py`,
- exception conditions,
- worked encoding examples.

The companion `architecture.md` motivates these choices from the eMamba paper
and the workload analysis in `workload/results/report.json`. Read it for
context; nothing in this document depends on it for correctness.

This is **v0.1**: FP32-only, decode-only single token, no branches. INT8/INT24
opcodes are reserved for future revisions and decode-only today.

## 0.2 Programmer's model

- **Op stream.** Programs are linear sequences of 128-bit ops, indexed by a
  32-bit program counter `pc`. There are no branches in v0.1: control flow is
  straight-line. The simulator advances `pc` by one op per `step()`.
- **Termination.** Execution halts on (a) `pc` reaching the end of the loaded
  op stream, or (b) any write to the SimCtrl region (§0.5). On halt, `pc` is
  observable.
- **Architectural state.** Two pieces:
  1. `pc: u32`.
  2. The bytes held in the address-mapped regions of §0.5. There is no
     general-purpose register file; ops name their operands by bus address.
- **Endianness.** Little-endian throughout. Multi-byte values in memory match
  `f32::to_le_bytes` / `u32::to_le_bytes`. The 128-bit instruction word is
  itself little-endian: bit 0 (the LSB) is the lowest bit of opcode, bit 127
  the MSB.
- **Concurrency.** v0.1 specifies functional semantics only. Ops execute one
  at a time, fully serialized; observed effects after `step()` returns are as
  if the op completed atomically.

## 0.3 Data types — `Dtype` field (3 bits)

| value   | mnemonic | bytes/element | status                                 |
|---------|----------|---------------|----------------------------------------|
| `0b000` | `FP32`   | 4             | implemented (v0.1)                     |
| `0b001` | `INT8`   | 1             | reserved; encode/decode only in v0.1   |
| `0b010` | `INT24`  | 3             | reserved (h_t state); decode only      |
| others  | —        | —             | reserved; raise `IllegalDtype`         |

When a v0.1 implementation encounters a reserved-but-decodable dtype on a
compute op, it MUST raise `IllegalDtype`. Decoders (disassemblers) MAY accept
any of `FP32`/`INT8`/`INT24` without error.

The `core::Dtype` enum (`crates/core/src/lib.rs`) is the canonical Rust
mirror; `Dtype::bytes()` returns the byte count above.

## 0.4 Common encoding header

All ops are exactly **128 bits** (16 bytes), little-endian. The lowest byte
(bits `[7:0]`) is the opcode. The remaining 120 bits are format-specific
(§0.7). Reserved bits in any format MUST be written as 0; an op with non-zero
reserved bits raises `MalformedOp`.

```
 bit 127                                                                  bit 0
 ┌────────────────────────────────────────────────────────────────┬───────────┐
 │                  format-specific fields (120 bits)             │  opcode   │
 │                                                                │   (8)     │
 └────────────────────────────────────────────────────────────────┴───────────┘
```

## 0.5 Address space (bus map, mamba-130m sizing)

| range                          | name        | size  | purpose                           |
|--------------------------------|-------------|-------|-----------------------------------|
| `0x0000_0000 – 0x003F_FFFF`    | W-SRAM      | 4 MB  | weight tile (ping-pong)           |
| `0x0040_0000 – 0x004F_FFFF`    | A-SRAM      | 1 MB  | activations (x, y, gate, Δ, B, C) |
| `0x0050_0000 – 0x006F_FFFF`    | STATE-SRAM  | 2 MB  | recurrent h_t + conv-cache        |
| `0x1000_0000 – 0x2FFF_FFFF`    | DRAM        | 512 MB| weight backing store               |
| `0x8000_0000 – 0x8000_0007`    | SimCtrl     | 8 B   | write-anything-to-halt             |

All on-chip-SRAM addresses fit in **23 bits** (max `0x6F_FFFF`). DRAM and
SimCtrl require 32-bit absolute addresses and are reachable only through
DMA ops or the explicit halt write.

**Access rules.**
- Compute ops (`mac_tile`, `norm`, `conv_step`, `ssm_step`, `silu`, `exp`)
  MUST address SRAM only. A compute-op address outside W/A/STATE-SRAM
  raises `BusFault`.
- DMA ops (`dma_load`, `dma_store`) cross between DRAM and any SRAM region.
  They MUST NOT cross between two SRAMs in v0.1 (use two DMA ops).
- Address alignment: byte addresses must be aligned to `dtype.bytes()`.
  Unaligned access raises `Misaligned`.
- A read or write that begins in-region but extends past the region boundary
  raises `OutOfBounds`.

## 0.6 Opcode table

| opcode | mnemonic    | format | block consumed (architecture.md §1) | one-line semantics                                |
|--------|-------------|--------|-------------------------------------|---------------------------------------------------|
| `0x01` | `mac_tile`  | M      | MAC array                            | `dst[m,n] = src_a[m,k] · src_b[k,n]`              |
| `0x02` | `norm`      | N      | Range-Norm unit                      | range-normalize `src` with `gamma`, `beta`         |
| `0x03` | `conv_step` | C      | Conv1D engine                        | one depthwise causal conv step over conv-cache     |
| `0x04` | `ssm_step`  | S      | SSM Scan unit + PWL-Exp              | fused recurrence; updates `state_h` in place       |
| `0x05` | `silu`      | A      | PWL-SiLU                             | `dst = silu_pw(src)` (17-seg PWL, [-7, 7])         |
| `0x06` | `exp`       | A      | PWL-Exp                              | `dst = exp_pw(src)` (11-seg PWL, [-4, 1])          |
| `0x07` | `dma_load`  | D      | DMA / W-pre-fetch                    | DRAM → SRAM byte copy                              |
| `0x08` | `dma_store` | D      | DMA                                  | SRAM → DRAM byte copy                              |

Opcodes `0x00` and `0x09…0xFF` are reserved; raise `IllegalOpcode`.

## 0.7 Per-format bit layouts

All address fields are byte addresses. All `len`, `m`, `n`, `k`, and `bytes`
fields are unsigned. `dtype` is the 3-bit code from §0.3.

### Format M — `mac_tile` (opcode `0x01`)

Computes `dst[m × n] = src_a[m × k] · src_b[k × n]`.

```
bits        width  field     type      meaning
[7:0]       8      opcode    u8        must be 0x01
[30:8]      23     src_a     SRAM addr base of src_a tile (m·k elements)
[53:31]     23     src_b     SRAM addr base of src_b tile (k·n elements)
[76:54]     23     dst       SRAM addr base of dst tile (m·n elements)
[92:77]     16     m         u16       rows of src_a / dst
[108:93]    16     n         u16       cols of src_b / dst
[124:109]   16     k         u16       reduction dimension
[127:125]   3      dtype     Dtype     element type
```

All three operand tiles use the same `dtype`. Tile elements are stored in
row-major order. `m`, `n`, `k` MUST each be in `1..=4096`; zero raises
`MalformedOp`.

### Format N — `norm` (opcode `0x02`)

Computes `dst[i] = gamma[i] · (src[i] − mean(src)) / (max(centered) − min(centered) + ε) + beta[i]`,
with `ε = 1e-5`, applied along the only axis. See §0.9 for the canonical
algorithm.

```
bits        width  field     type      meaning
[7:0]       8      opcode    u8        must be 0x02
[30:8]      23     src       SRAM addr base of src vector (len elements)
[53:31]     23     dst       SRAM addr base of dst vector (len elements)
[76:54]     23     gamma     SRAM addr base of γ vector (len elements)
[99:77]     23     beta      SRAM addr base of β vector (len elements)
[115:100]   16     len       u16       vector length
[118:116]   3      dtype     Dtype     element type
[127:119]   9      reserved  —         must be 0
```

### Format C — `conv_step` (opcode `0x03`)

One single-token step of depthwise causal Conv1D, channel count `len`, kernel
size `k_conv`. Reads `cache[(k_conv-1) · len]` (older taps), shifts in `src[len]`
(newest tap), writes `dst[len]` per-channel dot products against
`weights[k_conv · len]`. The cache is updated in place (oldest tap evicted).

```
bits        width  field     type      meaning
[7:0]       8      opcode    u8        must be 0x03
[30:8]      23     src       SRAM addr base of new x_t (len elements)
[53:31]     23     dst       SRAM addr base of conv output (len elements)
[76:54]     23     weights   SRAM addr base of weights (k_conv·len elements)
[99:77]     23     cache     SRAM addr base of conv-cache ((k_conv-1)·len elements; updated in place)
[115:100]   16     len       u16       number of channels (= ED)
[118:116]   3      dtype     Dtype     element type
[124:119]   6      k_conv    u8        kernel size (eMamba uses 4); 1..=63
[127:125]   3      reserved  —         must be 0
```

### Format S — `ssm_step` (opcode `0x04`)

Fused per-token SSM recurrence (architecture.md §2). Inputs are passed via a
**packed activation context** `ctx_in` and outputs via `ctx_out`. The state
`state_h` lives in STATE-SRAM and is updated in place.

```
bits        width  field     type      meaning
[7:0]       8      opcode    u8        must be 0x04
[30:8]      23     ctx_in    SRAM addr packed input context (layout below)
[53:31]     23     state_h   SRAM addr h_t for this layer (ed·n elements)
[76:54]     23     ctx_out   SRAM addr output buffer (layout below)
[92:77]     16     ed        u16       channel count (= ED)
[100:93]    8      n         u8        state size (= N); 1..=255
[103:101]   3      dtype     Dtype     element type
[127:104]   24     reserved  —         must be 0
```

`ctx_in` layout (offsets in elements, dtype `dtype`):
```
offset             field    size
0                  x_t      ed
ed                 delta    ed
2·ed               B        ed·n
2·ed + ed·n        C        ed·n
2·ed + 2·ed·n      A_diag   ed·n   (static; preloaded by the host or prior DMA)
3·ed + 2·ed·n      D_skip   ed     (static; preloaded)
total              4·ed + 2·ed·n
```

`ctx_out` layout: a single `y_t[ed]` buffer.

### Format A — `silu` / `exp` (opcodes `0x05`, `0x06`)

Element-wise PWL activation: `dst[i] = pwl(src[i])` for `i in 0..len`.

```
bits        width  field     type      meaning
[7:0]       8      opcode    u8        0x05 (silu) or 0x06 (exp)
[30:8]      23     src       SRAM addr base of src (len elements)
[53:31]     23     dst       SRAM addr base of dst (len elements)
[69:54]     16     len       u16       element count
[72:70]     3      dtype     Dtype     element type
[127:73]    55     reserved  —         must be 0
```

### Format D — `dma_load` / `dma_store` (opcodes `0x07`, `0x08`)

Byte-granular copy between DRAM and an SRAM region.

```
bits        width  field      type        meaning
[7:0]       8      opcode     u8          0x07 (load) or 0x08 (store)
[39:8]      32     dram_addr  u32         DRAM byte address
[62:40]     23     sram_addr  SRAM addr   SRAM byte address
[94:63]     32     bytes      u32         transfer length in bytes
[127:95]    33     reserved   —           must be 0
```

`dma_load` copies `bytes` from `dram_addr` to `sram_addr`. `dma_store` does the
reverse. `bytes` MAY be zero (no-op). Either endpoint extending past its region
raises `OutOfBounds`.

## 0.8 Per-op semantics

The semantics below define functional behavior. They reference the four
normative numeric primitives in §0.9 by name; implementations MUST use those
primitives, not generic alternatives (e.g., not stdlib `silu`).

### `mac_tile`

```
inputs:  src_a[m, k], src_b[k, n]
output:  dst[m, n]
algorithm:
    for i in 0..m:
        for j in 0..n:
            acc = 0.0_f32
            for r in 0..k:
                acc += src_a[i, r] * src_b[r, j]
            dst[i, j] = acc
validation: none (pure FP32 matmul)
```

Accumulation order is row-major / inner-product. Implementations MAY use FMA
or restructured loops only if results are bit-identical to the canonical
left-to-right summation above. v0.1 mandates the canonical order to keep
golden traces deterministic.

### `norm`

```
inputs:  src[len], gamma[len], beta[len]
output:  dst[len]
constants: eps = 1e-5_f32
algorithm (architecture.md §1.1; approximations.py:range_normalization):
    mu       = sum(src) / len
    centered = src - mu
    rng      = max(centered) - min(centered)
    dst      = gamma * centered / (rng + eps) + beta
validation: numerics::range_normalization
```

### `conv_step`

```
inputs:  src[len]                                  (newest x_t per channel)
         weights[k_conv * len]                     (per-channel filter taps;
                                                    weights[t * len + c] is tap t of channel c)
         cache[(k_conv-1) * len]                   (channel-major prior taps;
                                                    cache[t * len + c] is older-by-(t+1) tap of channel c)
output:  dst[len]
side effect: cache is updated in place — the oldest tap is dropped, src
             becomes the newest tap.
algorithm (depthwise causal, groups = len):
    for c in 0..len:
        // dot product: weights[0,c] is the newest-tap weight, weights[k-1,c] the oldest
        acc = weights[0 * len + c] * src[c]
        for t in 1..k_conv:
            acc += weights[t * len + c] * cache[(t - 1) * len + c]
        dst[c] = acc
    // shift: cache := [src, cache[0..k_conv-2]]  (channel-wise)
    for t in (1..k_conv-1).rev():
        cache[t * len + ..] = cache[(t - 1) * len + ..]
    cache[0 * len + ..] = src[..]
validation: none (pure FP32)
```

### `ssm_step`

```
inputs (read from ctx_in, see §0.7 layout):
    x_t[ed], delta[ed], B[ed, n], C[ed, n], A_diag[ed, n], D_skip[ed]
state (read/written at state_h):
    h[ed, n]
output (written to ctx_out):
    y_t[ed]
algorithm (architecture.md §2):
    delta_relu = relu(delta)                          // softplus → ReLU (Eq. 4 substitution)
    for i in 0..ed:
        for j in 0..n:
            d_bar = exp_pw(delta_relu[i] * A_diag[i, j])    // PWL-Exp inline
            b_bar = delta_relu[i] * B[i, j] * x_t[i]
            h[i, j] = d_bar * h[i, j] + b_bar
    for i in 0..ed:
        acc = D_skip[i] * x_t[i]
        for j in 0..n:
            acc += C[i, j] * h[i, j]
        y_t[i] = acc
validation: numerics::{relu, exp_pw}
```

State `h` and ctx tensors are stored row-major: `h[i, j]` is at offset
`(i * n + j) * dtype.bytes()`.

### `silu`

```
inputs:  src[len]
output:  dst[len] = silu_pw(src)
validation: numerics::silu_pw   // §0.9
```

### `exp`

```
inputs:  src[len]
output:  dst[len] = exp_pw(src)
validation: numerics::exp_pw    // §0.9
```

### `dma_load` / `dma_store`

Byte copy. Reads/writes ignore `dtype`. Bytes are transferred in ascending
address order; overlapping endpoints have undefined behavior in v0.1 (the
op is conceptually atomic, so this only matters if a single DMA aliases
itself, which is not legal).

```
dma_load:  for b in 0..bytes: sram[sram_addr + b] = dram[dram_addr + b]
dma_store: for b in 0..bytes: dram[dram_addr + b] = sram[sram_addr + b]
```

## 0.9 Numeric approximations (normative)

The four functions below are part of the ISA contract. A conformant
implementation MUST produce **bit-identical FP32 outputs** for bit-identical
FP32 inputs, when compared against `mamba/approximations.py`. This is testable
in CI via golden `.npy` fixtures.

### 0.9.1 `relu`

```
relu(x) = max(x, 0.0_f32)
```

NaN handling matches `f32::max`: `max(NaN, 0)` returns 0 in v0.1
(softplus → ReLU substitution; Mamba inputs do not produce NaN here).

### 0.9.2 `range_normalization`

Per §0.8 `norm`. Mean and min/max computed in FP32 left-to-right summation;
no Kahan compensation. `eps = 1e-5_f32`. `gamma` and `beta` are vectors of
the same length as `src`.

### 0.9.3 PWL construction (shared by SiLU and Exp)

Given a target function `f`, a domain `[lo, hi]`, and a segment count `n`:

```
breakpoints = linspace(lo, hi, n + 1)         // n+1 evenly spaced f32 values, endpoints inclusive
values      = f(breakpoints)                  // n+1 evaluations in f32
slopes      = (values[1..] - values[..n]) / (breakpoints[1..] - breakpoints[..n])
intercepts  = values[..n] - slopes * breakpoints[..n]
```

Lookup of `pwl(x)`:

```
if x < lo:    return below_value(x)
if x > hi:    return above_value(x)
otherwise:    let i = max(0, min(n - 1, partition_point(breakpoints, |bp| bp <= x) - 1))
              return slopes[i] * x + intercepts[i]
```

`partition_point(.., |bp| bp <= x) - 1` is the Rust analog of NumPy's
`searchsorted(side='right') - 1`. The clamp to `[0, n-1]` is required to
handle `x == hi` exactly (which lands on the right endpoint).

### 0.9.4 `silu_pw`

```
target:    silu(x) = x / (1 + exp(-x))    (computed in f32)
domain:    lo = -7.0, hi = 7.0
segments:  17
below(x):  0.0
above(x):  x          (linear identity — silu is approximately linear above 7)
```

### 0.9.5 `exp_pw`

```
target:    exp(x)
domain:    lo = -4.0, hi = 1.0
segments:  11
below(x):  0.0
above(x):  e ≈ 2.7182817_f32   (= np.float32(np.e))
```

## 0.10 Exceptions

| name             | raised when                                                 |
|------------------|-------------------------------------------------------------|
| `IllegalOpcode`  | opcode byte is not in §0.6                                  |
| `IllegalDtype`   | `dtype` field is `0b011..=0b111`, or is INT8/INT24 in v0.1  |
| `Misaligned`     | any address field is not aligned to `dtype.bytes()`         |
| `BusFault`       | any address falls outside every region in §0.5              |
| `MalformedOp`    | reserved bits non-zero, or zero-valued `m`/`n`/`k`/`len`    |
| `OutOfBounds`    | a read/write extends past its region                        |

Exceptions are reported by `OpExecutor::step()` returning `Err(OpException)`.
They do not modify architectural state (the op is treated as a no-op for
state-update purposes; `pc` does not advance).

## 0.11 Halt

Writing any value to address `0x8000_0000` halts the simulator after the
current op completes. The runner observes halt by polling the SimCtrl
device (analogous to rrs's `SimulationCtrlDevice`). At halt, `pc` points to
the op *after* the halting op.

## 0.12 Encoding examples

Two worked examples; constructed via field shifts so the reader can verify
their decoder.

### Example 1: `silu` (Format A, opcode 0x05)

Mnemonic: `silu  src=0x400000  dst=0x401000  len=1536  fp32`

Field values:
- `opcode  = 0x05`
- `src     = 0x400000` (A-SRAM base)
- `dst     = 0x401000`
- `len     = 1536`     (= ED)
- `dtype   = 0`         (FP32)

Construction:
```
bits = (0x05u128 <<  0)
     | (0x400000u128 << 8)
     | (0x401000u128 << 31)
     | (1536u128    << 54)
     | (0u128       << 70);
```

Equivalent value: `0x60_0000_0000_0000_0080_2000_0040_0000_05` (134-bit
literal trimmed to 128 bits little-endian; verify with the test harness in
`isa/src/op_formats.rs`).

Decode:
- bits[7:0]    = 0x05    → opcode = silu
- bits[30:8]   = 0x400000 → src
- bits[53:31]  = 0x401000 → dst
- bits[69:54]  = 0x600    → len = 1536
- bits[72:70]  = 0b000    → dtype = FP32

### Example 2: `mac_tile` (Format M, opcode 0x01)

Mnemonic: `mac_tile  s_a=0x000000  s_b=0x004800  dst=0x401000  m=16  n=16  k=768  fp32`

Field values:
- `opcode  = 0x01`
- `src_a   = 0x000000`  (W-SRAM base)
- `src_b   = 0x004800`  (W-SRAM offset)
- `dst     = 0x401000`  (A-SRAM)
- `m       = 16`
- `n       = 16`
- `k       = 768`        (= D)
- `dtype   = 0`          (FP32)

Construction:
```
bits = (0x01u128 <<  0)
     | (0x000000u128 <<  8)
     | (0x004800u128 << 31)
     | (0x401000u128 << 54)
     | (   16u128    << 77)
     | (   16u128    << 93)
     | (  768u128    << 109)
     | (    0u128    << 125);
```

Verifier in code: `MacTileOp::new(bits)` MUST round-trip these field values
exactly.

### Example 3: `dma_load` (Format D, opcode 0x07)

Mnemonic: `dma_load  dram=0x10000000  sram=0x000000  bytes=0x18000`
(loads 96 KB of weights from DRAM base into W-SRAM base)

Field values:
- `opcode    = 0x07`
- `dram_addr = 0x10000000` (DRAM base)
- `sram_addr = 0x000000`   (W-SRAM base)
- `bytes     = 0x18000`    (= 98304)

Construction:
```
bits = (0x07u128       << 0)
     | (0x10000000u128 << 8)
     | (0x000000u128   << 40)
     | (0x18000u128    << 63);
```

## 0.13 Versioning

This is **v0.1**. Future revisions append a changelog at the bottom of this
document and obey two compatibility rules:

1. **Opcode allocations are append-only.** A revision MAY assign a new
   opcode to a new instruction; it MUST NOT reuse a v0.1 opcode for a
   different operation.
2. **Reserved bits MAY become defined.** Any reserved bit field in v0.1
   MAY take on meaning in a future revision; v0.1 implementations that
   require zero reserved bits will then reject the new encoding (which is
   the intended behavior — they don't understand the new field).

Decoders SHOULD report the spec version their executor implements, so a
loader can refuse to run a stream that requires unimplemented features.

---

## Changelog

- **v0.1** — initial release. 8 opcodes, FP32-only, single-token decode.
