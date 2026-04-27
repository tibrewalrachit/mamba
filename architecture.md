# eMamba: implementation & simulator architecture

Status: draft
Scope: a single picture of (a) the eMamba accelerator we're modeling and (b) the
simulator that drives it. The two share a vocabulary — same op categories, same
on-chip resources, same dataflow — so they're documented together.

The op categories used everywhere below are the ones `workload/analyze_perf.py`
emits and that the eMamba paper Section 3 uses: `embedding`, `norm`, `in_proj`,
`conv1d`, `x_proj`, `dt_proj`, `ssm_scan_etc`, `out_proj`, `lm_head`.

---

## 1. eMamba accelerator — top-level

```
                              ┌──────────────────────────────────────┐
                              │              HOST / DRIVER           │
                              │  load weights · stream tokens · poll │
                              └───────────────────┬──────────────────┘
                                                  │ AXI / PCIe
              ┌───────────────────────────────────┼───────────────────────────────────┐
              │                          eMamba accelerator (ASIC)                    │
              │                                   │                                   │
              │   ┌───────────────────────────────▼─────────────────────────────────┐ │
              │   │                     OFF-CHIP DRAM (DDR4 / LPDDR5)               │ │
              │   │   weights (~516 MB for mamba-130m)  ·  KV-free, stateful Δ-LRU  │ │
              │   └────────────┬────────────────────────────────┬───────────────────┘ │
              │                │ weight reads (decode-bound)    │ activation spill    │
              │                ▼                                ▼                     │
              │   ┌─────────────────────────┐   ┌──────────────────────────────────┐  │
              │   │        DMA / WPF         │   │           ACT SPILL DMA          │  │
              │   │  double-buffered tiles   │   │  prefill activations only        │  │
              │   └────────────┬─────────────┘   └──────────────────┬───────────────┘  │
              │                │                                    │                  │
              │   ┌────────────▼────────────────────────────────────▼──────────────┐  │
              │   │                    ON-CHIP SRAM (banked)                       │  │
              │   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │  │
              │   │  │  W-SRAM     │  │  A-SRAM     │  │   STATE-SRAM         │    │  │
              │   │  │  weight tile│  │  x, y, gate │  │   h_t  (M × ED × N)  │    │  │
              │   │  │  (ping-pong)│  │  Δ, B, C    │  │   conv-cache (k-1)   │    │  │
              │   │  └──────┬──────┘  └──────┬──────┘  └──────────┬───────────┘    │  │
              │   └─────────┼────────────────┼────────────────────┼────────────────┘  │
              │             │                │                    │                    │
              │   ┌─────────▼──────┐  ┌──────▼─────────┐  ┌───────▼────────────────┐  │
              │   │   MAC ARRAY    │  │   RANGE-NORM   │  │     SSM SCAN UNIT      │  │
              │   │   (systolic)   │  │      UNIT      │  │  (fused recurrence)    │  │
              │   │                │  │                │  │                        │  │
              │   │ in_proj        │  │  μ, max, min   │  │  Δ̄ = exp(Δ·A)  [PWL]   │  │
              │   │ x_proj         │  │  γ·(x-μ)/range │  │  B̄ = Δ·B·x_t           │  │
              │   │ dt_proj        │  │  + β           │  │  h_t = Δ̄⊙h + B̄⊙x_t    │  │
              │   │ out_proj       │  │  (Eq. 2)       │  │  y_t = C·h_t + D·x_t   │  │
              │   │ lm_head        │  │                │  │  reads STATE-SRAM,     │  │
              │   │                │  │                │  │  writes back in place  │  │
              │   └────────┬───────┘  └────────┬───────┘  └───────────┬────────────┘  │
              │            │                   │                      │                │
              │            ▼                   ▼                      ▼                │
              │   ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
              │   │  CONV1D ENGINE │  │   PWL-SiLU     │  │       PWL-EXP          │  │
              │   │  depthwise k=4 │  │  17 segs       │  │  11 segs               │  │
              │   │  causal,       │  │  [-7, 7]       │  │  [-4, 1]               │  │
              │   │  groups = ED   │  │  approximations│  │  approximations        │  │
              │   │                │  │      .py       │  │      .py               │  │
              │   └────────────────┘  └────────────────┘  └────────────────────────┘  │
              │                                                                       │
              │   ┌─────────────────────────────────────────────────────────────────┐ │
              │   │                  TOKEN / SEQUENCE CONTROLLER                    │ │
              │   │   - per-layer FSM (M=24 layers for mamba-130m)                  │ │
              │   │   - prefill mode: stream L tokens through conv1d + scan          │ │
              │   │   - decode mode:  one token, conv-cache shift, single-step scan  │ │
              │   │   - emits MAC / scan / norm / activation micro-ops               │ │
              │   └─────────────────────────────────────────────────────────────────┘ │
              └───────────────────────────────────────────────────────────────────────┘
```

Sizing for the reference workload (mamba-130m-hf):
`D=768, ED=1536, N=16, d_conv=4, dt_rank=48, M=24 layers, vocab=50280`.

### 1.1 Why those blocks, in this order

Driven by `workload/results/report.txt`:

- **Decode** (the regime eMamba targets) is memory-bound at AI ≈ 0.5 F/B for every
  projection. → MAC array sized for *bandwidth match*, not peak FLOPs; W-SRAM
  ping-pong hides DRAM latency under the next tile's compute.
- **`ssm_scan_etc` is 41 % of decode time.** → it gets its own datapath with the
  recurrent state pinned in STATE-SRAM (no DRAM round-trip per token). PWL-Exp is
  fused inline so Δ̄ never materializes.
- **`conv1d` dominates HF prefill (95 %)** as a CPU artifact, not an ASIC one.
  On hardware it's a ~ED-MAC depthwise unit; small block, off the critical path.
- **`norm`** is range-norm (Eq. 2 of the paper, see `approximations.py`), not
  LayerNorm. No reciprocal-sqrt unit needed — just min/max trees.
- **`embedding` and `lm_head`** are weight-tied in mamba-130m. The 154 MB tensor
  is fetched once per token through the same MAC array path as out_proj.

---

## 2. SSM scan datapath (the load-bearing block)

The scan unit is the only block where Rust-port performance work will pay off
disproportionately, so it's worth a separate picture. Per Mamba step:

```
                    x_t (ED)            Δ (ED)              A (ED×N, static)
                       │                  │                       │
                       │                  ├────────┐              │
                       ▼                  ▼        ▼              ▼
                  ┌─────────┐        ┌──────────────────┐   ┌──────────┐
                  │ B (ED·N)│        │   Δ · A  (mul)   │   │  ReLU    │  (Δ = ReLU(dt_proj(x))
                  │  read   │        └────────┬─────────┘   │ for soft │   per Eq. 4 — softplus
                  └────┬────┘                 │             │  plus    │   replaced by ReLU)
                       │                      ▼             └──────────┘
                       ▼                ┌──────────────┐
                  ┌─────────┐           │  PWL-EXP     │   <-- 11-seg LUT, [-4, 1]
                  │ Δ · B   │           │  Δ̄ = e^{ΔA}  │
                  │ (ED·N)  │           └──────┬───────┘
                  └────┬────┘                  │
                       ▼                       ▼
                  ┌─────────┐             ┌─────────────┐
                  │ B̄ · x_t │             │  Δ̄ ⊙ h_{t-1}│ <── STATE-SRAM read (ED·N)
                  └────┬────┘             └──────┬──────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                              ┌──────────┐
                              │   add    │   →  h_t  ── STATE-SRAM write (in place)
                              └────┬─────┘
                                   ▼
                              ┌──────────┐         C (ED·N) ─┐
                              │  C · h_t │  ◄────────────────┘
                              └────┬─────┘
                                   │     D·x_t ──┐  (skip)
                                   ▼             ▼
                              ┌──────────────────┐
                              │     y_t          │
                              └────────┬─────────┘
                                       │     gate (ED) ──► ┌─────────┐
                                       │                   │ PWL-SiLU│
                                       ▼                   └────┬────┘
                                ┌──────────────┐                │
                                │  silu(gate)  │ ⊙   y_t  ◄─────┘
                                └──────┬───────┘
                                       ▼
                                  out_proj input  (ED)
```

Flop / byte budget per token (matches the formulas in `analyze_perf.py:167-172`):

```
ssm_flops/token = (2+3+3+2)·ED·N + 4·ED         = 250 K  (ED=1536, N=16)
ssm_bytes/token = (5·ED·N + 2·ED) · 4 B          = 503 KB
arithmetic intensity                              ≈ 0.50 F/B   ← bandwidth-bound
```

This is why the state must live on-chip: at AI = 0.5 F/B, even one DRAM round-
trip per token would dominate. STATE-SRAM size = `M · ED · N · 2 B` ≈ **1.13 MB**
for mamba-130m, well within a single SRAM bank.

---

## 3. Simulator stack

How the pieces in the repo fit together as a simulator. Existing files in **bold**;
the rest is what we're going to build.

```
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 1. WORKLOAD CAPTURE                                       (Python, CPU) │
   │                                                                         │
   │   ┌──────────────────────────┐       ┌──────────────────────────────┐   │
   │   │ workload/run_workload.py │       │   workload/analyze_perf.py   │   │
   │   │  HF mamba-130m-hf inf.   │──────▶│  hooks → per-op events:      │   │
   │   │  prompt + 10 decode      │       │  {category, shape, FLOPs,    │   │
   │   │  steps                   │       │   bytes, weight_bytes, …}    │   │
   │   └──────────────────────────┘       └─────────────┬────────────────┘   │
   │                                                    │                    │
   │                                                    ▼                    │
   │                                       workload/results/report.json     │
   │                                       (op-trace + analytical SSM rows)  │
   └─────────────────────────────────────────────────────┬───────────────────┘
                                                         │
                                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 2. NUMERIC REFERENCE                                      (Python, CPU) │
   │                                                                         │
   │   ┌──────────────────────────┐                                          │
   │   │   approximations.py      │   range_norm  ·  silu_pw  ·  exp_pw      │
   │   │  (Eq. 2, §4 of paper)    │   ReLU-for-softplus                      │
   │   └──────────────────────────┘                                          │
   │                  │                                                      │
   │                  ▼                                                      │
   │   golden tensors per op  (used by §3-step-3 to bit-check the ASIC model)│
   └─────────────────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 3. CYCLE-DRIVEN SIMULATOR                                       (Rust)  │
   │                                                                         │
   │   ┌─────────────────────────────────────────────────────────────────┐   │
   │   │                       OP-GRAPH SCHEDULER                        │   │
   │   │   - reads report.json, builds per-layer DAG                     │   │
   │   │   - tiles in_proj/out_proj/x_proj/dt_proj/lm_head for MAC array │   │
   │   │   - emits micro-ops: MAC_TILE, NORM, CONV_STEP, SSM_STEP,       │   │
   │   │                       SILU, EXP, DMA_LOAD, DMA_STORE            │   │
   │   └────────────────────────────────┬────────────────────────────────┘   │
   │                                    │                                    │
   │                                    ▼                                    │
   │   ┌─────────────────────────────────────────────────────────────────┐   │
   │   │             eMamba ACCELERATOR MODEL (simulators/)              │   │
   │   │                                                                 │   │
   │   │   MAC array · Range-Norm · SSM-Scan · PWL-SiLU · PWL-Exp ·      │   │
   │   │   Conv1D · DMA · SRAM banks · controller FSM                    │   │
   │   │                                                                 │   │
   │   │   each block = a Tick + (latency, throughput, ports) model;     │   │
   │   │   numerics call into approximations.py via FFI for validation   │   │
   │   └────────────────────────────────┬────────────────────────────────┘   │
   │                                    │ DMA / spill requests                │
   │                                    ▼                                    │
   │   ┌─────────────────────────────────────────────────────────────────┐   │
   │   │              MEMORY SYSTEM  (Ramulator 2 → Rust port)           │   │
   │   │                                                                 │   │
   │   │   per ramulator2_rust_design.md §2:                             │   │
   │   │     IFrontEnd  ──send()──▶  IMemorySystem                       │   │
   │   │                              ├── IAddrMapper                    │   │
   │   │                              ├── IDRAM   (DDR4 / LPDDR5)        │   │
   │   │                              └── Vec<IDRAMController>           │   │
   │   │                                    ├── IScheduler  (FRFCFS)     │   │
   │   │                                    └── IRefreshManager          │   │
   │   │                                                                 │   │
   │   │   the eMamba model IS the IFrontEnd: its DMAs replace SimpleO3  │   │
   │   └────────────────────────────────┬────────────────────────────────┘   │
   │                                    │                                    │
   │                                    ▼                                    │
   │                         ┌──────────────────────┐                        │
   │                         │  CYCLE TICK LOOP     │   single-thread        │
   │                         │  lcm(f_ratio, m_rat) │   (per §7 of design)   │
   │                         └──────────────────────┘                        │
   └────────────────────────────────────────────────────┬────────────────────┘
                                                        │
                                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 4. STATS & VALIDATION                                                   │
   │                                                                         │
   │   per-op cycles · MAC utilization · DRAM BW · SRAM occupancy            │
   │   bit-exact diff vs. approximations.py golden                           │
   │   end-to-end ms/token vs. report.txt baseline (104.5 ms/token CPU)      │
   └─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Layer-to-block mapping

This is the contract between layer 1 (op-graph scheduler) and layer 2 (accelerator
model). Every op category in `report.json` resolves to exactly one block:

| op category    | accelerator block       | dominant resource          | notes                            |
|----------------|-------------------------|-----------------------------|----------------------------------|
| `embedding`    | MAC array (gather)      | DRAM BW (vocab table)       | weight-tied with `lm_head`       |
| `norm`         | Range-Norm unit         | A-SRAM ports                | Eq. 2; no rsqrt                  |
| `in_proj`      | MAC array               | W-SRAM BW + DRAM (decode)   | tile along ED·2                  |
| `conv1d`       | Conv1D engine           | A-SRAM (conv-cache)         | depthwise, k=4                   |
| `x_proj`       | MAC array               | W-SRAM                      | produces Δ, B, C                 |
| `dt_proj`      | MAC array → ReLU        | W-SRAM                      | softplus → ReLU substitution     |
| `ssm_scan_etc` | SSM Scan + PWL-Exp      | STATE-SRAM (recurrent)      | bandwidth-bound, AI≈0.5          |
| `out_proj`     | MAC array               | W-SRAM + DRAM (decode)      | gate ⊙ y on input                |
| `lm_head`      | MAC array               | DRAM BW                     | shares weights with embedding    |

### 3.2 What runs where

- `workload/` — Python, CPU, run-once. Produces a per-op trace with measured
  shapes/FLOPs and analytically-derived SSM scan FLOPs. **Already done.**
- `approximations.py` — Python, the numeric source of truth for the four
  hardware approximations. **Already done.**
- `simulators/` — Rust, the eMamba accelerator model (§3 box 2). **To build.**
- `ramulator2/` — C++ today; the Rust port plan in `ramulator2_rust_design.md`
  becomes the memory backend. **To build (port).**
- `arch-hdl/` — RTL track, out of scope for the simulator but lives in the same
  repo so block-level latency/throughput parameters can be cross-checked against
  synthesis later. **Future.**

---

## 4. Open architectural questions

- **MAC array geometry.** The decode AI of 0.5 F/B argues for a small array kept
  busy by DRAM, not a large array starved by it. Pick the size from a roofline
  against LPDDR5 BW, not from peak FLOPs.
- **STATE-SRAM banking.** The SSM scan reads/writes h_t in place every token.
  One bank per layer (M=24) is simplest; one bank shared with double-buffering
  is smaller. Decide once we have scan-unit cycle counts.
- **Conv-cache vs. STATE-SRAM colocation.** `d_conv-1 = 3` columns per layer is
  ~9 KB total — small enough to live in STATE-SRAM and avoid a separate bank.
- **Weight-tying for embedding/lm_head.** If we model the cache between the two
  uses correctly, decode DRAM traffic drops by ~150 MB/token. Worth a
  flag in the simulator config.
- **Mixed precision.** Paper uses INT8 weights / INT16 activations. The numeric
  reference in `approximations.py` is fp32. The simulator should carry a
  precision-config knob and run the bit-check in fp32 only, with a separate
  quantization pass for tape-out numerics.
