# Plan: implement eMamba in arch-hdl

## Context

eMamba is the project's hardware accelerator for Mamba SSM decode. Three
implementation tiers already exist or are partially built; arch-hdl is the
third tier (RTL) and is the goal of this plan.

```
   isa_simulator/        →   simulators/             →   arch-hdl/  ← this plan
   functional ISA            cycle-accurate Rust         RTL in arch-hdl-lang
   (bit-exact FP32           (per-block latency           (synthesizable SV)
    golden reference)         models, mem system)
```

The RTL must be **functionally equivalent** to `isa_simulator` (bit-identical
FP32 outputs against `approximations.py` golden vectors and the eight ISA ops)
and **structurally consistent** with `simulators/` (same nine compute units,
same memory hierarchy, same Op/dispatch model — so the analytical cycle model
in `crates/compute/` becomes the cycle target for Verilator runs).

We use `arch-hdl/arch-com/` (already cloned) as the HDL/compiler. arch is a
purpose-built RTL language that emits SystemVerilog; it provides first-class
`fsm`, `pipeline`, `fifo`, `ram`, `arbiter`, and `thread` constructs that map
directly onto the blocks we need (controller FSM, MAC pipeline, op-stream
FIFO, SRAM banks, DMA arbitration, multi-cycle scan protocol).

Out of scope: changes to `isa_simulator/` or `simulators/` (used only as
reference); INT8/INT24 paths (reserved per ISA §0.3); branches (no
control flow in v0.1 per ISA §0.2).

---

## Approach

**Phased vertical slices.** Each phase produces a synthesizable, simulable
top-level that runs a real op stream end-to-end, and is checkable against
both upstream tiers. We do *not* build all leaf blocks first and integrate
last — we integrate from phase 1 with stubs and replace stubs as we go. This
keeps the dispatch/controller path under continuous test and surfaces
toolchain or address-map bugs early.

**Layout.** New code lives in a sibling directory to arch-com:

```
arch-hdl/
├── arch-com/          # the HDL compiler (vendored, do not edit)
└── emamba/            # NEW — the eMamba RTL
    ├── README.md
    ├── Makefile       # invokes ../arch-com/target/release/arch
    ├── src/
    │   ├── top.arch              # emamba_top: controller + blocks + SRAMs
    │   ├── ctrl/
    │   │   ├── op_fetch.arch     # pc, 128-bit op-stream BRAM read
    │   │   ├── decode.arch       # opcode demux, field extraction, malformed-op check
    │   │   ├── dispatch.arch     # one-op-at-a-time issue (ISA §0.2 serial)
    │   │   ├── addr_map.arch     # §0.5 bus-map decoder (W/A/STATE/DRAM/SimCtrl)
    │   │   ├── exceptions.arch   # IllegalOpcode/Dtype/Misaligned/BusFault/...
    │   │   └── simctrl.arch      # 0x8000_0000 halt device
    │   ├── mem/
    │   │   ├── w_sram.arch       # 4 MB ping-pong (`ram` construct)
    │   │   ├── a_sram.arch       # 1 MB activations
    │   │   ├── state_sram.arch   # 2 MB h_t + conv-cache (in-place RW)
    │   │   └── dma.arch          # DRAM↔SRAM byte copy (AXI-MM master)
    │   ├── compute/
    │   │   ├── mac_array.arch    # systolic FP32 MAC, mac_tile
    │   │   ├── conv1d.arch       # depthwise k=4 + cache-shift, conv_step
    │   │   ├── range_norm.arch   # min/max/mean trees, norm
    │   │   ├── ssm_scan.arch     # fused recurrence, ssm_step
    │   │   ├── pwl_silu.arch     # 17-seg PWL, silu
    │   │   ├── pwl_exp.arch      # 11-seg PWL, exp (also embedded in ssm_scan)
    │   │   └── relu.arch         # softplus→ReLU substitution (used by ssm_scan)
    │   └── pkg/
    │       ├── isa.arch          # 128-bit op struct, opcode constants, format unpackers
    │       └── dtypes.arch       # FP32 types, ε constants, breakpoint/slope LUTs
    ├── tests/
    │   ├── golden/               # .npy fixtures from approximations.py + isa_simulator
    │   ├── streams/              # .bin op streams produced by isa_simulator
    │   ├── tb/                   # Verilator C++ testbenches (one per block + top)
    │   └── snapshots/            # arch compiler regression snapshots
    └── tools/
        └── gen_golden.py         # pulls bit-exact fixtures from upstream tiers
```

The top-level integrates everything from phase 1; later phases swap stubs for
real blocks.

---

## Phases

Each phase ends with: (a) `arch check` + `arch build` clean, (b) `arch sim`
golden-vector test passing, (c) cycle count cross-checked against
`simulators/crates/compute/src/<unit>.rs` formulas (within a tolerance noted
in `tests/tb/<unit>_tb.cpp`).

### Phase 0 — Scaffolding & toolchain wiring (1–2 days)

- Create the directory layout above.
- `Makefile`: targets `check`, `build`, `sim-<unit>`, `sim-top`, `golden`.
  Each target shells to `arch-com/target/release/arch`.
- `tools/gen_golden.py`: imports `approximations.py`, links to
  `isa_simulator`'s op executor, dumps `.npy` per-op fixtures and `.bin` op
  streams to `tests/golden/` and `tests/streams/`. Run once per CI.
- `pkg/isa.arch`: define the 128-bit op record and per-format unpack
  functions (M/N/C/S/A/D from ISA §0.7) — pure combinational, no state.
- `pkg/dtypes.arch`: declare the FP32 alias, `EPS = 1e-5_f32` constant,
  PWL breakpoint/slope/intercept ROM init values (computed in
  `gen_golden.py` from `approximations.py` so they stay in sync).
- Smoke test: `arch check src/pkg/*.arch` succeeds.

### Phase 1 — Vertical slice: DMA + halt (3–4 days)

The smallest end-to-end stream that exercises fetch/decode/dispatch/halt.

- `mem/w_sram.arch`, `mem/a_sram.arch`, `mem/state_sram.arch`: BRAM-backed
  RAMs via arch's `ram` construct (see `arch-com/doc/ARCH_HDL_Specification.md`
  § ram). Latency-1 reads, single-write port, sized per ISA §0.5.
- `mem/dma.arch`: AXI-MM-style master. `thread`-based protocol (per
  `doc/thread_spec_section.md`): wait for issue, walk byte counter, drive
  DRAM master and SRAM write port, signal done. Pattern reference:
  `arch-com/examples/dma_engine.arch`.
- `ctrl/op_fetch.arch`: pc register + ROM-style op-stream BRAM (loaded by
  testbench).
- `ctrl/decode.arch`: opcode demux for opcodes `0x07`/`0x08` only (others →
  `IllegalOpcode` for now); field extraction.
- `ctrl/addr_map.arch`: bus decode per ISA §0.5; emits region select.
- `ctrl/simctrl.arch`: write to `0x8000_0000` raises `halt_pulse`.
- `ctrl/dispatch.arch`: serial issue, blocks until DMA done, advances pc.
- `top.arch`: instantiate the above; expose AXI master, op-stream load port,
  halt observable.
- **Validation**: `tests/streams/dma_smoke.bin` is one `dma_load` then a
  halt; testbench loads it, runs Verilator, checks W-SRAM contents match
  the .npy fixture and halt fires at the right pc.

### Phase 2 — PWL leaf blocks (2–3 days)

These are the simplest compute blocks and validate the FP32 numeric path.

- `compute/pwl_silu.arch`: 17-segment ROM (slopes + intercepts) + below/above
  saturation. One element/cycle steady state. Implements `silu` (Format A).
- `compute/pwl_exp.arch`: 11-segment ROM, below=0, above=e ≈ 2.7182817_f32.
  Implements `exp` (Format A). The same module is instantiated inside
  `ssm_scan.arch` later.
- Decode: extend `ctrl/decode.arch` to opcodes `0x05`, `0x06`.
- **Bit-exactness**: ROM contents and the LUT-index search (`partition_point`
  per ISA §0.9.3) must produce f32 outputs identical to
  `approximations.py:silu_pw / exp_pw`. Compare via `tools/gen_golden.py`
  output (`.npy` fixtures of (input, expected_output) pairs covering edge
  cases: domain endpoints, below/above ranges, sub/sup-normals, ±0).

### Phase 3 — Range-Norm + ReLU (2–3 days)

- `compute/relu.arch`: combinational `max(x, 0_f32)` per ISA §0.9.1
  (NaN→0). Used internally by `ssm_scan` later.
- `compute/range_norm.arch`: pipeline with three reduction trees (sum,
  min, max) over `len` elements, then mean = sum/len, range = max−min,
  output = γ·(x−μ)/(range+ε)+β. ε = `1e-5_f32`. Implements `norm`
  (Format N). Cycle target: matches
  `simulators/crates/compute/src/range_norm.rs` formula.
- Decode: extend to opcode `0x02`.
- **Validation**: golden vectors from `approximations.py:range_normalization`
  for representative `len` values used by mamba-130m (D=768).

### Phase 4 — MAC tile (4–6 days)

- `compute/mac_array.arch`: systolic FP32 array, parametric `MAC_ROWS` ×
  `MAC_COLS`. Sized for bandwidth-match per architecture.md §1.1
  (decode AI ≈ 0.5 F/B). Streaming operand feed (AXI-Stream-bundle from
  W-SRAM read port) + accumulator drain. Use arch's `pipeline` construct
  for the inner reduction; `arbiter` if multiple consumers contend on
  W-SRAM read ports.
- Implements `mac_tile` (Format M, opcode `0x01`). Tile sizes vary up to
  m,n,k ≤ 4096; large tiles are walked by an inner FSM in `dispatch`.
- Accumulation order is **canonical row-major / left-to-right** per ISA §0.8
  (no FMA reordering for v0.1) — keep it scalar-equivalent until phase 8.
- **Validation**: golden tiles from `isa_simulator`'s `MacTileOp::execute`.

### Phase 5 — Conv1D (2 days)

- `compute/conv1d.arch`: depthwise k=4, channel-wise dot product, in-place
  cache shift in STATE-SRAM (cache layout per ISA §0.7 Format C). One
  cycle/channel steady state on a vector of length ED.
- Implements `conv_step` (opcode `0x03`).
- **Validation**: golden against `isa_simulator`'s `ConvStepOp` — verify the
  cache-shift side effect bit-exactly (newest tap in slot 0, oldest evicted).

### Phase 6 — SSM scan (load-bearing block, 6–8 days)

This is the largest block; architecture.md §2 is the spec.

- `compute/ssm_scan.arch`:
  - Reads packed `ctx_in` (x_t, Δ, B, C, A_diag, D_skip) from A-SRAM and
    `state_h` from STATE-SRAM.
  - Inner loop over `(i ∈ ED, j ∈ N)`:
    `Δ' = relu(Δ[i])`,
    `Δ̄ = pwl_exp(Δ' · A_diag[i,j])` (instantiates `pwl_exp.arch`),
    `B̄ = Δ' · B[i,j] · x_t[i]`,
    `h[i,j] = Δ̄ · h[i,j] + B̄` (in-place STATE-SRAM RW).
  - Output loop: `y[i] = D_skip[i]·x_t[i] + Σ_j C[i,j]·h[i,j]`, written to
    `ctx_out`.
  - Use `pipeline` for the inner ED·N MAC chain; STATE-SRAM read-modify-write
    needs two ports (or one port with bypass FIFO).
- Implements `ssm_step` (opcode `0x04`, Format S).
- **Validation**: golden h_t before/after and y_t from `isa_simulator`'s
  `SsmStepOp`. ED=1536, N=16 reference run.
- **Cycle target**: ssm_state + ssm_output formulas in
  `simulators/crates/compute/src/ssm.rs`.

### Phase 7 — Full integration & exception path (3 days)

- Wire all 8 opcodes through `decode.arch` and `dispatch.arch`.
- Implement the full exception aggregator (`ctrl/exceptions.arch`) for all
  six exception types from ISA §0.10. Per-op address-alignment check
  (`Misaligned`), bounds check (`OutOfBounds`), reserved-bit-zero check
  (`MalformedOp`).
- Run an end-to-end mamba-130m single-decode-token op stream produced by
  `isa_simulator` (one full layer or full M=24-layer stack) and verify final
  STATE-SRAM and `ctx_out` against the functional simulator.
- Snapshot the arch compiler output of `top.arch` in `tests/snapshots/` so
  future edits are diffable.

### Phase 8 — Cycle-count calibration (2 days)

- Run Verilator with `--trace` on each block testbench, count cycles for
  fixed inputs, compare against
  `simulators/crates/compute/src/<unit>.rs::cycles_for(...)`.
- Tune block parameters (RAM port count, MAC pipeline depth, scan unroll
  factor) until each block lands within ±5 % of the analytical model.
- Document deltas (where RTL differs from the model) in
  `emamba/README.md` so the cycle model in `simulators/` can be updated to
  match if needed.

---

## Critical files to read before starting

| Purpose                              | Path                                                                 |
| ------------------------------------ | -------------------------------------------------------------------- |
| eMamba microarchitecture             | `mamba/architecture.md`                                              |
| ISA contract (opcodes, formats, sem) | `mamba/isa_simulator/isa_spec.md`                                    |
| FP32 numeric ground truth            | `mamba/approximations.py`                                            |
| Cycle model targets                  | `mamba/simulators/crates/compute/src/{conv1d,linear,range_norm,pwl,ssm}.rs` |
| Op type & registry                   | `mamba/simulators/crates/core/src/{op,registry,tensor,dtype}.rs`     |
| Trace format (op streams)            | `mamba/simulators/crates/trace/src/parser.rs`                        |
| arch language spec                   | `arch-hdl/arch-com/doc/ARCH_HDL_Specification.md`                    |
| arch AI cheat-sheet                  | `arch-hdl/arch-com/doc/Arch_AI_Reference_Card.md`                    |
| `thread` construct (DMA, scan)       | `arch-hdl/arch-com/doc/thread_spec_section.md`                       |
| Reference DMA design (closest pattern) | `arch-hdl/arch-com/examples/dma_engine.arch`                       |
| Reference pipeline design            | `arch-hdl/arch-com/examples/cpu_pipeline.arch`                       |
| Compiler implementation status       | `arch-hdl/arch-com/doc/COMPILER_STATUS.md`                           |

---

## Reused primitives (no need to reinvent)

- `ram` construct (BRAM, configurable latency, init from file) → all SRAM banks
- `fifo` construct (sync + async-CDC variants) → op-stream queue, DMA cmd queue
- `pipeline` construct (auto-generated valid/stall) → MAC inner loop, scan inner loop, range-norm
- `arbiter` construct (round-robin/priority) → W-SRAM read-port arbitration between MAC tile-walker and DMA
- `thread` construct (multi-cycle straight-line code) → DMA byte loop, dispatch's "wait for unit done" protocol
- `fsm` construct → top-level controller, per-unit micro-FSMs
- arch's auto-CDC for any clock-domain crossings (DRAM clock vs core clock if separated)

ISA helpers (`pkg/isa.arch`) are pure combinational format unpackers; reuse
them everywhere instead of hand-bit-slicing.

---

## Verification — end to end

Three layers, all driven from `tools/gen_golden.py`:

1. **Per-op bit-exact** (Verilator C++ tb): drive a single op, compare
   block outputs to `.npy` from `approximations.py` (PWL, range-norm) or
   `isa_simulator` (mac, conv, ssm, dma). FP32 bit comparison, not
   tolerance.
2. **Op-stream end-to-end** (Verilator on `top.arch`): load a `.bin` op
   stream, run to halt, dump SRAM regions, diff vs. `isa_simulator`'s post-
   run state. Reference stream: one mamba-130m decode token, all 24 layers.
3. **Cycle parity** (Verilator `--trace`): per-block cycle counts within
   ±5 % of `simulators/crates/compute/` formulas; per-token cycle count
   within ±10 % of the integrated cycle simulator (when `simulators/` is
   ready to drive the same op stream).

Reproduction commands (after Phase 0):

```
make -C arch-hdl/emamba golden        # pull fixtures from upstream tiers
make -C arch-hdl/emamba check         # arch check on every .arch file
make -C arch-hdl/emamba sim-pwl-silu  # one block test
make -C arch-hdl/emamba sim-top       # end-to-end op stream
make -C arch-hdl/emamba cycles        # cycle-parity report vs simulators/
```

---

## Risks & open questions

- **arch language coverage.** FP32 multiplier/adder primitives may not be
  in arch's stdlib; check `arch-com/stdlib/` and `COMPILER_STATUS.md`. If
  missing, instantiate hand-written FP32 modules (or wrap a SystemVerilog
  blackbox via arch's foreign-module syntax). **Action**: read
  `COMPILER_STATUS.md` in Phase 0.
- **Bit-exact PWL.** ISA §0.9.3 mandates a specific search rule
  (`partition_point(side='right') - 1`). Implementing this in hardware as a
  binary search vs. a linear comparator chain has the same numeric output
  but different cycle cost. Pick linear comparators for correctness first;
  optimize in Phase 8.
- **STATE-SRAM in-place RW for SSM.** Single-port BRAM with read-modify-
  write needs careful pipeline scheduling; dual-port is simpler but
  doubles area. **Decision deferred to Phase 6 entry**, after the cycle
  model is concrete.
- **MAC array geometry.** Architecture.md §4 leaves this open. Pick
  a small starting point (e.g., 8×8 FP32 PEs) for Phase 4; revisit in
  Phase 8 once the roofline against LPDDR5 is concrete.
