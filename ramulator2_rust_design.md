# Ramulator 2 → Rust: Design Doc

Status: draft
Target reader: someone who already knows DRAM simulation at a high level and wants to understand how Ramulator 2 is built in C++ and what a Rust reimplementation should look like.

## 1. Goals & non-goals

**Goals.**
- Port Ramulator 2's *architecture* (cycle-driven, plugin-based, YAML-configured, hierarchical DRAM state machine) to safe Rust.
- Keep numerical results bit-for-bit equivalent on the canonical traces in `ramulator2/example_*.trace` and the configs in `example_config*.yaml`.
- Preserve the extension model: third parties should be able to add a new scheduler / refresh manager / DRAM standard / RowHammer mitigation without touching the core.
- Get a measurable performance win (target: 1.5–3× single-thread on `simpleO3 + DDR4` workloads), and make multi-channel parallelism *possible* (not required for v1).

**Non-goals.**
- Not a behavioral rewrite. We are not redesigning the simulator; we are translating it. Behavior changes happen in a follow-up.
- No gem5 integration in v1. Ramulator 2 ships a `gem5_frontend` wrapper; we'll keep a C ABI hook so gem5 can call into the Rust core later, but the wrapper itself is out of scope.
- No GUI / web UI / new tracing format.

**Success criteria.**
- `cargo run --release -- -c example_config.yaml -t example_inst.trace` produces the same per-stat YAML output as the C++ build, modulo float ordering.
- Microbenchmark: ≥1.5× faster end-to-end on `rh_study` workloads.
- A new scheduler can be added in <100 LOC across one file, registered via a derive macro.

## 2. What Ramulator 2 is, in one diagram

```
   ┌──────────────────────────┐         ┌───────────────────────────────────────────────┐
   │  IFrontEnd               │  send() │  IMemorySystem                                │
   │  - SimpleO3 / Trace /    │────────▶│  - GenericDRAMSystem                          │
   │    BHO3 / gem5 wrapper   │         │     ├── IAddrMapper        (addr → addr_vec) │
   │  - has ITranslation      │         │     ├── IDRAM              (state + timing)  │
   │  - tick() each cycle     │         │     └── Vec<IDRAMController> (per channel)   │
   └──────────────────────────┘         │             ├── IScheduler                   │
              ▲                          │             ├── IRefreshManager             │
              │ callback(req)            │             ├── IRowPolicy                  │
              │                          │             └── Vec<IControllerPlugin>      │
              └──────────────────────────┴───────────────────────────────────────────────┘
                  Driven by a single tick loop in main() that mod-schedules
                  frontend.tick() and memory_system.tick() based on clock ratios.
```

Key C++ files (anchors I'll refer back to):

- `src/main.cpp:81-112` — the entire top-level loop.
- `src/base/base.h:46-240` — `Implementation` base class: parent/children, params, stats, logger.
- `src/base/base.h:246-277` — `RAMULATOR_REGISTER_INTERFACE` / `_IMPLEMENTATION` macros.
- `src/base/factory.h:26-86` — global registry, `create_*` entry points.
- `src/base/clocked.h:16-28` — `Clocked<T>` CRTP base.
- `src/base/request.h:12-71` — `Request`, `ReqBuffer`.
- `src/memory_system/impl/generic_DRAM_system.cpp:25-90` — how a memory system wires itself up.
- `src/dram_controller/impl/generic_dram_controller.cpp` — read/write/priority buffers, scheduling, command issue.
- `src/dram/dram.h:15-100+`, `src/dram/node.h:33-100+`, `src/dram/impl/DDR4.cpp` — hierarchical DRAM state machine + a real spec.

## 3. Architecture review (what we're translating)

### 3.1 Plugin/factory system

Ramulator 2's elegance is concentrated here. Two macros drive everything:

```cpp
RAMULATOR_REGISTER_INTERFACE(IDRAMController, "Controller", "...")
RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, GenericDRAMController,
                                  "Generic", "A generic DRAM controller.")
```

At static-init time, each registered type writes itself into `Factory::m_registry`, a `map<interface_name, InterfaceInfo>`. At runtime, `Factory::create_implementation(ifce_name, impl_name, yaml_node, parent)` looks up the constructor and instantiates it with its YAML subtree. Every component:

1. inherits from `Implementation` (config, params, stats, parent ptr, children vec, spdlog logger),
2. inherits from one or more interface traits (`IDRAMController`, `Clocked<T>`, etc.),
3. overrides `init()` (post-config setup), `setup(IFrontEnd*, IMemorySystem*)` (cross-component wiring after the whole tree exists), and `finalize()` (stats dump).

Children are created with `create_child_ifce<T>()`, which reads `this->m_config[T::interface_name]["impl"]` to pick the implementation. So a single YAML tree drives the entire component tree — and the tree is recursive: a controller creates its scheduler the same way the system creates a controller.

### 3.2 Request flow

`Request` (`src/base/request.h:12-44`) carries `addr`, `addr_vec`, type, source, current/final command, arrive/depart cycles, a 4-int scratchpad, and a `std::function<void(Request&)>` callback. The flow:

```
frontend.tick()
   → memsys.send(req)
       → addr_mapper.apply(&req)        // fills req.addr_vec
       → controllers[req.addr_vec[0]].send(req)   // route by channel
           → enqueue in read/write/priority ReqBuffer
controllers[i].tick()                    // every memsys cycle
   → scheduler.get_best_request()
   → dram.command_legal(req.command)?
   → dram.update(req.command, req.addr_vec)   // state + timing advance
   → if final command: req.callback(req)      // back to frontend
```

### 3.3 Cycle-driven simulation

`Clocked<T>` is just `m_clk` + a virtual `tick()`. The driver in `main.cpp` handles clock-ratio mismatch (e.g., memory at 4× core) by mod-scheduling:

```cpp
int frontend_tick = frontend->get_clock_ratio();   // e.g., 4
int mem_tick      = memory_system->get_clock_ratio(); // e.g., 1
int tick_mult     = frontend_tick * mem_tick;
for (uint64_t i = 0; ; i++) {
  if (((i % tick_mult) % mem_tick) == 0)        frontend->tick();
  if (frontend->is_finished())                   break;
  if ((i % tick_mult) % frontend_tick == 0)      memory_system->tick();
}
```

The whole simulator is one thread, one loop, one integer counter. No event queue. Easy to translate, easy to optimize, easy to make deterministic.

### 3.4 DRAM state machine

`IDRAM` (`src/dram/dram.h`) owns:

- a tree of `DRAMNode`s (channel → rank → bankgroup → bank → row), each with a current `m_state`, a `m_cmd_ready_clk[cmd]` vector ("when can this command next fire here"), and a circular `m_cmd_history[cmd]`.
- a compile-time-ish set of *spec definitions* (`SpecDef`, `ImplDef<N>`) — name↔ID maps for commands and states.
- `TimingConsInitializer` arrays: declarative `(prev_cmd, next_cmd, latency, window, sibling)` rows that get compiled into a 3D LUT `m_timing_cons[level][cmd][prev_cmd]`.
- per-command metadata flags (`is_opening`, `is_closing`, `is_accessing`, `is_refreshing`).
- optional power model.

`update(cmd, addr_vec)` walks the relevant nodes, applies the state transition lambdas (`src/dram/lambdas/`), and pushes the timing constraints forward. Each spec (`DDR4`, `DDR5`, `HBM3`, `LPDDR5`, `GDDR6`) is a separate `IDRAM` impl that fills these tables in `init()`.

## 4. Why Rust, and what changes

The C++ design leans on three things Rust does *differently*, not worse:

| C++ idiom                                  | Rust replacement                                                |
|--------------------------------------------|------------------------------------------------------------------|
| Static-init macros writing to a global map | `inventory` or `linkme` crate; or a `submit!{}` block per impl  |
| Virtual inheritance (`Implementation` + interface) | Trait + struct: components implement multiple traits, no diamond |
| `std::function` callbacks                  | `Box<dyn FnMut(&mut Request)>` (or an enum if hot-path)         |
| YAML node passed everywhere, parsed lazily | `serde_yaml::Value` at the boundary, typed structs inside       |
| Raw `IDRAM*` parent/child pointers         | `&dyn IDram` borrows; ownership lives in the system tree        |
| `std::list<Request>`                       | `VecDeque<Request>`                                              |
| `spdlog`                                   | `tracing` + `tracing-subscriber`                                |
| `argparse` + `yaml-cpp`                    | `clap` + `serde_yaml`                                           |

The two non-trivial design decisions for the port:

**(a) Component ownership.** C++ makes every component a heap-allocated `Implementation*` with a parent pointer and a children vector. Rust can do better: the system tree is a *tree* — model it as a tree. The memory system *owns* its DRAM, address mapper, and controllers (`Box<dyn …>`). Children don't need to point back at parents because the call graph is always parent → child during `tick()`. The two places that need cross-references are:
  - `IFrontEnd ↔ IMemorySystem` (frontend calls `send`, memsys fires callbacks back). Solve with a `Box<dyn FnMut(Request)>` send handle on the frontend, set during wiring.
  - `setup()` cross-wiring (e.g., a controller plugin needs to peek at the DRAM model). Pass a `&SystemContext { dram, frontend, … }` reference into `setup()`. No back-pointers stored.

**(b) Plugin registration.** We can't (and don't want to) reproduce static-init macros. Two options:

- **`inventory`**: each impl writes one `inventory::submit!` block. The runtime registry is built lazily on first lookup. Pro: matches C++ ergonomics 1:1. Con: a bit of magic, breaks under `--cfg miri`, and on some targets requires linker tricks.
- **Explicit registry**: each impl is a function `fn register(reg: &mut Registry)` and a top-level `register_all()` lists them. Pro: dead obvious, no dependencies, plays well with `cargo check`. Con: adding a new impl means touching `register_all()`.

**Recommendation: explicit registry**, with a `#[derive(RamulatorImpl)]` proc-macro that generates the boilerplate (`fn register`, name/desc constants, the `from_yaml(node, parent_ctx) -> Box<dyn Trait>` constructor). One central `register_all()` is fine; it's the same maintenance cost as Ramulator's CMake `target_sources` lists, and it makes the dependency graph greppable. We can switch to `inventory` later if the registry list grows past ~50 lines.

## 5. Crate layout

```
ramulator-rs/
├── Cargo.toml                # workspace
├── crates/
│   ├── core/                 # interfaces, Request, Clocked, Registry, Config
│   │   src/lib.rs
│   │   src/request.rs
│   │   src/clock.rs
│   │   src/registry.rs
│   │   src/config.rs
│   │   src/stats.rs
│   ├── dram/                 # IDram trait + DDR4/DDR5/HBM3/LPDDR5 impls
│   │   src/lib.rs
│   │   src/node.rs           # generic hierarchical state machine
│   │   src/spec.rs           # SpecDef, ImplDef equivalents
│   │   src/timing.rs         # TimingCons compilation
│   │   src/ddr4.rs … src/lpddr5.rs
│   ├── controller/           # IDramController + Generic/PRAC/BH
│   │   src/lib.rs
│   │   src/scheduler.rs      # FCFS, FRFCFS, …
│   │   src/refresh.rs
│   │   src/rowpolicy.rs
│   │   src/plugin.rs         # IControllerPlugin trait
│   ├── addr_mapper/
│   ├── translation/
│   ├── memory_system/        # IMemorySystem + GenericDRAMSystem + BHDRAMSystem
│   ├── frontend/             # ITranslation users, SimpleO3, BHO3, traces
│   ├── derive/               # proc-macro #[derive(RamulatorImpl)]
│   └── stats-derive/         # proc-macro #[derive(Stats)] (optional)
├── ramulator/                # binary crate: main.rs, register_all.rs
└── examples/
    ├── add-scheduler.md
    └── add-dram-spec.md
```

Why a workspace: each module compiles independently, and a downstream user can depend on `ramulator-core` + their own scheduler crate without pulling in every DRAM standard.

## 6. Core traits

The translation is mostly mechanical. Sketch:

```rust
// crates/core/src/lib.rs
pub type Addr  = u64;
pub type Clock = u64;
pub type AddrVec = SmallVec<[i32; 8]>;   // ch, rank, bg, ba, row, col, [extras]

pub trait Tick {
    fn tick(&mut self);
    fn clock_ratio(&self) -> u32 { 1 }
}

pub trait Component: Tick {
    fn name(&self) -> &str;
    fn finalize(&mut self, sink: &mut dyn StatSink) {}
}

pub trait Frontend: Component {
    fn connect(&mut self, send: SendHandle);
    fn is_finished(&self) -> bool;
    fn translation(&self) -> Option<&dyn Translation> { None }
}

pub trait MemorySystem: Component {
    fn send(&mut self, req: Request) -> Result<(), Request>;  // err returns the rejected req
    fn num_channels(&self) -> usize;
}

pub trait DramController: Tick {
    fn enqueue(&mut self, req: Request) -> Result<(), Request>;
    fn priority_enqueue(&mut self, req: Request) -> Result<(), Request>;
}

pub trait Dram {
    fn command_legal(&self, cmd: Cmd, addr: &AddrVec, now: Clock) -> bool;
    fn update(&mut self, cmd: Cmd, addr: &AddrVec, now: Clock);
    fn next_command(&self, req: &Request, now: Clock) -> Cmd;
    fn organization(&self) -> &Organization;
    fn timing(&self, level: Level, cmd: Cmd) -> &[TimingEntry];
}

pub trait AddrMapper {
    fn apply(&self, req: &mut Request);
}

pub trait Scheduler {
    fn pick<'a>(&mut self, queue: &'a mut ReqBuffer, dram: &dyn Dram, now: Clock) -> Option<&'a mut Request>;
}

pub trait RefreshManager: Tick { /* … */ }
pub trait RowPolicy { fn on_access(&mut self, … ); fn on_idle(&mut self, …); }
pub trait ControllerPlugin: Tick {
    fn on_request(&mut self, req: &mut Request, ctx: &mut PluginCtx) {}
    fn on_command(&mut self, cmd: Cmd, addr: &AddrVec, ctx: &mut PluginCtx) {}
}
```

`Request` mirrors the C++ struct. Two callouts:

- The callback. C++ uses `std::function<void(Request&)>`. In Rust, the request is owned by whoever has it at the moment, so the callback should be `Option<Box<dyn FnOnce(Request) + Send>>`. For the hot path on processor frontends, swap to a typed enum (`Callback::CoreId(u32)`) so we don't allocate one boxed closure per memory request.
- `scratchpad`. Keep `[i32; 4]`. Cheap, fixed, fine.

## 7. Cycle loop

The C++ loop in `main.cpp:100-112` ports almost verbatim:

```rust
fn run(mut frontend: Box<dyn Frontend>, mut memsys: Box<dyn MemorySystem>) {
    let f_ratio = frontend.clock_ratio();
    let m_ratio = memsys.clock_ratio();
    let lcm     = (f_ratio as u64 * m_ratio as u64) / gcd(f_ratio, m_ratio) as u64;
    let f_step  = lcm / f_ratio as u64;
    let m_step  = lcm / m_ratio as u64;

    let mut i: u64 = 0;
    loop {
        if i % f_step == 0 { frontend.tick(); }
        if frontend.is_finished() { break; }
        if i % m_step == 0 { memsys.tick(); }
        i = i.wrapping_add(1);
    }
}
```

Two improvements over the C++ version while we're here:

1. Use `lcm/step` instead of `tick_mult` + nested mod. Same behavior, fewer divisions.
2. Hoist `is_finished()` to a cheap atomic flag the frontend flips, so the inner loop doesn't do a v-table call every cycle.

## 8. Configuration & registry

YAML structure stays identical to the C++ version — that's a hard requirement for the bit-exact comparison test.

```yaml
Frontend:
  impl: SimpleO3
  num_expected_insts: 1000000
  ...
MemorySystem:
  impl: GenericDRAM
  DRAM:
    impl: DDR4
    org: { preset: DDR4_8Gb_x8 }
    timing: { preset: DDR4_2400R }
  Controller:
    impl: Generic
    Scheduler: { impl: FRFCFS }
    RefreshManager: { impl: AllBank }
    RowPolicy: { impl: OpenedRow }
    plugins:
      - { ControllerPlugin: { impl: PARA, threshold: 4096 } }
```

Registry shape:

```rust
// crates/core/src/registry.rs
pub struct Registry {
    by_iface: HashMap<&'static str, IfaceEntry>,
}
pub struct IfaceEntry {
    pub iface_name: &'static str,
    pub impls: HashMap<&'static str, ImplCtor>,
}
pub type ImplCtor = fn(node: &serde_yaml::Value, ctx: &BuildCtx) -> Box<dyn Any>;

// the proc-macro generates this:
impl GenericDramController {
    pub const IFACE: &'static str = "Controller";
    pub const NAME:  &'static str = "Generic";
    pub fn register(reg: &mut Registry) { reg.add::<dyn DramController, Self>(...); }
}
```

`BuildCtx` carries the things a constructor needs from "above": the parent's clock ratio, a logger, a stats sink, and a thunk that builds child interfaces (`ctx.build_child::<dyn Scheduler>(node)`). This replaces `Implementation::create_child_ifce<T>()`.

## 9. DRAM state machine in Rust

The hierarchical node tree is the part where Rust types should help, not get in the way.

```rust
// crates/dram/src/node.rs
pub struct DramNode {
    pub level: Level,
    pub state: NodeState,
    pub cmd_ready: Vec<Clock>,        // indexed by Cmd
    pub cmd_history: Vec<RingBuf<Clock>>,
    pub children: Vec<DramNode>,
    pub row_state: FxHashMap<u32, RowState>,  // only at bank level
}
```

We avoid the C++ template dance (`DRAMNodeBase<T>`) entirely. The level enum and `cmd`/`state` indices are `u8`s; the LUTs are `Vec<Vec<Vec<TimingEntry>>>` flat-indexed by `(level, cmd, prev_cmd)`. `TimingEntry` stays a 16-byte POD.

The per-spec `init()` work — filling timing tables, setting organization presets — translates to a normal `impl Dram for Ddr4 { fn from_config(cfg: &Ddr4Cfg) -> Self { … } }`. Each spec gets a typed config struct (with `serde::Deserialize`) instead of pulling values out of a YAML node by string key. That's both faster and catches typos at parse time.

## 10. Stats

C++ collects stats via `register_stat(var).name(...)` and prints YAML at finalize. Two paths in Rust:

- **Light**: a `StatSink` trait with `record(path: &[&str], value: StatValue)`; each component writes into it during `finalize()`.
- **Derive**: `#[derive(Stats)]` on a per-component stats struct emits a `report(&self, sink, prefix)` impl.

Start with the light version. The derive is a 1-day project we can add when we have 10+ components and the boilerplate hurts.

## 11. Concurrency

The C++ simulator is single-threaded. Per-channel work *could* be parallel — controllers don't share state with each other, only with the DRAM model and the address mapper, both of which are read-mostly inside a tick. The cleanest way to expose this in Rust:

- Split `IDram` into `&self` (queries: `command_legal`, `next_command`) and `&mut self` (`update`).
- Per-channel controllers each own their own slice of DRAM state (a sub-tree starting at one channel) — this is true today, it's just not visible in the C++ types.
- Drive the per-tick loop with rayon: `controllers.par_iter_mut().for_each(|c| c.tick())`.

This is a v1.5 feature, not v1. The point is: lay out the types so that future parallelism is a `.par_iter_mut()` swap, not a refactor. Concretely, that means **DRAM state must be split per channel from day one** — don't have a single shared `Dram` object behind `Rc<RefCell<>>`. Instead, the memory system holds `Vec<ChannelDram>`, and the global `IDram` is a thin facade that dispatches to them.

## 12. Migration plan (phasing)

Each phase ends with a green test and a runnable binary.

**Phase 0 — scaffolding (≈3 days).**
Workspace, `core` crate with `Request`, `Tick`, `Registry`, `Config`. Trivial `NullFrontend` + `NullMemorySystem` that wire up and tick. CLI parses YAML, builds component tree, runs loop.

**Phase 1 — DDR4 happy path (≈2 weeks).**
`Dram` trait + DDR4 impl. `AddrMapper` (the basic one). `Scheduler::FCFS`. `Generic` controller without refresh, without row policy plugins. `ReadWriteTrace` frontend. Goal: byte-identical stats with `example_config.yaml` + `example_inst.trace`.

**Phase 2 — fill in the matrix (≈3 weeks).**
- Schedulers: FRFCFS, FRFCFS-Cap, BlockingFCFS.
- Refresh: AllBank, PerBank.
- Row policies: open, close, opened-row.
- Address mappers: RIT, others.
- `SimpleO3` frontend + `Translation`.
- DDR5, LPDDR5. (HBM and GDDR6 in phase 4.)

**Phase 3 — RowHammer track (≈2 weeks).**
`ControllerPlugin` trait. Port PARA, Graphene, BlockHammer, TWiCe, Hydra, RRS, AQUA, OracleRefresh. `BHO3` frontend. `BHDRAMController`, `BHDRAMSystem`. `PRACDRAMController`. This is the largest LOC chunk; doing it after phase 2 means the trait surface is stable.

**Phase 4 — completeness (≈2 weeks).**
Remaining DRAM standards (HBM, HBM3, GDDR6). Power model. Stats derive macro. C ABI shim for gem5 integration (header + cdylib).

**Phase 5 — performance (≈1 week).**
Profile. Replace the boxed-closure callback. Add `par_iter_mut` per-channel. Inline the timing-LUT lookup. Target: 1.5–3× C++.

End-to-end timeline: ~10 weeks for a single engineer. Phases 1–3 are the load-bearing ones; 4 and 5 can land independently.

## 13. Validation strategy

Three layers, in order of cost:

1. **Snapshot tests on stat YAML.** Every config in `ramulator2/example_config*.yaml` paired with the matching trace produces a known-good stats YAML; the Rust binary must produce the same. Diff via a YAML-aware comparator that's tolerant to numeric formatting but not to values.
2. **Property tests on the DRAM state machine.** `proptest`-generated command sequences against a reference Python model (existing approximations.py shows the user is comfortable with Python references). Goal: catch off-by-one in timing constraints early, where they're cheap to fix.
3. **Verilog comparison.** Ramulator 2 ships `verilog_verification/`. Replicate the same harness: feed identical command streams to the Rust simulator and the Verilog model, compare cycle-by-cycle command issue.

Layer 1 is mandatory before each phase merge. Layers 2 and 3 are good investments for phase 1 and phase 4 respectively.

## 14. Open questions

- **Float reproducibility.** Stats include power numbers. Different libm = different bits. Either pin to a vendored `libm` (the `libm` crate) or relax the snapshot test to ULP-tolerance for power stats only. Lean toward the latter.
- **Logger.** `tracing` is the obvious pick, but Ramulator 2's spdlog output format is what users grep. Decide whether to match the format exactly (extra adapter) or break compatibility and document the migration. Lean toward breaking it.
- **`ITranslation`.** The C++ interface is small but oddly stateful (reservations, max-address). Worth checking whether any current user actually calls `reserve()` outside of test code; if not, simplify the trait.
- **Inventory vs. explicit registry.** Already discussed above; revisit at phase 3 when plugin count is highest.
- **Plugin ordering.** Controller plugins run in YAML-listed order in C++. Rust should preserve that. Document it as a contract; don't let `HashMap` iteration order leak in.

## 15. References to the existing tree

| Concept                | C++ file                                                         | Rust crate / file                          |
|------------------------|------------------------------------------------------------------|--------------------------------------------|
| Top-level loop         | `src/main.cpp:81-112`                                            | `ramulator/src/main.rs`                    |
| Implementation base    | `src/base/base.h:46-240`                                         | (gone — replaced by traits + `BuildCtx`)   |
| Macros                 | `src/base/base.h:246-277`                                        | `crates/derive` proc-macro                 |
| Factory                | `src/base/factory.h:26-86`                                       | `crates/core/src/registry.rs`              |
| Clocked                | `src/base/clocked.h:16-28`                                       | `crates/core/src/clock.rs` (`Tick` trait)  |
| Request / ReqBuffer    | `src/base/request.h:12-71`                                       | `crates/core/src/request.rs`               |
| Memory system          | `src/memory_system/impl/generic_DRAM_system.cpp`                 | `crates/memory_system/src/generic.rs`      |
| Generic controller     | `src/dram_controller/impl/generic_dram_controller.cpp`           | `crates/controller/src/generic.rs`         |
| DRAM state machine     | `src/dram/dram.h`, `src/dram/node.h`, `src/dram/lambdas/`        | `crates/dram/src/node.rs`, `…/timing.rs`   |
| DDR4 spec              | `src/dram/impl/DDR4.cpp`                                         | `crates/dram/src/ddr4.rs`                  |
| RowHammer plugins      | `src/dram_controller/impl/plugin/*`                              | `crates/controller/src/plugins/*`          |
| BH frontend            | `src/frontend/impl/processor/bhO3/*`                             | `crates/frontend/src/bho3.rs`              |
