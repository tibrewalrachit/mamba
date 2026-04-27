//! Workload-scale integration test.
//!
//! Generates a multi-layer × multi-token Mamba block trace programmatically,
//! runs the simulator, and cross-checks total_cycles against a CPU-derived
//! reference envelope computed from the same per-op formulas.
//!
//! "CPU reference" here means: apply the hardware cycle-count formula to every
//! op in the trace and compute the expected critical-path bound. This is the
//! same correctness model the paper uses — the simulator must reproduce the
//! formula, not exceed it by more than a small dispatch overhead.

use emamba_compute::linear::LinearProjUnit;
use emamba_compute::range_norm::RangeNormUnit;
use emamba_compute::units::MemmoveStage;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p
}

/// Build a trace string for `layers` × `tokens` complete Mamba blocks using
/// MARS dimensions: D=20, N=8, E=2, int8 activations.
fn make_trace(layers: u32, tokens: u32) -> String {
    let mut s = String::new();
    for l in 0..layers {
        for t in 0..tokens {
            // Use distinct addresses per (layer, token) to avoid aliasing.
            let base = 0x1000u64 + (l as u64 * 0x10000) + (t as u64 * 0x200);
            let dram_src = 0x8000_0000u64 + (l as u64 * 0x1000_0000) + (t as u64 * 0x1000);

            s.push_str(&format!("LAYER {} TOKEN {}\n", l, t));
            s.push_str(&format!(
                "LOAD src={:#x} src_mem=dram dst={:#x} dst_mem=sram bytes=20\n",
                dram_src,
                base
            ));
            s.push_str(&format!(
                "RANGE_NORM in={:#x} in_mem=sram out={:#x} out_mem=sram D=20 dtype=int8\n",
                base,
                base + 0x10
            ));
            s.push_str(&format!(
                "LINEAR in={:#x} in_mem=sram out={:#x} out_mem=sram in_dim=20 out_dim=40 dtype=int8\n",
                base + 0x10,
                base + 0x20
            ));
            s.push_str(&format!(
                "CONV1D in={:#x} in_mem=sram out={:#x} out_mem=sram D=20 K=4 dtype=int8\n",
                base + 0x20,
                base + 0x30
            ));
            s.push_str(&format!(
                "PWL_SILU in={:#x} in_mem=sram out={:#x} out_mem=sram D=20 dtype=int8\n",
                base + 0x30,
                base + 0x40
            ));
            s.push_str(&format!(
                "SSM_STATE in={:#x} in_mem=sram state={:#x} state_mem=sram D=20 N=8 E=2 dtype=int8 state_dtype=int24\n",
                base + 0x40,
                base + 0x100
            ));
            s.push_str(&format!(
                "SSM_OUTPUT state={:#x} state_mem=sram in={:#x} in_mem=sram out={:#x} out_mem=sram D=20 N=8 E=2 dtype=int8\n",
                base + 0x100,
                base + 0x40,
                base + 0x50
            ));
            s.push_str(&format!(
                "PWL_SILU in={:#x} in_mem=sram out={:#x} out_mem=sram D=20 dtype=int8\n",
                base + 0x50,
                base + 0x60
            ));
            s.push_str(&format!(
                "LINEAR in={:#x} in_mem=sram out={:#x} out_mem=sram in_dim=20 out_dim=20 dtype=int8\n",
                base + 0x60,
                base + 0x70
            ));
            s.push_str(&format!(
                "RESIDUAL a={:#x} a_mem=sram b={:#x} b_mem=sram out={:#x} out_mem=sram D=20 dtype=int8\n",
                base + 0x70,
                base,
                base + 0x80
            ));
            s.push_str(&format!(
                "STORE src={:#x} src_mem=sram dst={:#x} dst_mem=dram bytes=20\n",
                base + 0x80,
                dram_src + 0x1000
            ));
        }
    }
    s
}

/// CPU-derived reference: the DRAM memmove unit is the serialisation
/// bottleneck (single slot, bw=8, lat=80). Every block contributes one LOAD
/// and one STORE, each taking `memmove_cycles` cycles.  They are fully
/// serialised through the single MemmoveStage slot, so:
///
///   reference_lower ≈ num_blocks × 2 × memmove_cycles
///
/// All compute ops overlap with DRAM (different units), so they add at most
/// the dispatch overhead per block on top of the DRAM time.  We use a
/// generous upper bound of:
///
///   reference_upper = num_blocks × (2 × memmove_cycles + per_block_compute_sum)
fn reference_envelope(layers: u32, tokens: u32) -> (u64, u64) {
    let memmove = MemmoveStage::new().cycles_for(20) as u64; // 83
    let rn_cycles = RangeNormUnit::new(10, 4).cycles_for(20) as u64; // 54
    // Per-block compute sum (generous): RangeNorm + 2×Linear + Conv1D + 2×PwlSilu
    //   + SsmState + SsmOutput + ResidualAdd
    let linear1 = LinearProjUnit::new(64, 2).cycles_for(20, 40) as u64;
    let linear2 = LinearProjUnit::new(64, 2).cycles_for(20, 20) as u64;
    let per_block_compute = rn_cycles + linear1 + linear2 + 25 + 20 + 20 + 12 + 12 + 2;

    let num_blocks = (layers * tokens) as u64;
    let lower = num_blocks * 2 * memmove;
    let upper = num_blocks * (2 * memmove + per_block_compute) + 50; // +50 dispatch slack
    (lower, upper)
}

/// Write trace to a temp file and invoke `emamba::run`, then validate the
/// cycle count against the CPU-derived reference envelope.
fn run_workload_scale(layers: u32, tokens: u32) {
    use std::io::Write;
    let trace_str = make_trace(layers, tokens);
    let num_blocks = layers * tokens;
    let expected_ops = num_blocks * 11; // 11 ops per Mamba block

    // Write to a temp file so `emamba::run` can open it.
    let tmp_path = std::env::temp_dir().join(format!(
        "emamba_workload_scale_{}l_{}t.trace",
        layers, tokens
    ));
    std::fs::write(&tmp_path, &trace_str).expect("write trace");

    let root = workspace_root();
    let cfg = root.join("examples/mars.yaml");

    let result = emamba::run(&cfg, &tmp_path).expect("run ok");
    let yaml = result.stats.to_yaml_string();

    assert!(
        yaml.contains(&format!("ops_completed: {}", expected_ops)),
        "expected ops_completed={} for {}L×{}T:\n{}",
        expected_ops, layers, tokens, yaml
    );

    let total = result
        .stats
        .get_u64(&["simulator", "total_cycles"])
        .expect("total_cycles");

    let (lower, upper) = reference_envelope(layers, tokens);
    assert!(
        total >= lower,
        "total_cycles={} below lower bound {} — DRAM not being modelled ({}L×{}T)",
        total, lower, layers, tokens
    );
    assert!(
        total <= upper,
        "total_cycles={} above upper bound {} — possible dispatch loop bug ({}L×{}T)",
        total, upper, layers, tokens
    );
}

#[test]
fn workload_scale_1l_1t() {
    run_workload_scale(1, 1);
}

#[test]
fn workload_scale_2l_2t() {
    run_workload_scale(2, 2);
}

#[test]
fn workload_scale_4l_1t() {
    run_workload_scale(4, 1);
}
