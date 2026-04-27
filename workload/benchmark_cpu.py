"""CPU performance benchmark for the eMamba paper workload.

For each configuration:
  - runs WARMUP+ITERS iterations of model.generate(prompt, max_new_tokens=N)
  - times every leaf kernel (Embedding / Norm / Linear / Conv1d) and the
    MambaMixer (so the SSM scan + cached conv + activations show up as a
    `ssm_scan_etc` residual)
  - splits records into prefill (forward call with L>1) and decode (L=1)
  - computes mean/median/p95/min/max/stdev across iterations

Sweeps prefill length to illustrate SSM linear scaling. Saves
benchmark.txt + benchmark.json under results/.
"""

import json
import platform
import resource
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaMixer

MODEL_NAME = "state-spaces/mamba-130m-hf"
BASE_PROMPT = "Hey how are you doing?"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
WARMUP = 2
ITERS = 6
DECODE_TOKENS = 10
PREFILL_SWEEP = [6, 32, 128, 512]


# ---------- system info ----------

def cpu_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    try:
        for key, sysctl in [
            ("cpu_brand", "machdep.cpu.brand_string"),
            ("logical_cores", "hw.ncpu"),
            ("physical_cores", "hw.physicalcpu"),
            ("memory_bytes", "hw.memsize"),
        ]:
            v = subprocess.check_output(["sysctl", "-n", sysctl]).decode().strip()
            info[key] = int(v) if v.isdigit() else v
        if "memory_bytes" in info:
            info["memory_gb"] = round(info["memory_bytes"] / 1e9, 1)
    except Exception:
        pass
    return info


# ---------- stats ----------

def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(vs):
    if not vs:
        return {}
    return {
        "n": len(vs),
        "min": min(vs),
        "max": max(vs),
        "mean": statistics.mean(vs),
        "median": statistics.median(vs),
        "p95": percentile(vs, 95),
        "stdev": statistics.stdev(vs) if len(vs) > 1 else 0.0,
    }


# ---------- hook recorder ----------

def first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            t = first_tensor(v)
            if t is not None:
                return t
    if isinstance(x, dict):
        for v in x.values():
            t = first_tensor(v)
            if t is not None:
                return t
    return None


def categorize(name: str, module: nn.Module):
    if isinstance(module, nn.Embedding):
        return "embedding"
    if name.endswith("lm_head"):
        return "lm_head"
    if isinstance(module, nn.Conv1d):
        return "conv1d"
    if name.endswith("in_proj"):
        return "in_proj"
    if name.endswith("x_proj"):
        return "x_proj"
    if name.endswith("dt_proj"):
        return "dt_proj"
    if name.endswith("out_proj"):
        return "out_proj"
    cls = module.__class__.__name__.lower()
    if "norm" in cls:
        return "norm"
    return None


class Rec:
    def __init__(self):
        self.events = []
        self.mixer = []
        self._st = []

    def leaf_pre(self, cat):
        def _h(m, a):
            self._st.append((cat, time.perf_counter()))
        return _h

    def leaf_post(self, cat):
        def _h(m, a, o):
            c, t0 = self._st.pop()
            t1 = time.perf_counter()
            tin = first_tensor(a)
            tout = first_tensor(o)
            seq_len = 1
            if isinstance(m, nn.Linear) and tin is not None and tin.dim() >= 2:
                seq_len = tin.shape[-2]
            elif isinstance(m, nn.Conv1d) and tin is not None and tin.dim() >= 3:
                seq_len = tin.shape[-1]
            elif isinstance(m, nn.Embedding) and tin is not None and tin.dim() >= 1:
                seq_len = tin.shape[-1]
            elif tout is not None and tout.dim() >= 2:
                seq_len = tout.shape[-2]
            self.events.append({"cat": c, "time_ms": (t1 - t0) * 1000.0,
                                "seq_len": seq_len})
        return _h

    def mixer_pre(self):
        def _h(m, a):
            tin = first_tensor(a)
            sl = tin.shape[-2] if tin is not None and tin.dim() >= 2 else 1
            self._st.append(("__mixer__", time.perf_counter(), sl))
        return _h

    def mixer_post(self):
        def _h(m, a, o):
            _, t0, sl = self._st.pop()
            t1 = time.perf_counter()
            self.mixer.append({"time_ms": (t1 - t0) * 1000.0, "seq_len": sl})
        return _h


def install_hooks(model):
    rec = Rec()
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding)) or \
                "norm" in m.__class__.__name__.lower():
            cat = categorize(name, m)
            if cat is None:
                continue
            handles.append(m.register_forward_pre_hook(rec.leaf_pre(cat)))
            handles.append(m.register_forward_hook(rec.leaf_post(cat)))
    for name, m in model.named_modules():
        if isinstance(m, MambaMixer):
            handles.append(m.register_forward_pre_hook(rec.mixer_pre()))
            handles.append(m.register_forward_hook(rec.mixer_post()))
    return rec, handles


# ---------- one iteration ----------

def run_one_iter(model, tokenizer, input_ids, decode_tokens):
    rec, handles = install_hooks(model)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=decode_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id or 0)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    for h in handles:
        h.remove()

    # split events into prefill / decode
    prefill_events = [e for e in rec.events if e["seq_len"] > 1]
    decode_events = [e for e in rec.events if e["seq_len"] == 1]
    prefill_mixer = [e for e in rec.mixer if e["seq_len"] > 1]
    decode_mixer = [e for e in rec.mixer if e["seq_len"] == 1]

    prefill_ms = sum(e["time_ms"] for e in prefill_mixer)
    # prefill_mixer captures the whole MambaMixer time for the prefill forward
    # but other layers (norm, embedding, lm_head) sit outside the mixer.
    # Get a better total: sum of all leaf events by phase + mixer residual.
    inside = ("in_proj", "conv1d", "x_proj", "dt_proj", "out_proj")

    def phase_breakdown(leaves, mixer_evs):
        b = defaultdict(lambda: dict(time_ms=0.0, calls=0))
        for e in leaves:
            d = b[e["cat"]]
            d["time_ms"] += e["time_ms"]
            d["calls"] += 1
        leaf_in_mixer = sum(e["time_ms"] for e in leaves
                            if e["cat"] in inside)
        mixer_total = sum(e["time_ms"] for e in mixer_evs)
        b["ssm_scan_etc"] = dict(
            time_ms=max(0.0, mixer_total - leaf_in_mixer),
            calls=len(mixer_evs))
        b["__total__"] = dict(
            time_ms=sum(v["time_ms"] for v in b.values()),
            calls=sum(v["calls"] for v in b.values()))
        return dict(b)

    prefill_bd = phase_breakdown(prefill_events, prefill_mixer)
    decode_bd = phase_breakdown(decode_events, decode_mixer)

    # decode wall-clock per token = mixer_decode_total / num_decode_calls
    # but we want per-step accuracy: get per-call mixer times grouped by
    # decode-step. Each generate() decode forward triggers
    # `num_hidden_layers` mixer calls. So decode_steps = #mixer_decode / M.
    M = model.config.num_hidden_layers
    n_decode_calls = len(decode_mixer) // M
    decode_step_ms = []
    if n_decode_calls > 0:
        # Group adjacent M mixer events into one decode step. But we also need
        # to add the non-mixer portions per step. Easier path: derive per-step
        # latency from wall_ms minus prefill — works because generate() loops
        # are sequential and dominated by these forward calls.
        # We'll instead approximate per-step decode latency from the sum of
        # all decode-leaf events plus decode mixer residual, divided by steps.
        decode_total = decode_bd["__total__"]["time_ms"]
        per_step = decode_total / n_decode_calls
        decode_step_ms = [per_step] * n_decode_calls

    return {
        "wall_ms": wall_ms,
        "prefill_ms": prefill_bd["__total__"]["time_ms"],
        "decode_ms": decode_bd["__total__"]["time_ms"],
        "decode_steps": n_decode_calls,
        "decode_step_ms": decode_step_ms,
        "prefill_breakdown": prefill_bd,
        "decode_breakdown": decode_bd,
    }


# ---------- benchmark loop ----------

def benchmark(model, tokenizer, input_ids, label, decode_tokens=DECODE_TOKENS,
              warmup=WARMUP, iters=ITERS):
    L = int(input_ids.shape[-1])
    print(f"  [{label}] L={L} new_tokens={decode_tokens} "
          f"warmup={warmup} iters={iters}", flush=True)
    for _ in range(warmup):
        run_one_iter(model, tokenizer, input_ids, 2)
    walls, prefills, decodes, dec_steps_all = [], [], [], []
    # per-(cat, phase) total ms across iters
    kbd = {"prefill": defaultdict(list), "decode": defaultdict(list)}
    kcalls = {"prefill": defaultdict(int), "decode": defaultdict(int)}
    for _ in range(iters):
        r = run_one_iter(model, tokenizer, input_ids, decode_tokens)
        walls.append(r["wall_ms"])
        prefills.append(r["prefill_ms"])
        decodes.append(r["decode_ms"])
        dec_steps_all.extend(r["decode_step_ms"])
        for cat, v in r["prefill_breakdown"].items():
            if cat == "__total__":
                continue
            kbd["prefill"][cat].append(v["time_ms"])
            kcalls["prefill"][cat] = v["calls"]
        for cat, v in r["decode_breakdown"].items():
            if cat == "__total__":
                continue
            kbd["decode"][cat].append(v["time_ms"])
            kcalls["decode"][cat] = v["calls"]
    return {
        "label": label,
        "prefill_len": L,
        "decode_tokens": decode_tokens,
        "wall_ms": stats(walls),
        "prefill_ms": stats(prefills),
        "decode_ms": stats(decodes),
        "decode_step_ms": stats(dec_steps_all),
        "decode_tokens_per_sec_mean":
            decode_tokens * 1000.0 / statistics.mean(decodes)
            if decodes else 0.0,
        "prefill_tokens_per_sec_mean":
            L * 1000.0 / statistics.mean(prefills) if prefills else 0.0,
        "kernel_ms_per_iter": {
            "prefill": {k: stats(v) for k, v in kbd["prefill"].items()},
            "decode": {k: stats(v) for k, v in kbd["decode"].items()},
        },
        "kernel_calls_per_iter": {
            "prefill": dict(kcalls["prefill"]),
            "decode": dict(kcalls["decode"]),
        },
    }


# ---------- formatting ----------

def fmt_kernel_table(bench, phase, label):
    order = ["embedding", "norm", "in_proj", "conv1d", "x_proj",
             "dt_proj", "ssm_scan_etc", "out_proj", "lm_head"]
    bd = bench["kernel_ms_per_iter"][phase]
    calls = bench["kernel_calls_per_iter"][phase]
    sum_mean = sum(v["mean"] for v in bd.values()) or 1.0
    lines = [f"  {label}  (per-iteration totals across {ITERS} iters):",
             f"    {'kernel':<14}{'calls/it':>9}{'mean ms':>10}"
             f"{'p50 ms':>9}{'p95 ms':>9}{'min':>8}{'max':>8}{'stdev':>8}"
             f"{'time%':>8}"]
    for k in order:
        if k not in bd:
            continue
        v = bd[k]
        lines.append(
            f"    {k:<14}{calls.get(k, 0):>9}"
            f"{v['mean']:>10.3f}{v['median']:>9.3f}{v['p95']:>9.3f}"
            f"{v['min']:>8.3f}{v['max']:>8.3f}{v['stdev']:>8.3f}"
            f"{v['mean']/sum_mean*100:>7.1f}%")
    return "\n".join(lines)


def fmt_one(b):
    p = b["prefill_ms"]; d = b["decode_step_ms"]; w = b["wall_ms"]
    return (
        f"  L={b['prefill_len']:>4} | "
        f"prefill {p['mean']:>7.1f}±{p['stdev']:<5.1f} (p95 {p['p95']:>6.1f}) | "
        f"decode/tok {d['mean']:>5.2f}±{d['stdev']:<4.2f} (p95 {d['p95']:>5.2f}) | "
        f"{b['decode_tokens_per_sec_mean']:>5.2f} tok/s | "
        f"wall {w['mean']:>7.1f}±{w['stdev']:<5.1f}"
    )


def main():
    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME).eval()
    info = cpu_info()
    print(f"CPU: {info.get('cpu_brand', info.get('processor', '?'))}")
    print(f"     logical={info.get('logical_cores', '?')} "
          f"physical={info.get('physical_cores', '?')} "
          f"mem={info.get('memory_gb', '?')} GB  "
          f"torch_threads={info['torch_threads']}")

    results = {
        "model": MODEL_NAME,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "weight_MB": sum(p.numel() * p.element_size()
                         for p in model.parameters()) / 1e6,
        "cpu": info,
        "config": {"warmup": WARMUP, "iters": ITERS,
                   "decode_tokens": DECODE_TOKENS,
                   "prefill_sweep": PREFILL_SWEEP},
        "primary": None,
        "prefill_sweep": [],
    }

    # primary
    input_ids = tokenizer(BASE_PROMPT, return_tensors="pt")["input_ids"]
    print("\nPrimary benchmark (paper prompt):")
    results["primary"] = benchmark(model, tokenizer, input_ids,
                                   label="paper_prompt")

    # sweep prefill length
    big_ids = tokenizer(" ".join(["the"] * 600),
                        return_tensors="pt")["input_ids"]
    print("\nPrefill-length sweep:")
    for plen in PREFILL_SWEEP:
        ids = big_ids[:, :plen].clone()
        results["prefill_sweep"].append(
            benchmark(model, tokenizer, ids, label=f"prefill_{plen}"))

    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = ru.ru_maxrss if ru.ru_maxrss > 1e9 else ru.ru_maxrss * 1024
    results["peak_rss_MB"] = round(rss_bytes / 1e6, 1)

    # ----- pretty print -----
    L = []
    L.append("=" * 92)
    L.append(f"CPU performance benchmark — {MODEL_NAME}")
    L.append(f"  CPU: {info.get('cpu_brand', info.get('processor'))}")
    L.append(f"       logical={info.get('logical_cores')} "
             f"physical={info.get('physical_cores')} "
             f"mem={info.get('memory_gb')} GB")
    L.append(f"  torch={info['torch']}  threads={info['torch_threads']}  "
             f"interop={info['torch_interop_threads']}")
    L.append(f"  params={results['params_M']:.1f} M  "
             f"weights={results['weight_MB']:.1f} MB (fp32)")
    L.append(f"  warmup={WARMUP}  iters={ITERS}  decode_tokens={DECODE_TOKENS}")
    L.append(f"  peak_rss={results['peak_rss_MB']} MB")
    L.append("-" * 92)

    L.append("Primary (paper prompt):")
    L.append(fmt_one(results["primary"]))
    L.append("")
    L.append(fmt_kernel_table(results["primary"], "prefill",
                              f"PREFILL  L={results['primary']['prefill_len']}"))
    L.append("")
    L.append(fmt_kernel_table(results["primary"], "decode",
                              f"DECODE   {DECODE_TOKENS}×L=1"))

    L.append("")
    L.append("Prefill-length sweep:")
    for r in results["prefill_sweep"]:
        L.append(fmt_one(r))

    # delta-table for prefill scaling
    L.append("")
    L.append("Prefill scaling (mean prefill_ms / L  →  per-token prefill cost):")
    for r in results["prefill_sweep"]:
        per_tok = r["prefill_ms"]["mean"] / r["prefill_len"]
        L.append(f"  L={r['prefill_len']:>4}  "
                 f"prefill_mean={r['prefill_ms']['mean']:>8.2f} ms  "
                 f"per-token={per_tok:>6.3f} ms  "
                 f"({r['prefill_tokens_per_sec_mean']:>7.1f} tok/s)")

    L.append("")
    L.append("Notes:")
    L.append("  - Per-iteration totals = per kernel time within ONE generate() "
             "call (prefill + 10 decodes), aggregated across iters.")
    L.append("  - Decode steps share latency: decode_step_ms ≈ "
             "decode_ms_total / num_decode_steps per iter.")
    L.append("  - Prefill conv1d is dominated by HF's slow_forward "
             "depthwise Conv1d (1536 groups) on CPU; would be tiny on the "
             "eMamba ASIC.")
    L.append("=" * 92)

    txt = "\n".join(L)
    print("\n" + txt)
    (RESULTS / "benchmark.txt").write_text(txt)
    (RESULTS / "benchmark.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\nSaved {RESULTS}/benchmark.txt")
    print(f"Saved {RESULTS}/benchmark.json")


if __name__ == "__main__":
    main()
