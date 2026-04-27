"""Performance analysis of the eMamba paper workload.

Loads state-spaces/mamba-130m-hf, runs prefill on the paper's prompt and 10
decode steps, and reports per-paper-block time/FLOPs/bytes/arithmetic-intensity
broken down between prefill and decode. Categories follow Section 3-4 of the
eMamba paper: embedding, norm (range-norm target), in_proj, conv1d, x_proj,
dt_proj, ssm_scan_etc (the recurrence + Δ-discretization + activations),
out_proj, lm_head.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaMixer

MODEL_NAME = "state-spaces/mamba-130m-hf"
PROMPT = "Hey how are you doing?"
NEW_TOKENS = 10
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


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


def tensor_bytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (tuple, list)):
        return sum(tensor_bytes(v) for v in x)
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    return 0


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


class Recorder:
    def __init__(self):
        self.events = []
        self.mixer_events = []
        self._stack = []

    def leaf_pre(self, name, category):
        def _h(module, args):
            tin = first_tensor(args)
            self._stack.append(
                (name, category, time.perf_counter(),
                 tuple(tin.shape) if tin is not None else None,
                 tensor_bytes(args))
            )
        return _h

    def leaf_post(self, name, category):
        def _h(module, args, output):
            n, c, t0, in_shape, in_bytes = self._stack.pop()
            t1 = time.perf_counter()
            tout = first_tensor(output)
            out_bytes = tensor_bytes(output)

            tokens = 0
            seq_len = 1
            flops = 0

            if isinstance(module, nn.Linear):
                if in_shape is not None:
                    tokens = 1
                    for d in in_shape[:-1]:
                        tokens *= d
                    seq_len = in_shape[-2] if len(in_shape) >= 2 else 1
                flops = 2 * tokens * module.in_features * module.out_features
            elif isinstance(module, nn.Conv1d):
                # input shape (B, C, L)
                if tout is not None:
                    tokens = tout.shape[0] * tout.shape[-1]
                if in_shape is not None and len(in_shape) >= 3:
                    seq_len = in_shape[-1]
                k = module.kernel_size[0]
                in_c_per_group = module.in_channels // module.groups
                flops = 2 * tokens * k * in_c_per_group
            elif isinstance(module, nn.Embedding):
                if tout is not None:
                    tokens = tout.shape[0] * tout.shape[1]
                if in_shape is not None:
                    seq_len = in_shape[-1]
                flops = 0
            else:  # norm
                if tout is not None:
                    tokens = tout.shape[0] * tout.shape[1]
                    flops = 5 * tout.numel()
                if in_shape is not None and len(in_shape) >= 2:
                    seq_len = in_shape[-2]

            wb = sum(p.numel() * p.element_size()
                     for p in module.parameters(recurse=False))

            self.events.append({
                "name": n, "category": c,
                "time_ms": (t1 - t0) * 1000.0,
                "in_bytes": in_bytes, "out_bytes": out_bytes,
                "weight_bytes": wb, "flops": flops,
                "tokens": tokens, "seq_len": seq_len,
            })
        return _h

    def mixer_pre(self):
        def _h(module, args):
            tin = first_tensor(args)
            self._stack.append(
                ("__mixer__", "mixer", time.perf_counter(),
                 tuple(tin.shape) if tin is not None else None, 0)
            )
        return _h

    def mixer_post(self, ed, n_state, dtype_bytes):
        def _h(module, args, output):
            _, _, t0, in_shape, _ = self._stack.pop()
            t1 = time.perf_counter()
            seq_len = (in_shape[-2] if in_shape is not None and len(in_shape) >= 2
                       else 1)
            # Analytical FLOPs for ops inside the mixer that leaf hooks don't
            # cover (SSM scan, Δ-discretization, activations).
            #   Δ = softplus(dt_proj(x))                   ED   ops/token
            #   Ā = exp(Δ * A)                             2*ED*N ops/token
            #   B̄ = Δ * B * x_t                            3*ED*N ops/token
            #   h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t                3*ED*N ops/token
            #   y_t = C · h_t                              2*ED*N ops/token
            #   D * x_t                                    ED   ops/token
            #   silu(gate) ⊙ y_t                           2*ED ops/token
            ssm_flops = ((2 + 3 + 3 + 2) * ed * n_state + 4 * ed) * seq_len
            # Bytes streamed per step in the recurrence:
            #   h_{t-1} read + h_t write:        2 * ED * N
            #   Ā, B̄, C reads:                   3 * ED * N
            #   x_t read + y_t write:            2 * ED
            ssm_bytes = (5 * ed * n_state + 2 * ed) * seq_len * dtype_bytes
            self.mixer_events.append({
                "time_ms": (t1 - t0) * 1000.0,
                "seq_len": seq_len,
                "ssm_flops": ssm_flops,
                "ssm_bytes": ssm_bytes,
            })
        return _h


def aggregate(events):
    agg = defaultdict(lambda: dict(
        time_ms=0.0, flops=0, in_bytes=0, out_bytes=0,
        weight_bytes=0, calls=0))
    for e in events:
        a = agg[e["category"]]
        a["time_ms"] += e["time_ms"]
        a["flops"] += e["flops"]
        a["in_bytes"] += e["in_bytes"]
        a["out_bytes"] += e["out_bytes"]
        a["weight_bytes"] += e["weight_bytes"]
        a["calls"] += 1
    return dict(agg)


def fmt_table(agg, label):
    order = ["embedding", "norm", "in_proj", "conv1d", "x_proj",
             "dt_proj", "ssm_scan_etc", "out_proj", "lm_head", "other"]
    sum_t = sum(v["time_ms"] for v in agg.values()) or 1.0
    sum_f = sum(v["flops"] for v in agg.values()) or 1
    lines = [
        "",
        f"=== {label}  total={sum_t:.2f} ms  total={sum_f/1e9:.3f} GFLOPs ===",
        f"{'category':<14}{'calls':>6}{'time_ms':>11}{'time%':>7}"
        f"{'GFLOPs':>10}{'flop%':>7}{'in_MB':>8}{'wt_MB':>8}{'AI_F/B':>10}",
    ]
    for k in order:
        if k not in agg:
            continue
        v = agg[k]
        ai = v["flops"] / max(1, v["in_bytes"] + v["weight_bytes"])
        lines.append(
            f"{k:<14}{v['calls']:>6}{v['time_ms']:>11.3f}"
            f"{v['time_ms']/sum_t*100:>6.1f}%"
            f"{v['flops']/1e9:>10.3f}"
            f"{v['flops']/sum_f*100:>6.1f}%"
            f"{v['in_bytes']/1e6:>8.2f}"
            f"{v['weight_bytes']/1e6:>8.2f}"
            f"{ai:>10.2f}"
        )
    return "\n".join(lines)


def mixer_residual(mixer_events, leaf_events):
    inside = ("in_proj", "conv1d", "x_proj", "dt_proj", "out_proj")
    leaf_t = sum(e["time_ms"] for e in leaf_events if e["category"] in inside)
    total_t = sum(e["time_ms"] for e in mixer_events)
    flops = sum(e["ssm_flops"] for e in mixer_events)
    bytes_streamed = sum(e["ssm_bytes"] for e in mixer_events)
    return dict(time_ms=max(0.0, total_t - leaf_t),
                flops=flops, in_bytes=bytes_streamed, out_bytes=0,
                weight_bytes=0, calls=len(mixer_events))


def main():
    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME).eval()
    cfg = model.config
    print(f"  D={cfg.hidden_size} ED={cfg.intermediate_size} "
          f"N={cfg.state_size} d_conv={cfg.conv_kernel} "
          f"M={cfg.num_hidden_layers} vocab={cfg.vocab_size}")

    rec = Recorder()
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding)) or \
                "norm" in m.__class__.__name__.lower():
            cat = categorize(name, m)
            if cat is None:
                continue
            handles.append(m.register_forward_pre_hook(rec.leaf_pre(name, cat)))
            handles.append(m.register_forward_hook(rec.leaf_post(name, cat)))
    dtype_bytes = next(model.parameters()).element_size()
    for name, m in model.named_modules():
        if isinstance(m, MambaMixer):
            handles.append(m.register_forward_pre_hook(rec.mixer_pre()))
            handles.append(m.register_forward_hook(rec.mixer_post(
                cfg.intermediate_size, cfg.state_size, dtype_bytes)))

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    print(f"  prompt={PROMPT!r}  prefill_len={input_ids.shape[-1]}  "
          f"new_tokens={NEW_TOKENS}")

    # Warm up the *entire* generate path (prefill + at least one decode step)
    # so JIT / lazy init costs don't get attributed to the first hooked call.
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=2,
                           pad_token_id=tokenizer.eos_token_id or 0,
                           do_sample=False)
    rec.events.clear()
    rec.mixer_events.clear()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id or 0,
            do_sample=False,
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    decoded = tokenizer.batch_decode(out)[0]

    for h in handles:
        h.remove()

    prefill_events = [e for e in rec.events if e["seq_len"] > 1]
    decode_events = [e for e in rec.events if e["seq_len"] == 1]
    prefill_mixer = [e for e in rec.mixer_events if e["seq_len"] > 1]
    decode_mixer = [e for e in rec.mixer_events if e["seq_len"] == 1]

    pre_agg = aggregate(prefill_events)
    dec_agg = aggregate(decode_events)
    pre_agg["ssm_scan_etc"] = mixer_residual(prefill_mixer, prefill_events)
    dec_agg["ssm_scan_etc"] = mixer_residual(decode_mixer, decode_events)

    prefill_len = max((e["seq_len"] for e in prefill_events), default=0)

    pre_total_ms = sum(v["time_ms"] for v in pre_agg.values())
    dec_total_ms = sum(v["time_ms"] for v in dec_agg.values())
    pre_total_flops = sum(v["flops"] for v in pre_agg.values())
    dec_total_flops = sum(v["flops"] for v in dec_agg.values())

    def top_n(agg, key, n=3):
        items = sorted(agg.items(), key=lambda kv: kv[1][key], reverse=True)
        return ", ".join(
            f"{k}={v[key]/1e6:.1f}MB" if key == "weight_bytes"
            else f"{k}={v[key]/1e9:.3f}GF" if key == "flops"
            else f"{k}={v[key]:.1f}ms"
            for k, v in items[:n] if v[key] > 0)

    head = [
        "=" * 84,
        f"eMamba workload performance analysis — {MODEL_NAME}",
        f"  prompt={PROMPT!r}  prefill_len={prefill_len}  "
        f"decode_steps={NEW_TOKENS}",
        f"  generated: {decoded}",
        f"  config: D={cfg.hidden_size}  ED={cfg.intermediate_size}  "
        f"N={cfg.state_size}  d_conv={cfg.conv_kernel}  "
        f"M={cfg.num_hidden_layers}  vocab={cfg.vocab_size}",
        f"  weights: {sum(p.numel() for p in model.parameters())/1e6:.1f} M "
        f"params × {dtype_bytes} B = "
        f"{sum(p.numel()*p.element_size() for p in model.parameters())/1e6:.1f} MB",
        f"  wall-time(generate): {wall_ms:.2f} ms  "
        f"(≈ {wall_ms/(prefill_len + NEW_TOKENS):.1f} ms/token avg)",
        "-" * 84,
        "SUMMARY",
        f"  prefill   {pre_total_ms:>7.1f} ms   {pre_total_flops/1e9:>6.2f} GF   "
        f"top-time: {top_n(pre_agg,'time_ms')}",
        f"  decode    {dec_total_ms:>7.1f} ms   {dec_total_flops/1e9:>6.2f} GF   "
        f"top-time: {top_n(dec_agg,'time_ms')}",
        f"  decode/token  ~{dec_total_ms/NEW_TOKENS:.1f} ms",
        f"  decode top-weight-traffic: {top_n(dec_agg,'weight_bytes')}",
        "  Decode is memory-bound: AI < 1 F/B for every linear (each weight",
        "  byte fetched does < 1 FLOP because batch=1 reuses nothing). The",
        "  SSM scan is also bandwidth-bound on the recurrent state. This is",
        "  exactly the regime eMamba's on-chip pipeline (Fig. 5) attacks.",
        "  Caveats:",
        "   - prefill conv1d wall-time is inflated: HF slow_forward runs a",
        "     depthwise Conv1d (1536 groups) on CPU which is pathologically",
        "     slow without the causal_conv1d_fn kernel. On the eMamba ASIC",
        "     this is a small custom block, not a bottleneck.",
        "   - decode has no conv1d row: Mamba's step() bypasses the Conv1d",
        "     module and folds the cached conv into ssm_scan_etc.",
        "   - lm_head and embedding share weights in mamba-130m, so the two",
        "     154 MB-per-call entries are the same tensor (cache may reuse).",
        "=" * 84,
    ]
    text = "\n".join(head) + \
        fmt_table(pre_agg, f"PREFILL  (1 forward × L={prefill_len})") + \
        fmt_table(dec_agg, f"DECODE   ({NEW_TOKENS} forwards × L=1)")

    print(text)

    (RESULTS / "report.txt").write_text(text)
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "prefill_len": prefill_len,
        "decode_steps": NEW_TOKENS,
        "wall_time_ms": wall_ms,
        "config": {k: getattr(cfg, k) for k in (
            "hidden_size", "intermediate_size", "state_size",
            "conv_kernel", "num_hidden_layers", "vocab_size")},
        "prefill": pre_agg,
        "decode": dec_agg,
    }
    (RESULTS / "report.json").write_text(
        json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {RESULTS}/report.txt and report.json")


if __name__ == "__main__":
    main()
