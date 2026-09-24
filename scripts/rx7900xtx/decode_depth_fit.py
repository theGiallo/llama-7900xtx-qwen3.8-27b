#!/usr/bin/env python3
"""Split decode time into a fixed part and a per-context part, per model and KV type.

Runs `llama-bench -p 0 -n N -d <depths>` (token generation after a prefilled context of
each depth), then fits   ms/token = fixed + slope * depth   and reports:

  - fixed ms/token and the implied weight bandwidth (model bytes / fixed time).
    The fixed part also holds launch overhead, DeltaNet state, norms, sampling, so this is a
    lower bound on how well the weight matmuls do.
  - slope in ms per 1k tokens of context and the implied KV bandwidth
    (KV bytes per token of context / slope), compared to peak.

The KV bytes per token are read from the GGUF: attn_k / attn_v rows of every
full-attention layer of the main model (MTP/nextn layers excluded).

Examples:
  scripts/rx7900xtx/decode_depth_fit.py --bench build/bin/llama-bench -m model.gguf --ctk q4_0,f16
  scripts/rx7900xtx/decode_depth_fit.py --jsonl saved.jsonl -m model.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


def kv_elements_per_token(model_path: str) -> tuple[int, int, int]:
    """Return (K elements, V elements, n full-attention layers) per token of context."""
    from gguf import GGUFReader  # type: ignore

    reader = GGUFReader(model_path, "r")

    def field(name: str):
        f = reader.get_field(name)
        return None if f is None else f.contents()

    arch = field("general.architecture")
    n_layer = int(field(f"{arch}.block_count") or 0)
    n_nextn = int(field(f"{arch}.nextn_predict_layers") or 0)
    n_main = n_layer - n_nextn

    k_elems = v_elems = 0
    layers = set()
    for t in reader.tensors:
        parts = t.name.split(".")
        if len(parts) < 4 or parts[0] != "blk" or parts[3] != "weight":
            continue
        il = int(parts[1])
        if il >= n_main:
            continue
        if parts[2] == "attn_k":
            k_elems += int(t.shape[1])
            layers.add(il)
        elif parts[2] == "attn_v":
            v_elems += int(t.shape[1])
    return k_elems, v_elems, len(layers)


def fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares y = a + b*x; returns (a, b, r2)."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for _, y in points)
    ss_res = sum((y - a - b * x) ** 2 for x, y in points)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


def run_bench(args, model: str, ctk: str) -> list[dict]:
    cmd = [
        args.bench, "-m", model, "-ngl", "99", "-fa", "on", "-ctk", ctk, "-ctv", ctk,
        "-p", "0", "-n", str(args.n_gen), "-d", args.depths, "-r", str(args.reps), "-o", "jsonl",
    ] + args.extra
    env = dict(os.environ, GGML_CUDA_FATTN_LOG="1")
    print("running:", " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if args.save_dir:
        stem = Path(args.save_dir) / f"{Path(model).stem}.{ctk}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".jsonl").write_text(proc.stdout)
        stem.with_suffix(".log").write_text(proc.stderr)
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0:
        print(f"llama-bench exited with {proc.returncode} for {model} ctk={ctk} (OOM at a large depth?); "
              f"keeping {len(rows)} results", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
    for line in proc.stderr.splitlines():
        if line.startswith("fattn:"):
            print("  " + line, file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="build/bin/llama-bench", help="llama-bench binary")
    ap.add_argument("-m", "--model", action="append", required=True, help="GGUF model (repeatable)")
    ap.add_argument("--ctk", default="q4_0,q8_0,f16", help="comma separated KV cache types (K and V)")
    ap.add_argument("--depths", default="0,32768,65536,131072", help="comma separated context depths")
    ap.add_argument("-n", "--n-gen", type=int, default=64, help="tokens generated per measurement")
    ap.add_argument("-r", "--reps", type=int, default=2, help="repetitions per depth")
    ap.add_argument("--jsonl", action="append", help="parse saved llama-bench jsonl instead of running")
    ap.add_argument("--save-dir", help="save raw jsonl and stderr per model/KV type here")
    ap.add_argument("--peak", type=float, default=960.0, help="peak memory bandwidth in GB/s (1e9)")
    ap.add_argument("extra", nargs="*", help="extra llama-bench args after --")
    args = ap.parse_args()

    rows: list[dict] = []
    if args.jsonl:
        for path in args.jsonl:
            with open(path) as f:
                rows += [json.loads(line) for line in f if line.startswith("{")]
    else:
        for model in args.model:
            for ctk in args.ctk.split(","):
                rows += run_bench(args, model, ctk.strip())

    kv_info = {}
    for model in args.model:
        try:
            kv_info[Path(model).name] = kv_elements_per_token(model)
        except Exception as e:  # noqa: BLE001 - report and continue without KV bytes
            print(f"could not read KV shape from {model}: {e}", file=sys.stderr)

    groups: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    sizes: dict[str, int] = {}
    for r in rows:
        if int(r.get("n_prompt", 0)) != 0 or int(r.get("n_gen", 0)) == 0:
            continue
        name = Path(r["model_filename"]).name
        sizes[name] = int(r["model_size"])
        ms_per_tok = 1000.0 / float(r["avg_ts"])
        groups[(name, r["type_k"], r["type_v"])].append((int(r["n_depth"]), ms_per_tok))

    if not groups:
        print("no token-generation results to fit", file=sys.stderr)
        return 1

    print(f"# Decode time vs context depth (peak {args.peak:.0f} GB/s)")
    print()
    for (name, tk, tv), pts in sorted(groups.items()):
        pts.sort()
        print(f"## {name}  K={tk} V={tv}")
        print()
        print("| depth | tok/s | ms/token |")
        print("|---:|---:|---:|")
        for d, ms in pts:
            print(f"| {d} | {1000 / ms:.2f} | {ms:.2f} |")
        print()
        if len(pts) < 2:
            print("(need at least 2 depths to fit)\n")
            continue
        a, b, r2 = fit(pts)
        size = sizes.get(name, 0)
        w_gbs = size / (a / 1000) / 1e9 if a > 0 else float("nan")
        print(f"- fixed part: **{a:.2f} ms/token** -> model bytes {size / 1e9:.2f} GB at "
              f"**{w_gbs:.0f} GB/s ({100 * w_gbs / args.peak:.0f}% of peak)**")
        slope_ms_per_1k = b * 1000
        line = f"- per-context part: **{slope_ms_per_1k:.3f} ms per 1k tokens** (fit r2 = {r2:.3f})"
        info = kv_info.get(name)
        if info and tk in BYTES_PER_ELEMENT and tv in BYTES_PER_ELEMENT and b > 0:
            k_el, v_el, n_attn = info
            kv_bytes_per_tok = k_el * BYTES_PER_ELEMENT[tk] + v_el * BYTES_PER_ELEMENT[tv]
            kv_gbs = kv_bytes_per_tok / (b / 1000) / 1e9
            line += (f" -> KV {kv_bytes_per_tok / 1024:.1f} KiB per token of context "
                     f"({n_attn} attention layers) read at **{kv_gbs:.0f} GB/s ({100 * kv_gbs / args.peak:.0f}% of peak)**")
            ideal = kv_bytes_per_tok * 1000 / (args.peak * 1e9) * 1000
            line += f"; at peak it would be {ideal:.3f} ms per 1k"
        print(line)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
