#!/usr/bin/env python3
"""FlashAttention bandwidth report for the Qwen3.5/3.8-27B full-attention shape.

Runs (or reads the saved output of) `test-backend-ops perf -o FLASH_ATTN_EXT` for the
D=256 / 4 KV heads / gqa 6 cases and reports, per KV type, KV length and Q batch:

  - time per call
  - effective bandwidth = minimum bytes (K + V read once) / time, and % of peak
  - verify cost ratio t(nb) / t(nb=1)  (what a speculative verify step pays)
  - the kernel the CUDA/HIP backend picked, from GGML_CUDA_FATTN_LOG=1

A kernel reading each KV head once should approach the peak; a result far below it
at nb=1 means the KV cache is re-read (e.g. once per query head) or dequant-bound.

Examples:
  scripts/rx7900xtx/fattn_bw.py --bin build/bin/test-backend-ops --backend ROCm0 --save fattn.sql
  scripts/rx7900xtx/fattn_bw.py --sql fattn.sql --log fattn.log
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict

# bytes per element of the KV cache types test-backend-ops can use
BYTES_PER_ELEMENT = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
}

DEFAULT_FILTER = r"hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=(16384|65536|113408|262144),nb=[1-8],"

LOG_RE = re.compile(
    r"fattn: kernel=(?P<kernel>\w+) D=(?P<dk>\d+)/(?P<dv>\d+) n_q=(?P<nq>\d+) n_head=\d+ n_head_kv=\d+ "
    r"gqa=(?P<gqa>\d+) K=(?P<tk>\w+) V=(?P<tv>\w+) n_kv=(?P<nkv>\d+) .*?"
    r"f16_conv_K=(?P<ck>\d) f16_conv_V=(?P<cv>\d)"
)


def parse_params(s: str) -> dict[str, str]:
    """Split 'a=1,b=[6,1],c=q4_0' into a dict, keeping bracketed values intact."""
    out: dict[str, str] = {}
    depth = 0
    cur = ""
    for ch in s + ",":
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            if "=" in cur:
                k, v = cur.split("=", 1)
                out[k.strip()] = v.strip()
            cur = ""
        else:
            cur += ch
    return out


def load_sql(text: str) -> list[dict]:
    db = sqlite3.connect(":memory:")
    db.executescript(text)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT op_params, time_us, n_runs, backend_name FROM test_backend_ops "
        "WHERE op_name = 'FLASH_ATTN_EXT' AND test_mode = 'perf' AND supported = 1 AND passed = 1"
    ).fetchall()
    return [dict(r) for r in rows]


def kv_bucket(n: int) -> int:
    b = 0
    while (1 << (b + 1)) <= n:
        b += 1
    return b


def parse_log(text: str) -> dict[tuple, str]:
    """(n_q, type_K, type_V, n_kv bucket) -> 'KERNEL' or 'KERNEL+f16conv'."""
    kernels: dict[tuple, str] = {}
    for m in LOG_RE.finditer(text):
        name = m["kernel"]
        if m["ck"] == "1" or m["cv"] == "1":
            name += "+f16conv"
        kernels[(int(m["nq"]), m["tk"], m["tv"], kv_bucket(int(m["nkv"])))] = name
    return kernels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/bin/test-backend-ops", help="test-backend-ops binary")
    ap.add_argument("--backend", default="ROCm0", help="backend name passed to -b (ROCm0 for HIP builds)")
    ap.add_argument("--filter", default=DEFAULT_FILTER, help="regex passed to -p")
    ap.add_argument("--sql", help="parse this saved SQL output instead of running the benchmark")
    ap.add_argument("--log", help="stderr log with GGML_CUDA_FATTN_LOG lines (when using --sql)")
    ap.add_argument("--save", help="save the raw SQL output here (stderr goes to <save>.log)")
    ap.add_argument("--peak", type=float, default=960.0, help="peak memory bandwidth in GB/s (1e9)")
    ap.add_argument("--attn-layers", type=int, default=16, help="full-attention layers per token")
    args = ap.parse_args()

    if args.sql:
        with open(args.sql) as f:
            sql_text = f.read()
        log_text = ""
        if args.log:
            with open(args.log) as f:
                log_text = f.read()
    else:
        cmd = [args.bin, "perf", "-o", "FLASH_ATTN_EXT", "-b", args.backend, "-p", args.filter, "--output", "sql"]
        env = dict(os.environ, GGML_CUDA_FATTN_LOG="1")
        print("running:", " ".join(cmd), file=sys.stderr)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sql_text, log_text = proc.stdout, proc.stderr
        if args.save:
            with open(args.save, "w") as f:
                f.write(sql_text)
            with open(args.save + ".log", "w") as f:
                f.write(log_text)
        if proc.returncode != 0:
            print(log_text[-4000:], file=sys.stderr)
            print(f"test-backend-ops exited with {proc.returncode}", file=sys.stderr)
            return 1

    rows = load_sql(sql_text)
    if not rows:
        print("no FLASH_ATTN_EXT perf results found (wrong backend name or filter?)", file=sys.stderr)
        return 1
    kernels = parse_log(log_text)

    # (type_K, type_V, kv) -> {nb: (time_us, bytes)}
    table: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
    backend = rows[0]["backend_name"]
    for r in rows:
        p = parse_params(r["op_params"])
        hsk, hsv = int(p["hsk"]), int(p["hsv"])
        nh_kv = int(p["nh"])
        kv, nb = int(p["kv"]), int(p["nb"])
        tk, tv = p["type_K"], p["type_V"]
        if tk not in BYTES_PER_ELEMENT or tv not in BYTES_PER_ELEMENT:
            continue
        kv_bytes = nh_kv * kv * (hsk * BYTES_PER_ELEMENT[tk] + hsv * BYTES_PER_ELEMENT[tv])
        mask_bytes = kv * nb * 2  # f16 mask, one row per query
        table[(tk, tv, kv)][nb] = (float(r["time_us"]), kv_bytes + mask_bytes)

    print(f"# FLASH_ATTN_EXT bandwidth on {backend} (peak {args.peak:.0f} GB/s)")
    print()
    print("Effective GB/s counts each K/V byte once (the minimum any kernel must read).")
    print("verify x = time(nb) / time(nb=1): cost of a speculative verify step relative to one token.")
    print()

    all_nb = sorted({nb for d in table.values() for nb in d})
    for key in sorted(table, key=lambda k: (k[0], k[1], k[2])):
        tk, tv, kv = key
        d = table[key]
        print(f"## K={tk} V={tv} n_kv={kv}  (K+V = {next(iter(d.values()))[1] / 1e6:.1f} MB per layer)")
        print()
        print("| nb | us/call | GB/s | % peak | verify x | kernel |")
        print("|---:|---:|---:|---:|---:|---|")
        t1 = d.get(1, (None, None))[0]
        for nb in all_nb:
            if nb not in d:
                continue
            t, b = d[nb]
            gbs = b / (t * 1e-6) / 1e9
            ratio = f"{t / t1:.2f}" if t1 else "-"
            kern = kernels.get((nb, tk, tv, kv_bucket(kv)), "?")
            print(f"| {nb} | {t:.1f} | {gbs:.0f} | {100 * gbs / args.peak:.0f}% | {ratio} | {kern} |")
        print()

    # per-token attention cost at nb=1, per KV type and length
    print(f"## Decode attention cost per token ({args.attn_layers} full-attention layers, nb=1)")
    print()
    kvs = sorted({k[2] for k in table})
    types = sorted({(k[0], k[1]) for k in table})
    print("| n_kv | " + " | ".join(f"{a}/{b} ms" for a, b in types) + " | at peak (q4_0) ms |")
    print("|---:|" + "---:|" * (len(types) + 1))
    for kv in kvs:
        cells = []
        for tk, tv in types:
            d = table.get((tk, tv, kv), {})
            cells.append(f"{d[1][0] * args.attn_layers / 1000:.2f}" if 1 in d else "-")
        q4 = table.get(("q4_0", "q4_0", kv), {})
        ideal = f"{q4[1][1] * args.attn_layers / (args.peak * 1e9) * 1000:.2f}" if 1 in q4 else "-"
        print(f"| {kv} | " + " | ".join(cells) + f" | {ideal} |")
    print()
    if not kernels:
        print("(no kernel log lines: the backend is not CUDA/HIP, the build predates GGML_CUDA_FATTN_LOG, "
              "or --sql was used without --log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
