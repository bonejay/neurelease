#!/usr/bin/env python3
"""Time the Python binding: single-name loop, parse_batch, and what the dict view adds.

    python tools/bench_python.py build/benchmark_names.txt [repeat] [threads]

Every figure is a per-name mean over the whole run, printed twice (two passes) so the machine's
own variance is visible next to the numbers. Compare cells from ONE run only.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))
from neurelease import Parser  # noqa: E402


def main() -> int:
    names_file = Path(sys.argv[1] if len(sys.argv) > 1 else "build/benchmark_names.txt")
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else None
    names = [line.rstrip("\n") for line in names_file.open(encoding="utf-8") if line.strip()] * repeat
    print(f"{len(names):,} names, threads={threads or 'default'}, cpu={os.cpu_count()}")
    with Parser(threads=threads) as parser:
        parser.parse_batch(names[:64])                 # warm the kernels and the encoder cache
        for _pass in range(2):
            rows = []
            t = time.perf_counter(); [parser.parse(n) for n in names]; rows.append(("single, origins on", time.perf_counter() - t))
            t = time.perf_counter(); [parser.parse(n, origins=False) for n in names]; rows.append(("single, origins off", time.perf_counter() - t))
            t = time.perf_counter(); parsed = parser.parse_batch(names); rows.append(("batch, origins on", time.perf_counter() - t))
            t = time.perf_counter(); parser.parse_batch(names, origins=False); rows.append(("batch, origins off", time.perf_counter() - t))
            t = time.perf_counter(); [r.to_dict() for r in parsed]; rows.append(("to_dict() on batch results", time.perf_counter() - t))
            for label, seconds in rows:
                print(f"  {label:28} {1e6 * seconds / len(names):8.0f} us/name")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
