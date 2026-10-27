#!/usr/bin/env python3
"""Repeatable end-to-end AQLI benchmark runner for local Aljabr/Metal work.

This measures command wall time, not pure GPU decode time. Keep the prompt,
model, max token count, runtime flags, and machine state identical when
comparing revisions. Raw stdout/stderr are retained for later inspection.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="aqli", help="AQLI executable (default: aqli)")
    parser.add_argument("--model", required=True, help="Model ID or local GGUF path")
    parser.add_argument("--prompt", default="who are you", help="Fixed benchmark prompt")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs (default: 1)")
    parser.add_argument("--out-dir", default="build/metal-benchmark", help="Output directory")
    parser.add_argument("--no-java", action="store_true", help="Do not pass --java")
    parser.add_argument("--disable-metal", action="store_true",
                        help="Set aljabr.metal.disable=true for a CPU/reference comparison")
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0 or args.max_tokens < 1:
        parser.error("--runs and --max-tokens must be positive; --warmup cannot be negative")

    out_dir = Path(args.out_dir).expanduser().resolve()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "runs.csv"
    command = [args.cli, "run", "--model", args.model, "--no-banner",
               "--prompt", args.prompt, "--max-tokens", str(args.max_tokens)]
    if not args.no_java:
        command.append("--java")

    env = os.environ.copy()
    if args.disable_metal:
        env["JAVA_TOOL_OPTIONS"] = (env.get("JAVA_TOOL_OPTIONS", "") + " -Daljabr.metal.disable=true").strip()

    rows = []
    total = args.warmup + args.runs
    print(f"Command: {shlex.join(command)}")
    print(f"Warm-ups: {args.warmup}; measured runs: {args.runs}")
    print(f"Logs and CSV: {out_dir}")

    for index in range(total):
        label = "warmup" if index < args.warmup else "run"
        number = index + 1 if label == "warmup" else index - args.warmup + 1
        log_path = logs_dir / f"{label}-{number:02d}.log"
        start = time.perf_counter()
        completed = subprocess.run(command, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        log_path.write_text(completed.stdout, encoding="utf-8")
        print(f"{label} {number}/{args.warmup if label == 'warmup' else args.runs}: "
              f"{elapsed:.3f}s (exit {completed.returncode}) -> {log_path}")
        if completed.returncode != 0:
            print(completed.stdout[-6000:], file=sys.stderr)
            print(f"Benchmark stopped: command failed; inspect {log_path}", file=sys.stderr)
            return completed.returncode or 1
        if label == "run":
            # Preserve any reported rate as diagnostic text only; timing scopes vary by CLI.
            rates = re.findall(r"(?i)(?:tok(?:en)?s?/s|tokens per second)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
                               completed.stdout)
            rows.append({
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "run": number,
                "elapsed_seconds": f"{elapsed:.6f}",
                "reported_tokens_per_second": rates[-1] if rates else "",
                "exit_code": completed.returncode,
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "command": shlex.join(command),
                "machine": platform.machine(),
                "macOS_or_os": platform.platform(),
                "python": platform.python_version(),
                "metal_disabled": args.disable_metal,
                "log_file": str(log_path),
            })

    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    elapsed_values = sorted(float(row["elapsed_seconds"]) for row in rows)
    median = elapsed_values[len(elapsed_values) // 2] if len(elapsed_values) % 2 else (
        elapsed_values[len(elapsed_values) // 2 - 1] + elapsed_values[len(elapsed_values) // 2]
    ) / 2
    print(f"Measured wall time: min={elapsed_values[0]:.3f}s "
          f"median={median:.3f}s max={elapsed_values[-1]:.3f}s")
    print(f"CSV: {csv_path}")
    print("Note: elapsed time includes process startup/model loading; do not label it decode latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
