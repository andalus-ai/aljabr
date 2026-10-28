#!/usr/bin/env python3
"""Summarize Aljabr's opt-in [METAL DECODE PROFILE] log lines.

Reads one or more log files, or stdin when no files are given. Uses only the
Python standard library. Rows are grouped by token/vector path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, TextIO

PROFILE_RE = re.compile(r"\[METAL DECODE PROFILE\]\s+(.*)")
FIELD_RE = re.compile(r"([a-z_]+)=([^\s]+)")
METRICS = ("wall_ms", "encode_ms", "submit_wait_ms", "gpu_ms")


def parse_lines(lines: Iterable[str]) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    malformed = 0
    for line in lines:
        match = PROFILE_RE.search(line)
        if not match:
            continue
        fields = dict(FIELD_RE.findall(match.group(1)))
        try:
            row: dict[str, object] = {
                "pos": int(fields["pos"]),
                "layers": int(fields["layers"]),
                "path": fields["path"],
            }
            for metric in METRICS:
                value = float(fields[metric])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {metric}")
                row[metric] = value
        except (KeyError, ValueError):
            malformed += 1
            continue
        rows.append(row)
    return rows, malformed


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, with p in [0, 100]."""
    ordered = sorted(values)
    index = max(0, math.ceil((p / 100.0) * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: list[dict[str, object]], warmup: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["path"])].append(row)

    summaries: list[dict[str, object]] = []
    for path in sorted(grouped):
        samples = grouped[path][warmup:]
        if not samples:
            continue
        result: dict[str, object] = {
            "path": path,
            "samples": len(samples),
            "layers": samples[-1]["layers"],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in samples]
            result[f"{metric}_median"] = statistics.median(values)
            result[f"{metric}_p95"] = percentile(values, 95)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        summaries.append(result)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="log files; defaults to stdin")
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="discard this many initial samples per path (default: 0)",
    )
    parser.add_argument(
        "--format", choices=("table", "json", "csv"), default="table",
        help="output format (default: table)",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    rows: list[dict[str, object]] = []
    malformed = 0
    if args.files:
        for filename in args.files:
            try:
                with Path(filename).open("r", encoding="utf-8", errors="replace") as stream:
                    parsed, bad = parse_lines(stream)
            except OSError as exc:
                print(f"error: cannot read {filename}: {exc}", file=sys.stderr)
                return 2
            rows.extend(parsed)
            malformed += bad
    else:
        rows, malformed = parse_lines(sys.stdin)

    if malformed:
        print(f"warning: skipped {malformed} malformed profile line(s)", file=sys.stderr)
    if not rows:
        print("error: no valid [METAL DECODE PROFILE] records found", file=sys.stderr)
        return 1

    summaries = summarize(rows, args.warmup)
    if not summaries:
        print("error: no samples remain after warmup filtering", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(summaries, indent=2))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    else:
        columns = ["path", "samples"]
        for metric in METRICS:
            columns.extend([f"{metric}_median", f"{metric}_p95"])
        widths = {
            col: max(len(col), *(len(str(row[col])) if col in ("path", "samples")
                                 else f"{float(row[col]):.3f}".__len__()
                                 for row in summaries))
            for col in columns
        }
        print("  ".join(col.ljust(widths[col]) for col in columns))
        for row in summaries:
            cells = []
            for col in columns:
                value = row[col]
                rendered = str(value) if col in ("path", "samples") else f"{float(value):.3f}"
                cells.append(rendered.ljust(widths[col]))
            print("  ".join(cells))
    print(
        f"Parsed {len(rows)} samples; summarize by path with warmup={args.warmup}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
