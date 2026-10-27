# Metal decode profiling

Aljabr's Metal decode path supports opt-in host/GPU timing diagnostics. This
workflow captures comparable runs and summarizes the emitted profile records;
it does not itself benchmark or change the inference path.

## Capture comparable runs

Use the same model file, prompt, generation length, build, power mode, and
runtime options for each run. Run on the same Apple Silicon machine and avoid
other GPU-heavy workloads. Collect several repetitions per mode.

```bash
export ALJABR_METAL_DECODE_PROFILE=1
unset ALJABR_METAL_DISABLE_FUSED_TOKEN_EMBEDDING

# Replace this with your normal, fixed Aljabr benchmark command.
./your-existing-benchmark-command 2>&1 | tee metal-fused.log

export ALJABR_METAL_DISABLE_FUSED_TOKEN_EMBEDDING=1
./your-existing-benchmark-command 2>&1 | tee metal-fallback.log
```

The second mode is a diagnostic reference path, not necessarily the faster or
more correct path. Compare generated token IDs and/or logits as well as timing;
timing alone is not a parity test.

## Summarize profile records

The script reads `[METAL DECODE PROFILE]` lines, groups records by `path`
(`token` or `vector`), and reports sample count plus median and nearest-rank
p95 for wall, host-through-encoding, submit/wait, and command-buffer GPU time.
It uses only Python's standard library.

```bash
python3 scripts/metal-decode-profile-summary.py metal-fused.log metal-fallback.log
python3 scripts/metal-decode-profile-summary.py --warmup 2 --format csv metal-fused.log
python3 scripts/metal-decode-profile-summary.py --format json metal-fused.log
```

Use `--warmup N` to drop the first N records **per path**. Use separate log
files for separate modes when possible; path grouping does not distinguish
fused and fallback modes if they both report the same path label. For clean
comparisons, summarize each mode separately.

## Interpret the numbers

- `wall_ms` is host-observed time from before decode scratch/setup work until
  command-buffer completion.
- `encode_ms` is host time from the same start point through command encoding;
  it includes setup/allocation work and is **not** pure GPU time.
- `submit_wait_ms` is host time from just before commit until completion.
- `gpu_ms` is Metal's command-buffer GPU duration when reported by Metal; it
  may be zero when timestamps are unavailable.
- The command-buffer GPU duration aggregates its work. It does not identify
  per-kernel or per-layer GPU cost. Use a Metal GPU capture / Instruments when
  stage-level GPU attribution is needed.

Do not compare a single sample as a performance result. Look at repeated-run
medians and p95, preserve raw logs, and confirm output parity before drawing a
conclusion. Profiling is disabled by default; unset
`ALJABR_METAL_DECODE_PROFILE` for ordinary runs.
