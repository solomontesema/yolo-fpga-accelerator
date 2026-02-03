# YOLO2 Performance Report Tool

A lightweight CLI for tracking HLS/Vivado/KV260 performance metrics across optimization iterations.

**Key features:**
- Python 3 stdlib-only (no external dependencies for core functionality)
- Parses HLS `*_csynth.xml`, Vivado `.rpt` files, and KV260 inference logs
- Creates timestamped report bundles with JSON metrics and Markdown summaries
- Compare runs to see deltas in FPS, utilization, timing, etc.

## Quick Start

### 1. Initialize (one-time)

```bash
python3 scripts/yolo2_report.py init
```

This creates:
- `yolo2_report.local.json` — config file (edit to customize paths)
- `reports/` — directory for report bundles

### 2. Create a Report (local HLS parse)

```bash
# Parse existing HLS reports (no board needed)
python3 scripts/yolo2_report.py run \
  --label baseline \
  --hls-report-dir yolo2_int16/solution1/syn/report
```

Output:
```
reports/<timestamp>_baseline/
├── meta.json          # Git commit, timestamp, label, note
├── metrics.json       # All parsed metrics
├── summary.md         # Human-readable summary
├── hls/
│   ├── YOLO2_FPGA_csynth.xml  # Copy of top-level report
│   └── parsed_hls.json
├── vivado/            # (empty if not provided)
└── kv260/             # (empty if not provided)
```

### 3. List Reports

```bash
python3 scripts/yolo2_report.py list
```

Output:
```
Bundle                                        Label                Git             FPS        WNS (ns)
----------------------------------------------------------------------------------------------------
2026-01-07_14-55-59_baseline                  baseline             80ebaf34da5e*   N/A        N/A
2026-01-07_14-57-14_axi_test                  axi_test             80ebaf34da5e*   N/A        N/A
```

### 4. Compare Two Runs

```bash
python3 scripts/yolo2_report.py compare reports/<runA>/ reports/<runB>/
```

Output:
```
# Comparison: baseline vs axi_test

| Metric                    | baseline        | axi_test        | Delta                |
|---------------------------|-----------------|-----------------|----------------------|
| HLS DSP                   | 136             | 136             | +0 (+0.0%)           |
| HLS LUT                   | 74820           | 74820           | +0 (+0.0%)           |
| HLS Est. Clock (ns)       | 3.650           | 3.650           | +0.000               |
```

## Full Usage

### Create a Report with KV260 Log File

If you have a saved log from a KV260 run:

```bash
python3 scripts/yolo2_report.py run \
  --label perf_test \
  --note "Testing AXI width changes" \
  --hls-report-dir yolo2_int16/solution1/syn/report \
  --kv260-log /path/to/kv260_output.log
```

### Create a Report with Vivado Reports

```bash
python3 scripts/yolo2_report.py run \
  --label vivado_impl \
  --hls-report-dir yolo2_int16/solution1/syn/report \
  --vivado-report-dir vivado/yolov2_int16/yolov2_int16.runs/impl_1/
```

### Create a Report via SSH to KV260

```bash
python3 scripts/yolo2_report.py run \
  --label kv260_live \
  --hls-report-dir yolo2_int16/solution1/syn/report \
  --kv260-ssh \
  --kv260-cmd "./linux_app/start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg --max-frames 10"
```

**Auth options:**
- Recommended: passwordless SSH (key-based auth).
- If your KV260 requires a password, use one of:
  - `--kv260-password-prompt` (interactive prompt)
  - `--kv260-password-env YOLO2_KV260_PASSWORD` (reads from env var)

Example (prompt):
```bash
python3 scripts/yolo2_report.py run \
  --label kv260_live \
  --hls-report-dir yolo2_int16/solution1/syn/report \
  --kv260-ssh \
  --kv260-password-prompt \
  --kv260-cmd "./linux_app/start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg --max-frames 10"
```

Example (env var):
```bash
export KV260_SSH_PASSWORD='your_password_here'
python3 scripts/yolo2_report.py run \
  --label kv260_live \
  --hls-report-dir yolo2_int16/solution1/syn/report \
  --kv260-ssh \
  --kv260-password-env YOLO2_KV260_PASSWORD \
  --kv260-cmd "./linux_app/start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg --max-frames 10"
```

Edit `yolo2_report.local.json` to set `kv260.host`, `kv260.port`, etc.

## Config File

After `init`, edit `yolo2_report.local.json`:

```json
{
  "reports_dir": "reports",
  "hls_report_dir": "yolo2_int16/solution1/syn/report",
  "vivado_report_dir": "",
  "kv260": {
    "enabled": false,
    "host": "ubuntu@kria",
    "port": 22,
    "identity_file": "",
    "remote_cmd_template": "./linux_app/start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg --max-frames 10",
    "force_tty": true,
    "password_env": "KV260_SSH_PASSWORD",
    "timeout_s": 300
  }
}
```

CLI arguments override config file values.

## What Gets Parsed

### HLS (`*_csynth.xml`)

- Target clock period (ns) and estimated clock period (ns)
- Resource utilization: DSP, LUT, FF, BRAM_18K, URAM
- Available resources
- AXI port widths (DATA_BUS_IN, DATA_BUS_OUT, DATA_BUS1)

### Vivado Reports

- `report_timing_summary`: WNS/TNS/WHS/THS + target clock (period/frequency)
- `report_utilization`: LUT/FF/BRAM/DSP/URAM usage (used + available)
- `report_power`: Total/Dynamic/Static power + IP power for `YOLO2_FPGA_0` (if present)

### KV260 Logs

Parses `yolo2_linux` stdout for lines like:
- `Inference time: 1234.56 ms`
- `Frame 1 (infer 1) inference time: 1234.56 ms`

Computes: count, mean, median, p90, FPS

## Output Files

Each run creates a bundle in `reports/<timestamp>_<label>/`:

| File | Description |
|------|-------------|
| `meta.json` | Timestamp, git info, label, note, input paths |
| `metrics.json` | All parsed metrics (machine-readable) |
| `summary.md` | Human-readable Markdown summary |
| `hls/parsed_hls.json` | Detailed HLS parse results |
| `hls/*.xml` | Copy of input HLS reports |
| `vivado/parsed_vivado.json` | Detailed Vivado parse results |
| `vivado/*.rpt` | Copies of the timing/utilization/power reports used |
| `kv260/stdout.log` | Raw KV260 output |
| `kv260/parsed_kv260.json` | Parsed inference timings |

## Workflow Example

Track performance across HLS optimizations:

```bash
# 1. Baseline measurement
python3 scripts/yolo2_report.py run --label baseline \
  --hls-report-dir yolo2_int16/solution1/syn/report

# 2. Make HLS changes, rebuild
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl

# 3. Capture new metrics
python3 scripts/yolo2_report.py run --label axi_256_outstanding_16 \
  --note "Widened AXI to 256-bit, outstanding=16" \
  --hls-report-dir yolo2_int16/solution1/syn/report

# 4. Compare
python3 scripts/yolo2_report.py compare \
  reports/*baseline/ reports/*axi_256*/

# 5. List all runs
python3 scripts/yolo2_report.py list
```

## Design Notes

- **Robust parsing**: Missing fields or files are handled gracefully; the tool produces partial results rather than crashing.
- **Self-contained bundles**: Each bundle stores copies of input files, so you can understand results later without re-running.
- **Git-aware**: Automatically captures commit hash and dirty state.
- **No heavy dependencies**: Uses only Python stdlib (xml.etree, json, subprocess, etc.).

## Troubleshooting

### "No inference timing lines found"

The KV260 log parser looks for patterns like `inference time: 123.45 ms`. Make sure:
- `YOLO2_VERBOSE >= 1` when running `yolo2_linux`
- The log contains actual inference runs (not just setup messages)

### SSH times out

- Check that the KV260 is reachable: `ssh ubuntu@kria`
- Ensure passwordless auth is configured
- Increase the timeout in the script if inference takes >5 minutes

### HLS reports not parsed

- Ensure the path points to the directory containing `YOLO2_FPGA_csynth.xml`
- Check that the XML files are valid (not truncated or corrupted)
