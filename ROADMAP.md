# Roadmap

This roadmap is written for anyone who finds this repo (including future me). It’s intentionally broad: the goal is to keep the project **useful, reproducible, and learnable**, while leaving room for ambitious upgrades over time.

## Current status (what works today)

- End-to-end **YOLOv2 INT16** inference on **KV260** with correct detections
- Model artifact generation: packed weights/bias + INT16/FP32 quantization artifacts (Q tables)
- HLS accelerator IP export (Vitis HLS) + Vivado BD build (GUI + batch/no-GUI)
- KV260 Linux userspace app + `xmutil` firmware packaging/deploy helpers
- A YAML-driven staged “one script to run it all” pipeline (`pipeline.yaml`, `scripts/run_pipeline.py`)

---

## North-star goals (choose what “success” means)

### Goal A — Real-time camera demo on KV260 (most rewarding next step)
Target outcomes:
- Live camera input (USB/UVC or MIPI via libcamera) → overlay boxes → display/encode
- Stable performance telemetry (FPS, latency breakdown)
- Minimal friction “one command demo”

### Goal B — Make it “PhD-proof”: reproducible, scripted, and CI-friendly (highest leverage)
Target outcomes:
- A staged pipeline of scripts that take you from **fresh machine** → **detections on KV260** (✅ initial version exists)
- Clear “known-good” versions + checks (Vivado/Vitis/KV260 image)
- Regression checks that protect against “it broke overnight”
- Release artifacts that let others run without rebuilding everything

### Goal C — Push IP throughput (most technical)
Target outcomes:
- Big speedup (e.g., 3–10×) from architectural HLS changes
- Better DDR efficiency, streaming, overlapping compute/transfer
- Optional offload of pre/post-processing hotspots

### Goal D — (Optional, later) Turn it into a mini “model-to-FPGA” framework (most ambitious)
Target outcomes:
- Accept a standard model format (typically ONNX) + calibration dataset
- Automatically produce: packed weights + quant/Q files + runtime graph + validation report
- Run multiple supported models without hand-editing code

Quick note: ONNX (Open Neural Network Exchange) is a common file format for neural networks that helps move models between frameworks (PyTorch/TF/etc.) and tooling. This project does *not* need ONNX to stay valuable, but ONNX matters if the long-term direction is “drop in a model and the toolchain handles it”.

---

## Next milestones (suggested order)

### 1) Camera + video I/O (Goal A)
Scope (start simple, then refine):
- USB/UVC camera via V4L2 (fastest path to a live demo)
- MIPI camera via libcamera (once the pipeline is stable)
- Video file inference (decode → infer → write annotated output)

Deliverables:
- A “camera mode” in `linux_app/` with a clear CLI (`--camera`, `--video`, `--no-display`, `--save`)
- Docs: “Live camera demo” section with one copy/paste command

### 2) Profiling + quick software wins (Goal B/C)
Before changing the IP, make time measurable:
- Timing breakdown (preprocess / DMA / per-layer / postprocess)
- Make debug dumps and heavy logs opt-in (already supported; keep improving ergonomics)

Deliverables:
- A repeatable benchmark command that outputs a concise timing report
- A small “top hotspots” list to drive optimization work

### 3) Reproducibility + CI + releases (Goal B)
Make it harder to accidentally break the “it just works” flow:
- A minimal regression check (one image + expected detections or expected region summary)
- CI that runs without FPGA hardware (lint/smoke checks + host builds + doc link checks)
- Clear release artifacts (what to download to run without rebuilding everything)

Deliverables:
- `scripts/validate_end_to_end.*` (host-side and/or board-side)
- GitHub Actions workflow(s) for docs + builds + smoke checks

### 4) Real HLS/IP throughput work (Goal C)
Once input/output paths and measurement are solid:
- Measure per-layer performance and DDR traffic
- Reduce store-and-forward to DDR (stream more, buffer/tiling, dataflow)
- Consider fusing ops where it clearly reduces memory traffic (Conv+BN+Act)

Deliverables:
- A perf report (before/after) for one or two concrete architectural upgrades

### 5) Optional: Model bundles + ONNX import (Goal D)
If I decide to push toward a more general framework:
- Define a versioned “model bundle” format (JSON metadata + binary blobs)
- Stabilize a small runtime API so swapping models doesn’t mean editing the KV260 app
- Add an ONNX importer + minimal compiler skeleton (start with the subset needed for YOLO-like graphs)

Deliverables:
- `bundle.json` spec + loader + compatibility checks
- A minimal ONNX-to-bundle path for at least one additional model

---

## Non-goals (to keep the scope sane)

- I’m not trying to compete with or replicate commercial stacks (Vitis AI / DPU). They’re the inspiration, not the target.
- If a feature adds a lot of complexity but doesn’t improve **reproducibility**, **performance**, or **usability**, it’s probably not worth it.

