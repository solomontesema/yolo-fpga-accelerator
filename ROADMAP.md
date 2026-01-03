# Roadmap: From “working demo” to a reusable FPGA Object-Detection Accelerator Framework

This project is already a complete end-to-end pipeline for **one real model (YOLOv2)** on **real hardware (KV260)**:
- Model weights/bias extraction + Q-file generation (INT16/FP32) from `.h5` or Darknet `.weights`
- A model-agnostic HLS accelerator (tested with YOLOv2) with C-sim/C-synth/C-impl/C-cosim
- A Vivado BD flow (GUI + batch/no-GUI) producing bitstream + XSA
- A Linux userspace application on KV260 that loads the FPGA design and runs inference end-to-end
- Working documentation that lets you reproduce it from scratch

This is **absolutely “ground-work”** in the sense that the hardest integration points (toolchain → RTL → bitstream → Linux runtime → correct outputs) are done.  
What’s *not yet* built (compared to Vitis AI / DPU ecosystems) is the “productized compiler + runtime + operator library + model coverage” layer.

The plan below is intentionally broad: pick a goal (real-time camera demo vs. framework/compiler vs. research-grade performance) and follow the relevant track.

---

## 1) North-star goals (choose what “success” means)

### Goal A — Real-time camera demo on KV260 (most rewarding next step)
Target outcomes:
- Live camera input (USB/UVC or MIPI via libcamera) → overlay boxes → display/encode
- Stable performance telemetry (FPS, latency breakdown)
- Minimal friction “one command demo”

### Goal B — Make it “PhD-proof”: reproducible, scripted, and CI-friendly (highest leverage)
Target outcomes:
- A staged pipeline of scripts that take you from **fresh machine** → **detections on KV260**
- Clear “known-good” versions + checks (Vivado/Vitis/KV260 image)
- Regression checks that protect against “it broke overnight”
- Release artifacts that let others run without rebuilding everything

### Goal C — Push IP throughput (most technical)
Target outcomes:
- Big speedup (e.g., 3–10×) from architectural HLS changes
- Better DDR efficiency, streaming, overlapping compute/transfer
- Optional offload of pre/post-processing hotspots

### Goal D — (Optional, later) Turn it into a mini “DPU-like” framework (most ambitious)
Target outcomes:
- Accept a standard model format (typically ONNX) + calibration dataset
- Automatically produce: packed weights + quant/Q files + runtime graph + validation report
- Run multiple supported models without hand-editing code

You can do all three, but it’s much easier if you decide the *next 2–4 weeks* focus.

### Quick note: what ONNX is (so “Goal D” isn’t mysterious)
ONNX (Open Neural Network Exchange) is a common file format for neural networks that lets you move models between frameworks (PyTorch/TF/etc.) and tools.  
You do *not* need ONNX to keep this project valuable; it mainly matters if you want “drop in a new model and the toolchain handles it”.

---

## 2) Reality check: where you stand vs. a commercial “stack”

What Vitis AI / DPU stacks typically have:
- **Front-end**: model import (ONNX/TF), graph optimizations, operator fusions
- **Quantization toolchain**: calibration, per-channel/per-layer scales, accuracy reports
- **Compiler**: maps graph → hardware instructions/tiles/schedule; generates runtime artifacts
- **Runtime**: standard APIs, batching, async execution, profiling hooks, multi-model management
- **Operator library**: many ops supported, validated across models
- **Deployment packaging**: versioned firmware/images, reproducible installs, CI, release artifacts

What you already have (strong!):
- A working “compiler-like” *subset* for one family (YOLOv2) that produces the correct low-level artifacts
- A working accelerator + Linux integration path
- A documented, reproducible flow

What to build next (to approach “framework” status):
- A **model graph IR** + importer (preferably ONNX)
- A **calibration + quantization report** pipeline
- A **graph-to-accelerator mapping layer** (even if simplistic at first)
- A **runtime API** that is stable across models (not YOLOv2-specific)

---

## 3) Phase 0 (1–3 days): “Hardening” before expanding scope

This pays off immediately and makes every next step faster.

### 0.1 Add “golden validation” and regression checks
- Define a minimal set of test inputs (1–3 images) + expected outputs (CPU reference).
- Produce a consistent “golden artifacts” bundle:
  - packed weights
  - Q files
  - one layer output dump (or final region output)
  - post-processed detections (JSON)
- Add scripts to compare:
  - numeric diffs for intermediate dumps (tolerances)
  - exact diffs for metadata (tensor shapes, layer ordering)

Deliverable:
- `scripts/validate_end_to_end.sh` (or similar) that returns non-zero on mismatch.

### 0.2 Add stable performance instrumentation
- Separate *wall time* into:
  - capture/decode
  - preprocess
  - DMA transfer(s)
  - accelerator compute time (per layer if possible)
  - postprocess (NMS + decoding)
  - overlay/encode
- Ensure measurements are consistent and logged in one format (CSV/JSON).

Deliverable:
- A single “perf report” output (CSV/JSON) per run.

### 0.3 Make “verbosity” and “dump outputs” orthogonal
- Independent flags:
  - `--verbose N` (printing)
  - `--dump raw|proc|none` (files)
  - `--profile` (timers)
- Avoid tying debug dumps to verbose levels so you can profile without huge I/O overhead.

---

## 4) Track A: Camera pipeline on KV260 (high-impact next step)

### A.1 Decide camera stack
Recommended options (in order):
1) **UVC/USB camera + V4L2** (fast to implement, works everywhere)
2) **libcamera** (better long-term, MIPI support on platforms that expose it)
3) **GStreamer pipeline** (great for production pipelines; more dependencies)

Note on **KV260 MIPI**:
- MIPI CSI-2 usually needs a capture pipeline in the FPGA design (e.g., CSI-2 Rx + DMA path) *or* a known-good reference overlay that exposes `/dev/video*`.
- A practical approach is: get the full *software pipeline* working first with USB/V4L2, then integrate MIPI once you decide how you want to handle the video capture IP and device tree.

### A.2 Build a zero-copy-ish pipeline (as much as practical)
Target architecture:
1) Capture (V4L2 DMABUF if available)
2) Convert/resize (NEON-optimized or OpenCV if acceptable)
3) Quantize input (INT16) efficiently
4) Inference (accelerator)
5) Postprocess (decode + NMS)
6) Render (DRM/KMS, SDL2, or save/stream)

Implementation tips:
- Use **double/triple buffering** so capture and inference overlap.
- Keep tensors in a single persistent buffer pool (avoid malloc/free per frame).
- Move expensive conversions off the critical path (pre-allocate, reuse).

### A.3 UX / CLI for demo mode
Add a “demo” mode that is hard to break:
- `--camera /dev/video0`
- `--width/--height/--fps`
- `--display` / `--no-display`
- `--save-video out.mp4` (optional)
- `--thresh`, `--nms`, `--topk`

Deliverables:
- `linux_app/start_camera_yolo.sh` (or similar)
- A README section: “Live camera demo”

### A.4 Performance targets for camera mode
Pick two concrete targets to guide optimization:
- **Latency target**: e.g. < 300 ms end-to-end (capture→boxes)
- **Throughput target**: e.g. ≥ 5 FPS at 640×480 or ≥ 2 FPS at 416×416

Even if you don’t hit them initially, they force measurable progress.

---

## 5) Track B: Speedups without changing the HLS IP (software/runtime wins)

Even if the real gains are in HLS, there are still meaningful improvements:

### B.1 Reduce I/O overhead
- Make dumps optional and buffered.
- Avoid printing inside tight loops; aggregate logs per layer/frame.
- Use binary dumps (float32) instead of text when dumping large tensors.

### B.2 Pipeline host work
- Overlap CPU preprocessing of frame N+1 while FPGA runs frame N.
- Overlap post-processing of frame N while FPGA runs frame N+1.

### B.3 Remove unnecessary conversions
- Keep data in accelerator-friendly format as long as possible.
- If post-processing needs floats, convert only the region output (already done) and avoid extra intermediate conversions.

### B.4 Use vectorization (NEON) for hotspots
Likely hotspots:
- image resize/letterbox
- input quantization
- activation decoding / sigmoid/exp in postprocess
- NMS loops

Deliverable:
- A “top-5 hotspots” report from profiling and a clear optimization ticket list.

---

## 6) Track C: HLS IP throughput roadmap (where the big wins are)

### C.1 First: measure where time is actually spent
If the IP is “model-agnostic”, some layers may dominate:
- Conv layers with large feature maps
- Reorg/route layers if memory-bound
- Any layer that forces full round-trips to DDR

Action items:
- Add cycle counters / timestamps inside the IP (AXI-lite readable regs).
- Correlate per-layer runtimes with tensor sizes and bandwidth.

### C.2 Stream more, store less (DDR is the enemy)
Typical accelerators win by:
- Streaming feature maps through the pipeline (AXI-Stream where possible)
- Keeping partial sums / tiles on-chip (BRAM/URAM)
- Fusing adjacent ops to reduce memory traffic (Conv+BN+LeakyReLU)

Concrete upgrades:
- **Tiling**: process convolution in tiles that fit on chip.
- **Weight caching**: reuse weights efficiently; consider URAM for frequently used blocks.
- **Dataflow**: use `#pragma HLS DATAFLOW` and ping-pong buffers.

### C.3 Improve arithmetic and precision strategy
Options:
- Keep INT16 but use per-channel scales for accuracy.
- Move to INT8 (bigger speedups, harder quantization/accuracy work).
- Mixed precision: INT8 activations + INT16 accumulators, etc.

Deliverable:
- A documented quantization strategy and accuracy vs speed tradeoff chart.

### C.4 Multi-kernel / multi-CU scaling (if fabric allows)
- Instantiate multiple compute units for parallel layer execution or multi-stream.
- Or split “fast path conv” vs “misc ops” into separate kernels.

### C.5 Consider offloading parts of post-processing
If postprocess becomes the bottleneck after speeding up the IP:
- Offload sigmoid/exp and partial decode to FPGA
- Offload NMS partially (grid-level filtering / topK), leave final NMS on CPU

---

## 7) Track D: From “YOLOv2 solution” to a general framework (compiler + runtime)

This is the path toward “my own mini Vitis AI”.

### D.1 Choose a standard interchange format: ONNX
Goal:
- Accept ONNX models and parse them into a graph IR.

Deliverables:
- `tools/import_onnx.py` (or C++ importer) that outputs:
  - graph (nodes, tensors, shapes)
  - constants (weights)
  - operator attributes

### D.2 Define a graph IR and a “hardware contract”
You need a clean boundary:
- Graph IR: nodes, edges, tensor shapes, quant parameters
- Hardware contract: the op subset the accelerator supports + constraints
  - supported kernel sizes/strides/padding
  - supported activations
  - supported tensor layouts
  - supported quant scheme

Deliverables:
- `docs/hardware_contract.md`
- `docs/graph_ir.md`

### D.3 Implement a minimal “compiler” (even if naive)
Start with a restricted subset:
- Conv2D + BN + LeakyReLU
- Maxpool
- Reorg/Concat (if needed)
- YOLO region head decode on CPU

Compiler steps:
1) Shape inference
2) Quant parameter assignment
3) Memory planning (tensor reuse, lifetimes)
4) Weight packing
5) Emit runtime config (layer list + addresses + Q values)

Deliverable:
- A single command that takes ONNX + calibration set and outputs a runnable bundle for KV260.

### D.4 Runtime API stabilization
Create a small stable C API around inference:
- `accel_init()`, `accel_load_model(bundle)`, `accel_run(input, output)`, `accel_shutdown()`
- Separate:
  - “model bundle” (weights/Q/graph)
  - “device runtime” (DMA buffers, FPGA control)

Deliverable:
- `linux_app/src/libaccel/` (or similar) with headers, examples, and ABI stability goals.

### D.5 Model bundles, versioning, reproducibility
A “bundle” should be self-describing:
- model name/version
- input size + preprocessing
- quant scheme
- operator set version
- checksum of weights
- compatible IP version

Deliverable:
- `bundle.json` + binary blobs + a loader that validates compatibility.

---

## 8) Track E: Expand model support (after the compiler skeleton exists)

### E.1 Add YOLO variants
Recommended order:
1) YOLOv3-tiny (good stepping stone)
2) YOLOv4-tiny / YOLOv5n (depending on ops)
3) YOLOv5s / YOLOv8n (more ops, more postproc variations)

Each new model should be a “compiler test”, not a manual integration.

### E.2 Add non-YOLO detectors (optional)
SSD MobileNet / RetinaNet introduce different operator mixes.
Do this only if your operator coverage and compiler flow are already stable.

---

## 9) Product-quality improvements (open-source friendliness)

### 9.0 Staged “from scratch → detections” scripts (this matches your stated goal)
Create a top-level staged flow that’s easy to follow and easy to automate. Example stages:
1) **Host tooling sanity**: check Vivado/Vitis versions, board files, licenses, PATH
2) **Build HLS IP repo**: run the `vitis/` HLS TCL flow and verify `component.xml` exists
3) **Build Vivado bitstream/XSA**: run `vivado/build_from_bd.sh`
4) **Package KV260 firmware**: convert `.bit` → `.bit.bin`, generate/copy `.dtsi`/`.dtbo` (using XSCT flow you already validated)
5) **Deploy to KV260**: `scp` the firmware + app binaries
6) **Load overlay**: `xmutil loadapp ...`
7) **Run demo**: inference on image(s) or camera, plus optional `--profile` and dumps

Guidelines:
- Every stage should be restartable and idempotent.
- Every stage should emit a clear artifact path on success.
- Prefer a single “driver script” that calls stage scripts and supports `--from`, `--to`, and `--dry-run`.

### 9.1 Reproducible environments
- Provide a “known-good toolchain matrix”:
  - Vivado/Vitis versions
  - Ubuntu version on host + KV260
  - required packages
- Optional: Docker/container recipes for host-side tooling.

### 9.2 CI that works without FPGA hardware
You can’t run HW in GitHub Actions easily, but you can:
- Unit test weight extraction + packing
- Unit test shape inference + compiler outputs
- Run CPU reference inference for a tiny model
- Lint/format checks

Also add “CI for the *docs pipeline*”:
- Markdown link checker (avoid broken paths as the repo evolves)
- Script smoke checks (`bash -n`, `shellcheck` if you want)
- Build the Linux userspace app (cross or native) if toolchain is available

### 9.3 Releases and artifacts
Define what a “release” means:
- firmware bundle (bit.bin + dtbo + metadata)
- example model bundle
- prebuilt linux_app binary (optional)

---

## 10) Suggested “next 30 days” milestone plan (practical and motivating)

Pick one of these “Month goals”:

### Month Goal 1: Live camera demo
- Week 1: V4L2 capture + preprocess + run existing inference per frame
- Week 2: Double-buffering + stable perf report + overlay
- Week 3: Reduce CPU overhead + optional display/encode
- Week 4: Polish docs + record demo video + tag release

### Month Goal 2: Compiler skeleton (ONNX → bundle)
- Week 1: ONNX importer + graph IR + shape inference
- Week 2: Quant assignment + weight packing generalized
- Week 3: Emit runtime config + run YOLOv2 from bundle (no hand edits)
- Week 4: Add one more model (e.g., YOLOv3-tiny) to validate “framework-ness”

### Month Goal 3: HLS throughput push
- Week 1: Add cycle counters + bandwidth measurements
- Week 2: DDR traffic reduction (tiling / buffering)
- Week 3: Conv+BN+Act fusion + improved dataflow
- Week 4: Benchmark + accuracy/perf report + tag release

---

## 11) A candid answer to your “did I finish the groundwork?” question

Yes: you proved the full stack works (model → FPGA → Linux → correct detections). That’s the “hard proof” many projects never reach.

If your goal is to feel like “I built a small DPU stack”, the missing pieces are mostly:
- A generalized importer (ONNX)
- A generalized compiler/mapping step
- A stable runtime/model bundle concept
- Broader operator/model coverage + automated validation

The fastest, most satisfying next step is usually **camera mode**, because it turns the project into a living system.  
The most intellectually “framework-like” next step is **ONNX → bundle**, because it reduces manual glue and makes it scalable across models.
