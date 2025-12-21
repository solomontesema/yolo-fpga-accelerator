# yolo-fpga-accelerator

This project provides a modular C++/HLS implementation of YOLO for FPGA acceleration. The design is model-agnostic: common parsing, math, layers, and post-processing are shared, while the HLS path is factored into reusable `core_*` building blocks plus thin, model-specific wrappers (YOLOv2 today, YOLOv3/v4 and multiple precisions next).

## Features
- End-to-end detection CLI (`yolov2_detect`) with configurable thresholds and I/O paths.
- Structured host/HLS split: cfg parsing, math/activations, layer builders, and post-processing live in `src/`, while the accelerator top/scheduler/I/O/compute live in `hls/`.
- Hardware-ready flow: `scripts/hw_params_gen.py` emits on-chip tiling constants, and `make gen` reorganizes weights for the accelerator.
- Dual precision: fp32 is the baseline; int16 uses per-layer activation/weight/bias Q tables generated from calibrated activations.
- Vitis HLS integration: Complete build scripts and co-simulation testbench for FPGA validation (see `vitis/README.md`).

## Quick Start
```bash
# Build (auto-generates HLS params)
make test

# One-time weight reorganization
make gen
./yolov2_weight_gen

# Run fp32 detection (outputs land in results/)
./yolov2_detect \
  --cfg config/yolov2.cfg \
  --names config/coco.names \
  --input examples/test_images/dog.jpg \
  --output predictions \
  --thresh 0.5 --nms 0.45 --backend hls

# Int16 build + detection (requires int16 weights/Q tables)
make test-int16
./yolov2_detect --precision int16 \
  --cfg config/yolov2.cfg \
  --names config/coco.names \
  --input examples/test_images/dog.jpg
```
Weights expected in `weights/`: `weights.bin`, `bias.bin`, and generated `weights/weights_reorg.bin` (see `weights/README.md`).
The detector saves a single annotated image; if `--output` is omitted it writes `results/<input>_prediction.png`.

## Layout at a Glance
- `include/core/` — shared APIs/types
- `include/models/yolov2/` — YOLOv2-specific pragmas
- `src/core/` — cfg/IO/math/layer builders/post-processing
- `src/models/yolov2/` — YOLOv2 CLI entry
- `hls/core/` — accelerator I/O/compute/scheduler building blocks
- `hls/models/yolov2/` — YOLOv2 accelerator wrapper and model descriptor
- `vitis/` — Vitis HLS build scripts and co-simulation testbench (see `vitis/README.md`)
- `scripts/` — hardware parameter generator
- `config/`, `examples/`, `weights/` — model config, sample images, and weight blobs

## Build Targets
- `make test` (default): build detector
- `make test-int16`: build detector with `INT16_MODE`
- `make gen`: reorganize weights for HLS
- `make debug`: debug build
- `make clean` / `make distclean`: remove artifacts and generated blobs

## Int16 Workflow (overview)
- Generate int16 weights/bias + Q tables using the companion extractor (see `weights/README.md`):
  - `weight_int16.bin`, `bias_int16.bin`
  - `weight_int16_Q.bin`, `bias_int16_Q.bin`
  - `iofm_Q.bin` derived from activation ranges on a calibration set
- Build with `make test-int16`
- Run with `./yolov2_detect --precision int16 ...`

## Vitis HLS Workflow

The project includes complete Vitis HLS build scripts for FPGA validation:

```bash
# Recommended: FP32 build (skip co-simulation due to current SIGSEGV issues)
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_cli.tcl

# Recommended: INT16 build (skip co-simulation)
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl
```

The build scripts support:
- C simulation validation
- RTL synthesis
- Co-simulation with full network inference (currently has stability issues)
- IP export for Vivado integration

**Note**: Co-simulation may encounter SIGSEGV with large memory allocations. It is recommended to skip co-simulation and proceed to IP export for Vivado integration.

See `vitis/README.md` for detailed usage and configuration options.
