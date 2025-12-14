# yolo-fpga-accelerator

This project provides a modular C++/HLS implementation of YOLO for FPGA acceleration. The design is model-agnostic: common parsing, math, layers, and post-processing are shared, while the HLS path is factored into reusable `core_*` building blocks plus thin, model-specific wrappers (YOLOv2 today, YOLOv3/v4 and multiple precisions next).

## Features
- End-to-end detection CLI (`yolov2_detect`) with configurable thresholds and I/O paths.
- Structured host/HLS split: cfg parsing, math/activations, layer builders, and post-processing live in `src/`, while the accelerator top/scheduler/I/O/compute live in `hls/`.
- Hardware-ready flow: `scripts/hw_params_gen.py` emits on-chip tiling constants, and `make gen` reorganizes weights for the accelerator.

## Quick Start
```bash
# Build (auto-generates HLS params)
make test

# One-time weight reorganization
make gen
./yolov2_weight_gen

# Run detection (outputs land in results/)
./yolov2_detect \
  --cfg config/yolov2.cfg \
  --names config/coco.names \
  --input examples/test_images/dog.jpg \
  --output predictions \
  --thresh 0.5 --nms 0.45 --backend hls
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
- `scripts/` — hardware parameter generator
- `config/`, `examples/`, `weights/` — model config, sample images, and weight blobs

## Build Targets
- `make test` (default): build detector
- `make gen`: reorganize weights for HLS
- `make debug`: debug build
- `make clean` / `make distclean`: remove artifacts and generated blobs
