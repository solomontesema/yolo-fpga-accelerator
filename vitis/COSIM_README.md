# YOLO2_FPGA Co-Simulation Guide

This guide explains how to run co-simulation of the YOLO2_FPGA accelerator using Vitis HLS.

## Overview

Co-simulation validates the RTL implementation against the C++ testbench by running full inference on real images. The testbench supports both FP32 and INT16 precision modes.

## Prerequisites

1. **Vitis HLS 2024.2** or compatible version installed and sourced
2. **Weight files generated**:
   ```bash
   make gen
   ./yolov2_weight_gen
   ```
   For INT16 mode, ensure INT16 weight files are generated:
   - `weights/weights_reorg_int16.bin`
   - `weights/bias_int16.bin`
   - `weights/weight_int16_Q.bin`
   - `weights/bias_int16_Q.bin`
   - `weights/iofm_Q.bin`

3. **Test image available**: Default is `examples/test_images/dog.jpg`

## Quick Start

### FP32 Mode

```bash
vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
```

### INT16 Mode

```bash
vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl
```

## Testbench Details

The co-simulation testbench (`vitis/yolo2_cosim_tb.cpp`) performs:

1. **Network configuration loading** from `config/yolov2.cfg`
2. **Image preprocessing**: Letterbox resize to 416x416x3
3. **Weight and bias loading**: Automatically selects FP32 or INT16 files based on build mode
4. **Full inference**: Executes all 32 layers through the accelerator
5. **Post-processing**: Runs region layer and NMS to generate detections
6. **Output saving**: Saves annotated image and output buffers

### INT16 Mode Features

- Automatic input quantization using activation Q values
- Per-layer quantization parameter handling (Qw, Qb, Qa_in, Qa_out)
- Route layer quantization alignment for proper concatenation

## Expected Outputs

After successful execution:
- `cosim_output/cosim_output.bin`: Final layer output buffer
- `cosim_output/cosim_output.png`: Annotated image with detections (if labels available)
- `cosim_output/layer_XX_output.bin`: Intermediate layer outputs (first 5 layers)

## Configuration

### AXI Interface Settings

The design uses AXI4 interfaces configured in `hls/models/yolov2/yolo2_accel.cpp`:
- Input/Output: `DATA_BUS_IN` / `DATA_BUS_OUT` (depth: 6,922,240 / 5,537,792 words)
- Weight/Beta: `DATA_BUS1` (depth: 50,941,792 / 10,761 words)

### Build Options

- `HLS_RUN_CSIM=0`: Skip C simulation
- `HLS_RUN_COSIM=0`: Skip co-simulation
- `HLS_TB=fast`: Use minimal testbench instead of full co-simulation testbench

## Known Limitations

Co-simulation may encounter stability issues with very large memory allocations. The testbench includes extensive memory management and bounds checking to mitigate these issues. For production validation, consider using C simulation or hardware deployment.

## Troubleshooting

### Weight File Issues

Ensure weight files are generated and match the expected precision mode:
```bash
ls -lh weights/weights_reorg*.bin
ls -lh weights/bias*.bin
```

### Image Loading Issues

Verify the test image exists:
```bash
ls -lh examples/test_images/dog.jpg
```

### Alphabet Images

The testbench requires `data/labels/` directory for annotated image generation. If missing, detections will still be printed but no annotated image will be saved.

## File Layout

```
project_root/
├── vitis/
│   ├── yolo2_cosim_tb.cpp          # Co-simulation testbench
│   ├── yolo2_cli.tcl                # FP32 build script
│   └── yolo2_int16_cli.tcl          # INT16 build script
├── examples/test_images/
│   └── dog.jpg                      # Test image
├── config/
│   └── yolov2.cfg                   # Network config
├── weights/                          # Weight files
└── data/labels/                     # Alphabet images for annotation
```
