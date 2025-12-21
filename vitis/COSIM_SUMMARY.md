# Co-Simulation Summary

## Overview

The YOLO2_FPGA accelerator includes a comprehensive co-simulation testbench that validates the RTL implementation against the C++ reference model. The testbench supports both FP32 and INT16 precision modes.

## Testbench Features

**File**: `vitis/yolo2_cosim_tb.cpp`

- Full network inference (all 32 layers)
- Real image processing with letterbox preprocessing
- Automatic weight file selection based on precision mode
- INT16 quantization support with per-layer Q values
- Detection post-processing and visualization
- Output buffer saving for verification

## Build Scripts

- `vitis/yolo2_cli.tcl`: FP32 mode build configuration
- `vitis/yolo2_int16_cli.tcl`: INT16 mode build configuration

Both scripts configure:
- AXI interface depths matching the design requirements
- Proper include paths and compilation flags
- Testbench integration with host-side support files

## Usage

### FP32 Mode
```bash
vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
```

### INT16 Mode
```bash
vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl
```

## Output Files

- `cosim_output/cosim_output.bin`: Final layer output
- `cosim_output/cosim_output.png`: Annotated detection image
- `cosim_output/layer_XX_output.bin`: Intermediate layer outputs

## Requirements

- Weight files generated for the target precision mode
- Test image available (`examples/test_images/dog.jpg` by default)
- `data/labels/` directory for image annotation (optional)

## Notes

The testbench automatically handles project root detection and path resolution. For INT16 mode, ensure all quantization parameter files are present in the weights directory.
