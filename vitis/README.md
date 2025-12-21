# Vitis HLS Build and Co-Simulation

This directory contains scripts and testbenches for building the YOLO2_FPGA accelerator using Vitis HLS and running co-simulation validation.

## Overview

The Vitis HLS flow provides:
- **C Simulation**: Validates the HLS design functionality
- **C Synthesis**: Generates RTL from the HLS design
- **Co-Simulation**: Validates RTL against the C++ testbench using real images
- **IP Export**: Generates IP catalog for Vivado integration

## Directory Contents

### Build Scripts
- **`yolo2_cli.tcl`**: FP32 precision build script
- **`yolo2_int16_cli.tcl`**: INT16 precision build script
- **`run_cosim.sh`**: Convenience script for co-simulation workflow

### Testbenches
- **`yolo2_cosim_tb.cpp`**: Full co-simulation testbench supporting both FP32 and INT16 modes
  - Loads real images and network configuration
  - Runs complete inference through all layers
  - Performs post-processing and detection visualization
  - Supports INT16 quantization with per-layer Q values

### Documentation
- **`COSIM_README.md`**: Detailed co-simulation guide
- **`COSIM_SUMMARY.md`**: Brief overview of co-simulation capabilities
- **`IP_EXPORT_README.md`**: Guide for exporting and using IP in Vivado

### Utilities
- **`clean_and_rebuild.sh`**: Script for clean rebuilds

## Quick Start

### Recommended: Build with Co-Simulation Skipped

Co-simulation currently encounters stability issues with large memory allocations. It is recommended to skip co-simulation and proceed directly to IP export:

**FP32 Build:**
```bash
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
```

**INT16 Build:**
```bash
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl
```

This performs C simulation, synthesis, and IP export without running co-simulation.

### Full Build (Including Co-Simulation)

If you need to run co-simulation (note: may encounter SIGSEGV with current memory allocations):

**FP32 Build:**
```bash
vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
```

**INT16 Build:**
```bash
vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl
```

### Build Options

Control which stages run using environment variables:

```bash
# Skip C simulation
HLS_RUN_CSIM=0 vitis-run --mode hls --tcl vitis/yolo2_cli.tcl

# Skip co-simulation (recommended due to current SIGSEGV issues)
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_cli.tcl

# Skip IP export
HLS_RUN_EXPORT=0 vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
```

**Recommended workflow**: Use `HLS_RUN_COSIM=0` to skip co-simulation and proceed directly to IP export for Vivado integration.

## Build Process

The TCL scripts perform the following steps:

1. **Project Setup**: Creates HLS project with design files
2. **C Simulation**: Validates functionality (optional, controlled by `HLS_RUN_CSIM`)
3. **C Synthesis**: Generates RTL from HLS design
4. **Co-Simulation**: Validates RTL against testbench (optional, controlled by `HLS_RUN_COSIM`)
   - **Note**: Co-simulation may encounter SIGSEGV with current large memory allocations
   - Recommended to skip using `HLS_RUN_COSIM=0` until memory issues are resolved
5. **IP Export**: Generates IP catalog for Vivado (optional, controlled by `HLS_RUN_EXPORT`)

## Output Directories

After building, the following directories are created:

- **`yolo2_fp32/`**: FP32 precision project
- **`yolo2_int16/`**: INT16 precision project

Each contains:
- **`solution1/syn/`**: Synthesis reports and RTL
- **`solution1/sim/`**: Co-simulation results
- **`<project>_ip/`**: Exported IP catalog (if export enabled)

## Co-Simulation Testbench

The `yolo2_cosim_tb.cpp` testbench:

- Automatically detects project root and resolves file paths
- Loads network configuration from `config/yolov2.cfg`
- Preprocesses input images (letterbox to 416x416x3)
- Loads weights and bias files (automatically selects FP32 or INT16 based on build mode)
- For INT16 mode: loads Q value tables and performs proper quantization/dequantization
- Runs full inference through all 32 layers
- Performs post-processing (NMS, detection visualization)
- Saves annotated output images and intermediate layer outputs

### INT16 Mode Features

- Automatic input quantization using activation Q values
- Per-layer quantization parameter handling (Qw, Qb, Qa_in, Qa_out)
- Route layer quantization alignment for proper concatenation
- Proper dequantization of region layer output for bounding box calculation

## Prerequisites

- **Vitis HLS 2024.2** or compatible version
- **Weight files**: See `weights/README.md` for weight file generation
- **Test images**: Default test image at `examples/test_images/dog.jpg`
- **Alphabet images**: For annotated output, requires `data/labels/` directory

## Expected Outputs

After co-simulation:

- **`cosim_output/cosim_output.bin`**: Final layer output buffer
- **`cosim_output/cosim_output.png`**: Annotated image with detections
- **`cosim_output/layer_XX_output.bin`**: Intermediate layer outputs (first 5 layers)

## Integration with Vivado

After IP export, the generated IP can be integrated into Vivado projects. See `IP_EXPORT_README.md` for detailed instructions.

## Troubleshooting

Common issues and solutions are documented in `COSIM_README.md`. Key points:

- Ensure weight files are generated before running
- Check that image paths are correct
- Verify all source files are present
- For INT16 mode, ensure all Q value files are available

## Notes

- **Co-Simulation Status**: Co-simulation currently encounters SIGSEGV issues with large memory allocations. It is recommended to skip co-simulation (`HLS_RUN_COSIM=0`) and proceed to IP export for Vivado integration.
- The testbench automatically handles both FP32 and INT16 precision modes
- Memory allocations are optimized for AXI co-simulation compatibility
- The testbench matches the CPU version behavior for validation consistency
- C simulation provides functional validation without the memory constraints of co-simulation

