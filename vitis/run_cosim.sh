#!/bin/bash
# Co-simulation run script for YOLO2_FPGA
# Usage: ./vitis/run_cosim.sh [image_path] [config_path] [weights_dir] [output_dir]

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths
IMAGE_PATH="${1:-examples/test_images/dog.jpg}"
CFG_PATH="${2:-config/yolov2.cfg}"
WEIGHTS_DIR="${3:-weights}"
OUTPUT_DIR="${4:-cosim_output}"

# Convert to absolute paths
IMAGE_PATH="$(cd "$(dirname "$PROJ_ROOT/$IMAGE_PATH")" && pwd)/$(basename "$PROJ_ROOT/$IMAGE_PATH")"
CFG_PATH="$(cd "$(dirname "$PROJ_ROOT/$CFG_PATH")" && pwd)/$(basename "$PROJ_ROOT/$CFG_PATH")"
WEIGHTS_DIR="$(cd "$(dirname "$PROJ_ROOT/$WEIGHTS_DIR")" && pwd)/$(basename "$PROJ_ROOT/$WEIGHTS_DIR")"
OUTPUT_DIR="$PROJ_ROOT/$OUTPUT_DIR"

echo "=========================================="
echo "YOLO2_FPGA Co-Simulation Run Script"
echo "=========================================="
echo "Project root: $PROJ_ROOT"
echo "Image:        $IMAGE_PATH"
echo "Config:       $CFG_PATH"
echo "Weights:      $WEIGHTS_DIR"
echo "Output:       $OUTPUT_DIR"
echo ""

# Check prerequisites
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image file not found: $IMAGE_PATH"
    exit 1
fi

if [ ! -f "$CFG_PATH" ]; then
    echo "ERROR: Config file not found: $CFG_PATH"
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/weights_reorg.bin" ]; then
    echo "ERROR: Weights file not found: $WEIGHTS_DIR/weights_reorg.bin"
    echo "Please generate weights first using: make gen && ./yolov2_weight_gen"
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/bias.bin" ]; then
    echo "ERROR: Bias file not found: $WEIGHTS_DIR/bias.bin"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if Vitis HLS is available
if ! command -v vitis_hls &> /dev/null; then
    echo "ERROR: vitis_hls not found in PATH"
    echo "Please source Vitis settings: source /tools/Xilinx/Vitis/2024.2/settings64.sh"
    exit 1
fi

# Build project with co-simulation
echo "Building HLS project with co-simulation testbench..."
cd "$PROJ_ROOT"

# Use the new TCL build scripts
vitis-run --mode hls --tcl vitis/yolo2_cli.tcl 2>&1 | tee "$OUTPUT_DIR/build.log"

# Check if build succeeded
if [ ! -d "yolo2_fp32" ]; then
    echo "ERROR: Build failed. Check $OUTPUT_DIR/build.log"
    exit 1
fi

echo ""
echo "Co-simulation completed!"
echo "Results are in: $OUTPUT_DIR"

