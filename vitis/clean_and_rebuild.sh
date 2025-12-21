#!/bin/bash
# Clean rebuild script to ensure wrapc is regenerated with correct depths

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Cleaning HLS project..."
cd "$PROJ_ROOT"

# Remove the entire project directory to force clean rebuild
if [ -d "yolo2_fp32" ]; then
    echo "  Removing yolo2_fp32 directory..."
    rm -rf yolo2_fp32
fi

# Also clean any autopilot cache
if [ -d ".autopilot" ]; then
    echo "  Removing .autopilot cache..."
    rm -rf .autopilot
fi

echo ""
echo "Verifying depth settings in TCL file..."
echo "  Input depth should be: 6922240 words = 27688960 bytes"
echo "  Output depth should be: 5537792 words = 22151168 bytes"
grep -A4 "set_directive_interface.*Input" "$SCRIPT_DIR/yolo2_cli.tcl" | head -1
grep -A4 "set_directive_interface.*Output" "$SCRIPT_DIR/yolo2_cli.tcl" | head -1

echo ""
echo "Rebuilding HLS project..."
echo "  This will regenerate wrapc with correct depths"
echo "  Run: vitis-run --mode hls --tcl vitis/yolo2_cli.tcl"
echo ""
echo "After rebuild, verify wrapc depths:"
echo "  grep 'port28.nbytes' yolo2_fp32/solution1/sim/wrapc/apatb_YOLO2_FPGA.cpp"
echo "  Should show: { 27688960 } for Input"
echo "  grep 'port29.nbytes' yolo2_fp32/solution1/sim/wrapc/apatb_YOLO2_FPGA.cpp"
echo "  Should show: { 22151168 } for Output"

