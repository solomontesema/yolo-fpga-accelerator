#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning build artifacts..."

if [ -d "yolo2_int16" ]; then
    rm -rf yolo2_int16
    echo "  removed yolo2_int16/"
fi

if [ -f "yolo2_int16_ip.zip" ]; then
    rm -f yolo2_int16_ip.zip
    echo "  removed yolo2_int16_ip.zip"
fi

if [ -d "vivado/yolov2_int16" ]; then
    rm -rf vivado/yolov2_int16
    echo "  removed vivado/yolov2_int16/"
fi

bitstream_count=$(find . -type f \( -name "*.bit" -o -name "*.xsa" \) | wc -l)
if [ "$bitstream_count" -gt 0 ]; then
    find . -type f \( -name "*.bit" -o -name "*.xsa" \) -delete
    echo "  removed $bitstream_count bit/xsa file(s)"
fi

if [ -d "cosim_output" ]; then
    rm -rf cosim_output
fi
mkdir -p cosim_output
echo "  reset cosim_output/"

if [ -d ".autopilot" ]; then
    rm -rf .autopilot
    echo "  removed .autopilot/"
fi

mkdir -p build
if [ -n "$(find build -mindepth 1 -print -quit)" ]; then
    find build -mindepth 1 -delete
    echo "  cleared build/"
fi

log_count=$(find . -maxdepth 2 -type f -name "*.log" ! -path "./reports/*" | wc -l)
if [ "$log_count" -gt 0 ]; then
    find . -maxdepth 2 -type f -name "*.log" ! -path "./reports/*" -delete
    echo "  removed $log_count log file(s)"
fi

if [ -d "tmp/dts_output/.Xil" ]; then
    rm -rf tmp/dts_output/.Xil
    echo "  removed tmp/dts_output/.Xil/"
fi

jou_count=$(find . -type f -name "*.jou" | wc -l)
if [ "$jou_count" -gt 0 ]; then
    find . -type f -name "*.jou" -delete
    echo "  removed $jou_count .jou file(s)"
fi

echo "Cleanup done."
