#!/usr/bin/env bash
set -euo pipefail

ACCEL_NAME="yolov2_accel"
FIRMWARE_DIR="/lib/firmware/xilinx/$ACCEL_NAME"

echo "Deploying $ACCEL_NAME to KV260..."

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo)" >&2
  exit 1
fi

mkdir -p "$FIRMWARE_DIR"

cp "${ACCEL_NAME}.bit.bin" "$FIRMWARE_DIR/"
if [[ -f "${ACCEL_NAME}.dtbo" ]]; then
  cp "${ACCEL_NAME}.dtbo" "$FIRMWARE_DIR/"
else
  echo "WARNING: ${ACCEL_NAME}.dtbo not found in package folder (xmutil may fail)" >&2
fi
cp shell.json "$FIRMWARE_DIR/"

echo ""
echo "Files deployed to $FIRMWARE_DIR:"
ls -la "$FIRMWARE_DIR/"
echo ""
echo "To load the accelerator:"
echo "  sudo xmutil unloadapp"
echo "  sudo xmutil loadapp $ACCEL_NAME"
