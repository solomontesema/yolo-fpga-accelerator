#!/bin/bash
#
# Deploy accelerator to KV260
# Run this ON the KV260 board
#

ACCEL_NAME="yolov2_accel"
FIRMWARE_DIR="/lib/firmware/xilinx/$ACCEL_NAME"

echo "Deploying $ACCEL_NAME to KV260..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Create firmware directory
mkdir -p "$FIRMWARE_DIR"

# Copy files
cp ${ACCEL_NAME}.bit.bin "$FIRMWARE_DIR/" 2>/dev/null || cp ${ACCEL_NAME}.bit "$FIRMWARE_DIR/${ACCEL_NAME}.bit.bin"
cp ${ACCEL_NAME}.dtbo "$FIRMWARE_DIR/" 2>/dev/null || echo "WARNING: dtbo not found"
cp shell.json "$FIRMWARE_DIR/"

echo ""
echo "Files deployed to $FIRMWARE_DIR:"
ls -la "$FIRMWARE_DIR/"

echo ""
echo "To load the accelerator:"
echo "  sudo xmutil unloadapp"
echo "  sudo xmutil loadapp $ACCEL_NAME"
echo ""
echo "Or manually via fpgautil:"
echo "  sudo fpgautil -b $FIRMWARE_DIR/${ACCEL_NAME}.bit.bin -o $FIRMWARE_DIR/${ACCEL_NAME}.dtbo"
