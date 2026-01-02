#!/bin/bash
#
# Manual udmabuf installation after build
# Run this if install_udmabuf.sh fails at step 4
#

set -e

echo "========================================="
echo "Manual udmabuf Installation"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Check if module was built
if [ ! -f /tmp/udmabuf/u-dma-buf.ko ]; then
    echo "ERROR: Module not found at /tmp/udmabuf/u-dma-buf.ko"
    echo "       Run install_udmabuf.sh first to build it"
    exit 1
fi

echo "[1/4] Copying module to kernel modules directory..."
KERNEL_VER=$(uname -r)
MODULE_DIR="/lib/modules/${KERNEL_VER}/extra"
mkdir -p "$MODULE_DIR"
cp /tmp/udmabuf/u-dma-buf.ko "$MODULE_DIR/"
echo "      Copied to $MODULE_DIR/u-dma-buf.ko"

echo ""
echo "[2/4] Updating module dependencies..."
depmod -a
echo "      Done"

echo ""
echo "[3/4] Loading module with buffer sizes..."
# Unload if already loaded
rmmod u-dma-buf 2>/dev/null || true

# Load with buffer configuration
# 3 buffers: 128MB for weights, 1MB for bias, 32MB for inference
insmod /tmp/udmabuf/u-dma-buf.ko udmabuf0=134217728 udmabuf1=1048576 udmabuf2=33554432
echo "      Module loaded"

echo ""
echo "[4/4] Verifying..."
if lsmod | grep -q u_dma_buf; then
    echo "      SUCCESS: Module is loaded"
else
    echo "      ERROR: Module not loaded"
    exit 1
fi

# Check devices
echo ""
echo "Available udmabuf devices:"
if [ -d /sys/class/u-dma-buf ]; then
    for dev in /sys/class/u-dma-buf/udmabuf*; do
        if [ -d "$dev" ]; then
            name=$(basename $dev)
            size=$(cat $dev/size 2>/dev/null || echo "N/A")
            phys=$(cat $dev/phys_addr 2>/dev/null || echo "N/A")
            echo "  $name: size=$size bytes ($(echo "scale=1; $size/1048576" | bc) MB), phys=$phys"
        fi
    done
else
    echo "  WARNING: No devices found in /sys/class/u-dma-buf"
fi

# Check /dev entries
echo ""
echo "Device files:"
ls -la /dev/udmabuf* 2>/dev/null || echo "  No /dev/udmabuf* devices found"

echo ""
echo "========================================="
echo "Installation complete!"
echo ""
echo "To load on boot, add to /etc/modules-load.d/:"
echo "  echo 'u-dma-buf' | sudo tee /etc/modules-load.d/udmabuf.conf"
echo ""
echo "Now try: sudo ./test_dma"
echo "========================================="
