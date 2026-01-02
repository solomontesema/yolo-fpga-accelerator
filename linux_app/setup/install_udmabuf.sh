#!/bin/bash
#
# Install udmabuf kernel module on KV260
#
# This script installs the udmabuf kernel module which provides
# physically contiguous DMA buffer allocation from userspace.
#

set -e

echo "========================================="
echo "udmabuf Installation Script for KV260"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: Expected aarch64 architecture, got $ARCH"
fi

echo "[1/5] Installing dependencies..."
apt-get update
apt-get install -y git build-essential linux-headers-$(uname -r)

echo ""
echo "[2/5] Cloning udmabuf repository..."
cd /tmp
if [ -d "udmabuf" ]; then
    rm -rf udmabuf
fi
git clone https://github.com/ikwzm/udmabuf.git
cd udmabuf

echo ""
echo "[3/5] Building kernel module..."
make clean
make

echo ""
echo "[4/5] Installing kernel module..."
make install

echo ""
echo "[5/5] Loading module..."
modprobe u-dma-buf

# Verify installation
echo ""
echo "Verifying installation..."
if lsmod | grep -q u_dma_buf; then
    echo "SUCCESS: udmabuf module loaded"
else
    echo "ERROR: Module not loaded"
    exit 1
fi

# Create module load configuration
echo ""
echo "Creating module configuration..."
echo "u-dma-buf" > /etc/modules-load.d/udmabuf.conf

# Create buffer configuration
# These sizes are for YOLOv2 INT16:
# - weights buffer: ~97 MiB  -> allocate 128 MiB
# - bias buffer:    ~22 KiB  -> allocate 1 MiB
# - inference buf:  ~14 MiB  -> allocate 32 MiB (or larger)
cat > /etc/modprobe.d/udmabuf.conf << 'EOF'
# udmabuf buffer configuration for YOLOv2 accelerator
# Buffer sizes: 128MB for weights, 1MB for bias, 32MB for inference
options u-dma-buf udmabuf0=134217728 udmabuf1=1048576 udmabuf2=33554432
EOF

echo ""
echo "Reloading module with buffer configuration..."
rmmod u-dma-buf || true
modprobe u-dma-buf

# Check devices
echo ""
echo "Checking udmabuf devices..."
if [ -d /sys/class/u-dma-buf ]; then
    ls -la /sys/class/u-dma-buf/
    echo ""
    for dev in /sys/class/u-dma-buf/udmabuf*; do
        if [ -d "$dev" ]; then
            name=$(basename $dev)
            size=$(cat $dev/size 2>/dev/null || echo "N/A")
            phys=$(cat $dev/phys_addr 2>/dev/null || echo "N/A")
            echo "$name: size=$size bytes, phys=$phys"
        fi
    done
else
    echo "WARNING: No udmabuf devices found"
    echo "Buffers may need device tree configuration"
fi

echo ""
echo "========================================="
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. If no devices are shown, configure via device tree"
echo "2. Test with: cat /sys/class/u-dma-buf/udmabuf0/phys_addr"
echo "3. Run the YOLOv2 application: sudo ./yolo2_linux"
echo "========================================="
