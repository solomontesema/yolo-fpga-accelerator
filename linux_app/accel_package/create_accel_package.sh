#!/bin/bash
#
# Create Accelerator Package for KV260 xmutil
#
# This script creates a loadable accelerator package from the Vivado output files.
# Run this on a machine with Vivado/Vitis tools installed.
#
# Usage: ./create_accel_package.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Input files (adjust paths if needed)
XSA_FILE="$PROJECT_ROOT/vivado/yolov2_int16/design_1_wrapper.xsa"
BIT_FILE="$PROJECT_ROOT/vivado/yolov2_int16/yolov2_int16.runs/impl_1/design_1_wrapper.bit"

# Output directory
ACCEL_NAME="yolov2_accel"
OUTPUT_DIR="$SCRIPT_DIR/$ACCEL_NAME"

echo "========================================="
echo "KV260 Accelerator Package Creator"
echo "========================================="
echo ""
echo "Input files:"
echo "  XSA: $XSA_FILE"
echo "  BIT: $BIT_FILE"
echo ""

# Check input files exist
if [ ! -f "$XSA_FILE" ]; then
    echo "ERROR: XSA file not found: $XSA_FILE"
    exit 1
fi

if [ ! -f "$BIT_FILE" ]; then
    echo "ERROR: BIT file not found: $BIT_FILE"
    exit 1
fi

# Create output directory
echo "[1/4] Creating output directory..."
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "      Created: $OUTPUT_DIR"
echo ""

# Step 1: Convert bitstream to .bit.bin format
echo "[2/4] Converting bitstream to .bit.bin format..."

# Create BIF file for bootgen
BIF_FILE="$OUTPUT_DIR/bitstream.bif"
cat > "$BIF_FILE" << EOF
all:
{
    $BIT_FILE
}
EOF

# Check if bootgen is available
if command -v bootgen &> /dev/null; then
    bootgen -image "$BIF_FILE" -arch zynqmp -process_bitstream bin -o "$OUTPUT_DIR/${ACCEL_NAME}.bit.bin"
    echo "      Created: ${ACCEL_NAME}.bit.bin"
else
    echo "      WARNING: bootgen not found in PATH"
    echo "      Trying alternative method..."
    
    # Alternative: Just copy and rename (may need manual conversion)
    cp "$BIT_FILE" "$OUTPUT_DIR/${ACCEL_NAME}.bit"
    echo "      Copied bitstream (needs manual conversion with bootgen)"
    echo ""
    echo "      Run on a machine with Vivado:"
    echo "      bootgen -image bitstream.bif -arch zynqmp -process_bitstream bin"
fi
echo ""

# Step 2: Create device tree overlay
echo "[3/4] Creating device tree overlay..."

# For KV260, we create a simple overlay that works with the base platform
# The addresses match what's in yolo2_config.h
cat > "$OUTPUT_DIR/${ACCEL_NAME}.dtsi" << 'EOF'
/*
 * Device Tree Overlay for YOLOv2 FPGA Accelerator
 * 
 * This overlay describes the custom IP blocks in the PL:
 * - YOLOv2 accelerator (HLS IP) at 0xA0000000
 * - AXI GPIO for Q values at 0xA0010000-0xA0040000
 */

/dts-v1/;
/plugin/;

&fpga_full {
    firmware-name = "yolov2_accel.bit.bin";
};

&amba {
    /* YOLOv2 HLS Accelerator */
    yolo2_accel: yolo2_accel@a0000000 {
        compatible = "xlnx,yolo2-fpga-1.0";
        reg = <0x0 0xa0000000 0x0 0x1000>;
        /* interrupt-parent = <&gic>; */
        /* interrupts = <0 89 4>; */
    };
    
    /* AXI GPIO for Weight Q value */
    axi_gpio_qw: gpio@a0010000 {
        compatible = "xlnx,axi-gpio-2.0";
        reg = <0x0 0xa0010000 0x0 0x1000>;
        xlnx,gpio-width = <32>;
    };
    
    /* AXI GPIO for Input Activation Q value */
    axi_gpio_qa_in: gpio@a0020000 {
        compatible = "xlnx,axi-gpio-2.0";
        reg = <0x0 0xa0020000 0x0 0x1000>;
        xlnx,gpio-width = <32>;
    };
    
    /* AXI GPIO for Output Activation Q value */
    axi_gpio_qa_out: gpio@a0030000 {
        compatible = "xlnx,axi-gpio-2.0";
        reg = <0x0 0xa0030000 0x0 0x1000>;
        xlnx,gpio-width = <32>;
    };
    
    /* AXI GPIO for Bias Q value */
    axi_gpio_qb: gpio@a0040000 {
        compatible = "xlnx,axi-gpio-2.0";
        reg = <0x0 0xa0040000 0x0 0x1000>;
        xlnx,gpio-width = <32>;
    };
};
EOF

# Compile to dtbo if dtc is available
if command -v dtc &> /dev/null; then
    dtc -@ -O dtb -o "$OUTPUT_DIR/${ACCEL_NAME}.dtbo" "$OUTPUT_DIR/${ACCEL_NAME}.dtsi" 2>/dev/null || {
        echo "      WARNING: dtc compilation failed (may need device tree includes)"
        echo "      Manual compilation may be needed on KV260"
    }
    if [ -f "$OUTPUT_DIR/${ACCEL_NAME}.dtbo" ]; then
        echo "      Created: ${ACCEL_NAME}.dtbo"
    fi
else
    echo "      WARNING: dtc not found, dtbo not created"
    echo "      Install: sudo apt install device-tree-compiler"
fi
echo ""

# Step 3: Create shell.json metadata
echo "[4/4] Creating shell.json metadata..."

cat > "$OUTPUT_DIR/shell.json" << EOF
{
    "shell_type": "XRT_FLAT",
    "num_slots": 1,
    "uuid": "$(uuidgen 2>/dev/null || echo "00000000-0000-0000-0000-000000000001")",
    "pcie_config": {
        "device_id": "0x0001",
        "vendor_id": "0x10ee"
    }
}
EOF
echo "      Created: shell.json"
echo ""

# Create a simple deployment script
cat > "$OUTPUT_DIR/deploy_to_kv260.sh" << 'DEPLOY_EOF'
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
DEPLOY_EOF

chmod +x "$OUTPUT_DIR/deploy_to_kv260.sh"
echo "      Created: deploy_to_kv260.sh"

# Summary
echo ""
echo "========================================="
echo "Package created successfully!"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "1. If bootgen wasn't available, run it manually to create .bit.bin"
echo "2. Copy the $ACCEL_NAME folder to KV260:"
echo "   scp -r $OUTPUT_DIR ubuntu@kv260:/home/ubuntu/"
echo ""
echo "3. On KV260, deploy the accelerator:"
echo "   cd /home/ubuntu/$ACCEL_NAME"
echo "   sudo ./deploy_to_kv260.sh"
echo ""
echo "4. Load and test:"
echo "   sudo xmutil loadapp $ACCEL_NAME"
echo "   # Or use fpgautil directly"
echo "========================================="
