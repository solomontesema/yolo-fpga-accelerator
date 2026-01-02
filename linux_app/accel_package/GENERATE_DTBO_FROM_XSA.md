# Generating DTBO from XSA using Official AMD/Xilinx Method

This document describes how to generate the device tree overlay (DTBO) from your hardware design XSA file using the official AMD/Xilinx tools.

## Prerequisites

- XSA file from your Vivado/Vitis design
- Vitis/XSCT tools installed
- Access to the XSA file: `vivado/yolov2_int16/design_1_wrapper.xsa`

## Method 1: Using XSCT (Xilinx Software Command-Line Tool)

### Step 1: Launch XSCT

```bash
# Source Vitis environment
source /tools/Xilinx/Vitis/2022.1/settings64.sh
# Or adjust path to your Vitis installation

# Start XSCT
xsct
```

### Step 2: Generate Device Tree from XSA

Within the XSCT shell:

```tcl
# Open hardware design
hsi::open_hw_design /path/to/vivado/yolov2_int16/design_1_wrapper.xsa

# Generate device tree overlay
createdts -hw /path/to/vivado/yolov2_int16/design_1_wrapper.xsa \
          -zocl \
          -platform-name yolov2_kv260 \
          -git-branch xlnx_rel_v2022.1 \
          -overlay \
          -compile \
          -out /tmp/dts_output
```

### Step 3: Extract and Compile DTBO

```bash
# Navigate to generated device tree directory
cd /tmp/dts_output/yolov2_kv260/psu_cortexa53_0/device_tree_domain/bsp

# Compile DTSI to DTBO
dtc -@ -O dtb -o pl.dtbo pl.dtsi

# Copy to firmware directory
sudo cp pl.dtbo /lib/firmware/xilinx/yolov2_accel/
```

## Method 2: Using PetaLinux (Alternative)

If you have PetaLinux installed:

```bash
# Create PetaLinux project from BSP
petalinux-create -t project -s xilinx-k26-starterkit-2022.1.bsp

# Configure project
cd <project_directory>
petalinux-config

# Enable fpgamanager_dtg class in configuration
# Build project
petalinux-build
```

## Notes

- The generated `pl.dtsi` will include all IP blocks from your hardware design
- You may need to adjust compatible strings for UIO access
- The official method ensures device tree matches your exact hardware configuration
- For manual overlays (current approach), the simpler `&amba` overlay works well for KV260

## Current Manual Approach

The current `yolov2_accel.dtsi` uses a manual overlay approach that:
- Overlays directly onto `&amba` (simpler, works well for KV260)
- Uses `generic-uio` compatible string for userspace access
- Matches the addresses from `yolo2_config.h`

This manual approach is valid and often preferred for simple overlays when you know the exact addresses.
