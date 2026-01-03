# Generating DTBO from XSA using Official AMD/Xilinx Method

This document describes how to generate the device tree overlay (DTBO) from your hardware design XSA file using the official AMD/Xilinx tools.

## Prerequisites

- XSA file from your Vivado/Vitis design
- Vitis/XSCT tools installed
- Access to the XSA file: `vivado/yolov2_int16/design_1_wrapper.xsa`

## Pipeline shortcut (recommended)

If you are using the repo’s staged pipeline runner, it can run this XSCT flow for you when `package_firmware.dtbo.method: xsct`:

```bash
python3 scripts/run_pipeline.py --config pipeline.local.yaml --from package_firmware --to package_firmware
```

## Method 1: Using XSCT (Xilinx Software Command-Line Tool)

### Step 1: Launch XSCT

```bash
# Source Vitis environment
source /tools/Xilinx/Vitis/2024.2/settings64.sh
# Or adjust the path to your Vitis installation

# Start XSCT
xsct
```

Tip (headless machines / invalid `$DISPLAY`):

```bash
xsct -nodisp
```

### Step 2: Generate Device Tree from XSA

Within the XSCT shell:

```tcl
# Generate device tree overlay (createdts opens the XSA internally)
createdts -hw /path/to/vivado/yolov2_int16/design_1_wrapper.xsa \
          -zocl \
          -platform-name yolov2_kv260 \
          -git-branch xlnx_rel_v2024.1 \
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

# Copy into the xmutil package folder (recommended)
cp pl.dtbo <repo_root>/linux_app/accel_package/yolov2_accel/yolov2_accel.dtbo

# Or copy directly to the firmware directory on the KV260 (if you prefer)
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
- Manual overlays are possible but can be fragile across KV260 images; prefer the XSCT flow for reproducibility.

## Current Manual Approach

The current `yolov2_accel.dtsi` uses a manual overlay approach that:
- Overlays directly onto `&amba` (simpler, but not guaranteed to apply on every KV260 image)
- Uses `generic-uio` compatible string for userspace access
- Matches the addresses from `yolo2_config.h`

This manual approach is valid and often preferred for simple overlays when you know the exact addresses.
