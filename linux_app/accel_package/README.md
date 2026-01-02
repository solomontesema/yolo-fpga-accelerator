# Accelerator Package (`xmutil`) – `linux_app/accel_package`

This folder contains scripts and artifacts to package the FPGA design for **Kria `xmutil loadapp`**.

The resulting directory on the KV260 is:

`/lib/firmware/xilinx/yolov2_accel/`

It typically contains:
- `yolov2_accel.bit.bin` (bitstream in `.bit.bin` format)
- `yolov2_accel.dtbo` (device-tree overlay)
- `shell.json` (metadata used by the Kria tooling)

## Quick Deploy (using the included package folder)

From your host machine:

```bash
scp -r linux_app/accel_package/yolov2_accel ubuntu@kria:/home/ubuntu/
ssh ubuntu@kria "cd /home/ubuntu/yolov2_accel && sudo ./deploy_to_kv260.sh"
```

Then on the KV260:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp yolov2_accel
```

## Rebuild the Package from Vivado Outputs

If you generated a new bitstream/XSA (or changed addresses), rebuild the package.

### Inputs (expected by the script)

`create_accel_package.sh` expects:
- XSA: `vivado/yolov2_int16/design_1_wrapper.xsa`
- BIT: `vivado/yolov2_int16/yolov2_int16.runs/impl_1/design_1_wrapper.bit`

### Create `.bit.bin` + a simple overlay

On a machine with Vivado installed (for `bootgen`):

```bash
cd linux_app/accel_package
./create_accel_package.sh
```

This writes files into `linux_app/accel_package/yolov2_accel/`.

## DTBO Generation (Recommended: Official XSCT Flow)

For the most reliable DT overlay generation, use the official AMD/Xilinx flow from the XSA:

`linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md`

You can then copy/rename the generated DTBO into this folder as `yolov2_accel.dtbo` before deploying.

