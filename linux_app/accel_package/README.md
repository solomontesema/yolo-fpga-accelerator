# Accelerator Package (`xmutil`) – `linux_app/accel_package`

This folder contains scripts and artifacts to package the FPGA design as a **Kria `xmutil loadapp` application**.

On the KV260, the installed firmware folder is:

`/lib/firmware/xilinx/yolov2_accel/`

The package typically contains:
- `yolov2_accel.bit.bin` (required)
- `yolov2_accel.dtbo` (required for `xmutil` on most images; recommended to generate from the XSA)
- `shell.json` (metadata used by the Kria tooling)

## Quick Deploy (use the included prebuilt package)

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

## Rebuild the Package (after regenerating Vivado bitstream/XSA)

If you generated a new bitstream/XSA (or changed the BD/address map), rebuild the package.

### Inputs

You need:
- XSA: `vivado/yolov2_int16/design_1_wrapper.xsa`
- Bitstream:
  - Prefer a Vivado-generated `*.bit.bin` if present, e.g.:
    - `vivado/yolov2_int16/yolov2_int16.runs/impl_1/design_1_wrapper.bit.bin`
  - Otherwise you need `bootgen` (Vivado) to convert `*.bit` → `*.bit.bin`.

### 1) Create/refresh the package folder (`.bit.bin` + metadata)

Run from the **repo root**:

```bash
./linux_app/accel_package/create_accel_package.sh \
  --accel-name yolov2_accel \
  --out-dir linux_app/accel_package/yolov2_accel \
  --xsa vivado/yolov2_int16/design_1_wrapper.xsa
```

Notes:
- `create_accel_package.sh` **does not modify** your Vivado `.xsa/.bit` files; it reads them and writes into the package folder.
- The script backs up any existing output folder under `linux_app/accel_package/_backups/` before swapping the new package into place.

### 2) Generate the DTBO (recommended: XSCT/createdts from the XSA)

The “manual overlay” approach is highly platform/image dependent; the most reliable method is the official AMD/Xilinx flow.

You can do this either:

- Automatically via the staged pipeline runner (`package_firmware` stage with `dtbo.method: xsct`)
- Manually (follow `linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md`)

If you generate DTBO manually, copy/rename it into the package folder as:
- `linux_app/accel_package/yolov2_accel/yolov2_accel.dtbo`

### 3) Deploy the refreshed package to KV260

```bash
scp -r linux_app/accel_package/yolov2_accel ubuntu@kria:/home/ubuntu/
ssh ubuntu@kria "cd /home/ubuntu/yolov2_accel && sudo ./deploy_to_kv260.sh"
```

## Pipeline Integration

If you use the repo’s staged pipeline runner, this is the `package_firmware` stage:
- Config: `pipeline.yaml`
- Runner: `scripts/run_pipeline.py`

With `package_firmware.dtbo.method: xsct`, the pipeline runs the official XSCT/createdts flow and passes the generated `pl.dtbo` into `create_accel_package.sh` via `--dtbo` (so the package swap is atomic).

Example (run only this stage):

```bash
python3 scripts/run_pipeline.py --config pipeline.local.yaml --from package_firmware --to package_firmware
```
