# yolo-fpga-accelerator (KV260 YOLOv2 INT16)

An end-to-end, reproducible example of taking an object-detection model → extracting/quantizing weights → building an HLS accelerator → integrating it in Vivado → deploying it on a **Kria KV260** → running a **Linux userspace application** that produces correct detections.

Validated end-to-end with **YOLOv2 INT16** on KV260.

## What’s in this repo

- `weights/`: weight file expectations + how to generate weights/Q tables (uses external `nn-weight-extractor`, cloned locally into `weights/nn-weight-extractor/`)
- `hls/`: model-agnostic accelerator building blocks + YOLOv2 wrapper
- `vitis/`: Vitis HLS scripts to build/export the HLS IP repo (used by Vivado)
- `vivado/`: exported Block Design TCL + **no-GUI** Vivado build script to generate bitstream + XSA
- `linux_app/`: KV260 deployment package + userspace inference app (`yolo2_linux`)
- `ROADMAP.md`: future directions (camera, CI, performance, “framework-ization”)

## End-to-end: From scratch → detections on KV260 (one place, in order)

This is the “tomorrow I wake up and everything is formatted” path.

### 1) Host tooling sanity (Vivado/Vitis/XSCT)

- Install **Vivado 2024.2** and **Vitis HLS 2024.2**
- Ensure KV260 board files are installed in Vivado
- Ensure `xsct` is available (needed for the recommended DTBO generation flow)

See: `vitis/README.md`, `vivado/README.md`, `linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md`

### 2) Generate model artifacts (weights + INT16 quantization + Q tables)

Generate fp32/int16 weights + Q tables (includes the “where do I get/clone the extractor?” steps):
- `weights/README.md`

#### Quick start (host sanity-check before FPGA)

This is the most “encouraging” checkpoint: you validate that weight extraction + reorganization + inference all work on your host machine before going deeper into the FPGA flow.

```bash
# Build (auto-generates HLS params)
make test

# One-time weight reorganization
make gen
./yolov2_weight_gen

# Run fp32 detection (outputs land in results/ unless --output is set)
./yolov2_detect \
  --cfg config/yolov2.cfg \
  --names config/coco.names \
  --input examples/test_images/dog.jpg \
  --output predictions \
  --thresh 0.5 --nms 0.45 --backend hls

# Int16 build + detection (requires int16 weights/Q tables)
make test-int16
./yolov2_weight_gen --precision int16
./yolov2_detect --precision int16 \
  --cfg config/yolov2.cfg \
  --names config/coco.names \
  --input examples/test_images/dog.jpg
```

For the KV260 INT16 app you ultimately need these files (paths shown as they are used later):
- `weights/weights_reorg_int16.bin`
- `weights/bias_int16.bin`
- `weights/weight_int16_Q.bin`
- `weights/bias_int16_Q.bin`
- `weights/iofm_Q.bin`

### 3) Build/export the HLS IP repo (INT16)

This produces the Vivado IP repo that contains `xilinx.com:hls:YOLO2_FPGA:1.0`.

```bash
HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl
```

Expected output IP repo (used in the next step):
- `yolo2_int16/solution1/impl/ip`

See: `vitis/README.md`

### 4) Build Vivado bitstream + XSA (GUI or no-GUI)

No-GUI (recommended for reproducibility):

```bash
source /tools/Xilinx/Vivado/2024.2/settings64.sh

vivado/build_from_bd.sh \
  --bd-tcl vivado/bd/kv260_yolov2_int16_bd.tcl \
  --proj-dir vivado/yolov2_int16 \
  --xsa vivado/yolov2_int16/design_1_wrapper.xsa \
  --ip-repo yolo2_int16/solution1/impl/ip \
  --jobs 8
```

See: `vivado/README.md`

### 5) Package KV260 firmware (`.bit.bin` + `.dtbo`) for `xmutil`

The KV260 tooling expects a firmware folder under:
- `/lib/firmware/xilinx/yolov2_accel/`

The repo provides packaging scripts here:
- `linux_app/accel_package/README.md`

Create the `.bit.bin` (uses `bootgen` from Vivado):

```bash
cd linux_app/accel_package
./create_accel_package.sh
```

Recommended DTBO generation flow (official XSCT approach):
- `linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md`

### 6) Deploy to KV260 (copy files + install firmware + build app)

Assume KV260 is reachable as `ubuntu@kria` (change as needed).

```bash
# Linux app
scp -r linux_app ubuntu@kria:/home/ubuntu/

# Firmware package (bit.bin + dtbo + shell.json)
scp -r linux_app/accel_package/yolov2_accel ubuntu@kria:/home/ubuntu/
ssh ubuntu@kria "cd /home/ubuntu/yolov2_accel && sudo ./deploy_to_kv260.sh"

# Weights + config + a test image
ssh ubuntu@kria "mkdir -p /home/ubuntu/weights /home/ubuntu/config /home/ubuntu/test_images"
scp weights/weights_reorg_int16.bin weights/bias_int16.bin weights/weight_int16_Q.bin weights/bias_int16_Q.bin weights/iofm_Q.bin ubuntu@kria:/home/ubuntu/weights/
scp config/yolov2.cfg config/coco.names ubuntu@kria:/home/ubuntu/config/
scp examples/test_images/dog.jpg ubuntu@kria:/home/ubuntu/test_images/

# Build on the board
ssh ubuntu@kria "cd /home/ubuntu/linux_app && make clean && make"
```

See: `linux_app/README.md` and `linux_app/accel_package/README.md`

### 7) Load the overlay + run inference

Use the recommended entrypoint (loads overlay + `udmabuf` + runs inference):

```bash
ssh ubuntu@kria "cd /home/ubuntu/linux_app && ./start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg"
```

If this is a fresh board and `udmabuf` is missing, install it once:
- `linux_app/setup/README.md`

See: `linux_app/README.md`

## Documentation map

- KV260 app + runtime details: `linux_app/README.md`
- `xmutil` packaging: `linux_app/accel_package/README.md`
- DTBO generation from XSA (XSCT): `linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md`
- Vitis HLS build + IP export: `vitis/README.md`
- Vivado block design + batch build: `vivado/README.md`
- Weight generation + quantization: `weights/README.md`
- Next steps: `ROADMAP.md`

If you’re focused on the KV260 “it just works” path, start with the 7-step section above and the `linux_app/README.md` quick start.
