# YOLOv2 FPGA Accelerator (KV260) – Linux App

This folder contains a userspace Linux application that runs **YOLOv2 INT16 inference on a Kria KV260** using a custom **HLS-generated FPGA accelerator**.

What this includes:
- Userspace driver via `/dev/mem` (AXI-Lite control + AXI GPIO Q-values)
- DMA-capable buffers via `udmabuf` (`u-dma-buf` kernel module)
- End-to-end pipeline: preprocessing → INT16 inference → region forward (sigmoid/softmax) → NMS
- Runtime verbosity control (`-v 0..3`) and optional region output dumps

If you want one file to get you from “fresh board” → “detections”, this is it.

---

## Quick Start (KV260)

Assumptions:
- You are on **Kria Ubuntu 22.04** on KV260 (or a compatible image that includes `xmutil`)
- Your KV260 is reachable over SSH as `ubuntu@kria` (change as needed)

### 1) Copy required files to the KV260

```bash
# Linux app
scp -r linux_app ubuntu@kria:/home/ubuntu/

# Weights + config + a test image
ssh ubuntu@kria "mkdir -p /home/ubuntu/weights /home/ubuntu/config /home/ubuntu/test_images"
scp weights/weights_reorg_int16.bin weights/bias_int16.bin weights/weight_int16_Q.bin weights/bias_int16_Q.bin weights/iofm_Q.bin ubuntu@kria:/home/ubuntu/weights/
scp config/yolov2.cfg config/coco.names ubuntu@kria:/home/ubuntu/config/
scp examples/test_images/dog.jpg ubuntu@kria:/home/ubuntu/test_images/
```

### 2) Deploy the FPGA firmware package (bitstream + dtbo) for `xmutil`

This repo includes an accelerator package template in `linux_app/accel_package/yolov2_accel/`.

```bash
scp -r linux_app/accel_package/yolov2_accel ubuntu@kria:/home/ubuntu/
ssh ubuntu@kria "cd /home/ubuntu/yolov2_accel && sudo ./deploy_to_kv260.sh"
```

### 3) Build the Linux app on the KV260

```bash
ssh ubuntu@kria "cd /home/ubuntu/linux_app && make clean && make"
```

### 4) Run

The recommended entrypoint is `start_yolo.sh` because it:
- loads the FPGA app via `xmutil`
- loads `u-dma-buf` with known-good buffer sizes
- sets `udmabuf*/sync_mode=1`
- runs `yolo2_linux` with the arguments you provide

```bash
ssh ubuntu@kria "cd /home/ubuntu/linux_app && ./start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg"
```

Try quiet vs debug:

```bash
ssh ubuntu@kria "cd /home/ubuntu/linux_app && ./start_yolo.sh -v 0 -i /home/ubuntu/test_images/dog.jpg"
ssh ubuntu@kria "cd /home/ubuntu/linux_app && ./start_yolo.sh -v 3 -i /home/ubuntu/test_images/dog.jpg"
```

---

## End-to-End Firmware Build (Host Machine)

You only need this section if you modify the HLS/Vivado design (or you don’t trust the included artifacts).

### 1) Generate the Vivado outputs

From Vivado, generate:
- Bitstream: `vivado/yolov2_int16/yolov2_int16.runs/impl_1/design_1_wrapper.bit`
- XSA: `vivado/yolov2_int16/design_1_wrapper.xsa`

### 2) Convert `.bit` → `.bit.bin` (required by Kria tooling)

On a machine with Vivado installed (for `bootgen`):

```bash
cd linux_app/accel_package

# Uses bootgen if available and writes into linux_app/accel_package/yolov2_accel/
./create_accel_package.sh
```

This produces:
- `linux_app/accel_package/yolov2_accel/yolov2_accel.bit.bin`
- a device-tree overlay (see next section)
- `shell.json`

### 3) Generate the DTBO (recommended: official XSCT flow)

The DTBO that worked reliably is generated from the XSA using **XSCT / createdts** (official AMD/Xilinx approach).

Follow: `linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md`

Practical notes:
- The official flow produces a `pl.dtsi` (and often a `pl.dtbo` after compiling with `dtc`)
- For use with the provided package/scripts, rename/copy your overlay as:
  - `yolov2_accel.dtbo` → `/lib/firmware/xilinx/yolov2_accel/`

If you generated only a `.dtsi` (and want to compile on the KV260):

```bash
# From the host:
scp /tmp/dts_output/yolov2_kv260/psu_cortexa53_0/device_tree_domain/bsp/pl.dtsi ubuntu@kria:/home/ubuntu/yolov2_accel/yolov2_accel.dtsi

# On the KV260:
ssh ubuntu@kria "sudo dtc -@ -O dtb -o /home/ubuntu/yolov2_accel/yolov2_accel.dtbo /home/ubuntu/yolov2_accel/yolov2_accel.dtsi"
ssh ubuntu@kria "sudo cp /home/ubuntu/yolov2_accel/yolov2_accel.dtbo /lib/firmware/xilinx/yolov2_accel/"
```

### 4) Copy the updated firmware package to KV260 and deploy

```bash
scp -r linux_app/accel_package/yolov2_accel ubuntu@kria:/home/ubuntu/
ssh ubuntu@kria "cd /home/ubuntu/yolov2_accel && sudo ./deploy_to_kv260.sh"
```

---

## UDMABUF (DMA Buffers)

`yolo2_linux` uses `udmabuf` to get physically contiguous buffers that the PL can DMA to/from.

### What sizes do we need?

Typical INT16 sizes (approx):
- Weights: ~97 MiB → allocate **≥ 128 MiB**
- Inference buffer: ~14 MiB → allocate **≥ 32 MiB**
- Bias: ~22 KiB → allocate **≥ 1 MiB**

### The working approach used by this repo

`linux_app/start_yolo.sh` loads the module each run with known-good sizes:
- `udmabuf0=134217728` (128 MiB)
- `udmabuf1=1048576` (1 MiB)
- `udmabuf2=33554432` (32 MiB)

It then sets:
- `/sys/class/u-dma-buf/udmabuf*/sync_mode = 1`

### Installing `u-dma-buf` on the KV260

If `/lib/modules/$(uname -r)/extra/u-dma-buf.ko` does not exist on your board:
- `linux_app/setup/install_udmabuf.sh` builds + installs it (requires network access for `apt` + `git`)
- `linux_app/setup/manual_udmabuf_install.sh` helps if install fails after build

See: `linux_app/setup/README.md`

---

## Running the App

### Recommended: `start_yolo.sh`

```bash
cd /home/ubuntu/linux_app
./start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg
```

### Direct binary execution

If you already loaded the overlay and `udmabuf` yourself:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp yolov2_accel

cd /home/ubuntu/linux_app
sudo ./yolo2_linux -v 1 -i /home/ubuntu/test_images/dog.jpg
```

### CLI options

```text
Usage: sudo ./yolo2_linux [options]

Options:
  -i <image>    Input image path (default: /home/ubuntu/test_images/dog.jpg)
  --camera <dev>           Camera device (e.g., /dev/video0)
  --video <path>           Video file path (decoded via ffmpeg)
  -w <dir>      Weights directory (default: /home/ubuntu/weights)
  -c <config>   Network config file (default: /home/ubuntu/config/yolov2.cfg)
  -l <labels>   Labels file (default: /home/ubuntu/config/coco.names)
  -t <thresh>   Detection threshold (default: 0.24)
  -n <nms>      NMS threshold (default: 0.45)
  -v <level>    Verbosity 0..3 (overrides YOLO2_VERBOSE)
  --max-frames <N>          Stop after N inference runs (0 = infinite)
  --infer-every <N>         Run inference every N frames (default: 1)
  --cam-width <W>           Camera width (default: 640)
  --cam-height <H>          Camera height (default: 480)
  --cam-fps <fps>           Camera FPS (default: 30)
  --cam-format mjpeg|yuyv   Camera format (default: mjpeg; falls back to yuyv)
  --video-width <W>         Video output width (default: 640)
  --video-height <H>        Video output height (default: 480)
  --video-fps <fps>         Video output FPS (default: 30)
  --save-annotated-dir <d>  Save annotated PNG frames to directory
  --output-json <path>      Write detections JSONL (one object per inference)
  -h            Show help
```

### Verbosity levels

- `-v 0`: quiet (errors + detections only)
- `-v 1`: high-level progress (default)
- `-v 2`: per-layer info
- `-v 3`: debug (addresses, status polling)

---

## Camera mode (USB / V4L2)

Defaults:
- `/dev/video0`
- MJPEG 640×480 @ 30 FPS (falls back to YUYV if MJPEG isn't supported)

Example:

```bash
cd /home/ubuntu/linux_app
YOLO2_VERBOSE=3 YOLO2_NO_DUMP=1 ./start_yolo.sh \
  --camera /dev/video0 \
  --max-frames 5 \
  --save-annotated-dir /home/ubuntu/out_cam
```

Direct run (if overlay + `udmabuf` are already loaded):

```bash
cd /home/ubuntu/linux_app
sudo YOLO2_VERBOSE=3 YOLO2_NO_DUMP=1 ./yolo2_linux \
  --camera /dev/video0 \
  --max-frames 5 \
  --save-annotated-dir ./out
```

This writes annotated frames:
- `/home/ubuntu/out_cam/frame_000001.png`, ...

## Video file mode (ffmpeg)

Requires `ffmpeg` on the KV260:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Example:

```bash
cd /home/ubuntu/linux_app
YOLO2_VERBOSE=1 YOLO2_NO_DUMP=1 ./start_yolo.sh \
  --video /home/ubuntu/test_videos/test.mp4 \
  --infer-every 5 \
  --max-frames 10 \
  --save-annotated-dir /home/ubuntu/out_vid \
  --output-json /home/ubuntu/out_vid/dets.jsonl
```


The environment variable `YOLO2_VERBOSE=0..3` is also supported. If you run via `sudo`, prefer `-v` or use `start_yolo.sh` (it forwards `YOLO2_*` variables through `sudo env ...`).

### Environment variables

- `YOLO2_LAYER_TIMEOUT_MS` (default: `60000`): per-layer watchdog timeout
- `YOLO2_NO_DUMP=1`: disable region dump files
- `YOLO2_DUMP_REGION_RAW=/path/file.txt`: override raw dump path
- `YOLO2_DUMP_REGION=/path/file.txt`: override processed dump path
- `YOLO2_VERBOSE=0..3`: verbosity (see note above)

### Output artifacts (by default)

Unless `YOLO2_NO_DUMP=1` is set, the app dumps:
- `yolov2_region_raw_hw.txt` (raw dequantized conv output, pre-sigmoid/softmax)
- `yolov2_region_proc_hw.txt` (after sigmoid/softmax)

---

## Hardware/Software Interface (What Must Match Your Bitstream)

The userspace driver uses the base addresses in `linux_app/include/yolo2_config.h`. These must match your Vivado Address Editor (and your DT overlay if you use one).

Base addresses:
- `0xA0000000`: YOLOv2 accelerator AXI-Lite control
- `0xA0010000`: AXI GPIO Qw
- `0xA0020000`: AXI GPIO Qa_in
- `0xA0030000`: AXI GPIO Qa_out
- `0xA0040000`: AXI GPIO Qb

Important control-register details:
- Addresses are 64-bit (written as low/high 32-bit words)
- `ap_done` / `ap_ready` are **clear-on-read** in this design
- Output address register offset is `0x1c` (not `0x18`)

---

## Project Structure

```
linux_app/
├── src/
│   ├── main.c                 # Main application
│   ├── yolo2_accel_linux.c    # Accelerator driver
│   ├── dma_buffer_manager.c   # DMA buffer allocation
│   ├── yolo2_inference.c      # Inference orchestration
│   ├── yolo2_network.c        # Network config parsing
│   ├── yolo2_postprocess.c    # NMS and detection
│   ├── yolo2_image_loader.c   # Image loading (stb_image)
│   ├── yolo2_log.c            # Verbosity-controlled logging
│   ├── yolo2_labels.c         # Label loading
│   ├── file_loader.c          # Binary file loading
│   └── stb_image_impl.c       # stb_image implementation
├── include/
│   ├── yolo2_config.h         # Hardware configuration
│   ├── yolo2_accel_linux.h    # Accelerator driver API
│   ├── dma_buffer_manager.h   # DMA buffer API
│   ├── yolo2_inference.h      # Inference API
│   ├── yolo2_network.h        # Network structures
│   ├── yolo2_postprocess.h    # Post-processing API
│   ├── yolo2_image_loader.h   # Image loader API
│   ├── yolo2_log.h            # Logging macros + verbosity
│   ├── yolo2_labels.h         # Labels API
│   └── file_loader.h          # File loader API
├── accel_package/             # Tools/artifacts for xmutil package
├── setup/
│   ├── install_udmabuf.sh     # udmabuf setup script
│   └── device_tree/           # Device tree overlays
├── tests/
│   ├── test_accel.c           # Accelerator test
│   └── test_dma.c             # DMA buffer test
├── Makefile
├── start_yolo.sh              # Load firmware + udmabuf and run
└── README.md
```

## More Docs

- `linux_app/accel_package/README.md`: build/deploy the `xmutil` accelerator package
- `linux_app/setup/README.md`: install/configure `u-dma-buf` / `udmabuf`
- `linux_app/tests/README.md`: small sanity-check binaries

## License

This project is provided as-is for educational and research purposes.
