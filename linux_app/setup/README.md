# KV260 Setup Helpers – `linux_app/setup`

This folder contains helper scripts for setting up **`udmabuf` / `u-dma-buf`** on the KV260.

`yolo2_linux` allocates DMA-capable buffers through `udmabuf` devices (e.g. `/dev/udmabuf0`) so the PL accelerator can DMA to/from DDR.

## Recommended Runtime Flow (what `start_yolo.sh` does)

`linux_app/start_yolo.sh` assumes the module exists at:

`/lib/modules/$(uname -r)/extra/u-dma-buf.ko`

It then loads it with known-good sizes:
- `udmabuf0=134217728` (128 MiB, weights)
- `udmabuf1=1048576` (1 MiB, bias)
- `udmabuf2=33554432` (32 MiB, inference)

and sets `sync_mode=1` for each buffer.

If that’s your workflow, you mostly just need the kernel module installed once.

## Install Script (requires internet access)

On the KV260:

```bash
cd /home/ubuntu/linux_app
sudo ./setup/install_udmabuf.sh
```

This script installs build deps, clones `udmabuf`, builds, and installs the kernel module.

## Manual Install Helper (if the install script built but did not install)

If `install_udmabuf.sh` built the module into `/tmp/udmabuf/` but the install step failed:

```bash
cd /home/ubuntu/linux_app
sudo ./setup/manual_udmabuf_install.sh
```

## Optional: Device-Tree Reserved-Memory Overlay

If your system’s CMA is too small (or you want fixed physical regions), you can use a DT overlay to reserve memory for `udmabuf`.

Template:
- `linux_app/setup/device_tree/yolo2_udmabuf.dts`

Typical compile/load flow:

```bash
dtc -@ -O dtb -o yolo2_udmabuf.dtbo yolo2_udmabuf.dts

sudo mkdir -p /sys/kernel/config/device-tree/overlays/yolo2
sudo cp yolo2_udmabuf.dtbo /sys/kernel/config/device-tree/overlays/yolo2/dtbo
```

