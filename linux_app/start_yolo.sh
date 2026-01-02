#!/bin/bash
# Load accelerator and udmabuf, then run YOLOv2

echo "Loading YOLOv2 accelerator..."
sudo xmutil unloadapp 2>/dev/null
sudo xmutil loadapp yolov2_accel

echo "Loading udmabuf..."
sudo rmmod u-dma-buf 2>/dev/null
sudo insmod /lib/modules/$(uname -r)/extra/u-dma-buf.ko udmabuf0=134217728 udmabuf1=1048576 udmabuf2=33554432

echo "Setting sync mode..."
echo 1 | sudo tee /sys/class/u-dma-buf/udmabuf0/sync_mode > /dev/null
echo 1 | sudo tee /sys/class/u-dma-buf/udmabuf1/sync_mode > /dev/null
echo 1 | sudo tee /sys/class/u-dma-buf/udmabuf2/sync_mode > /dev/null

echo "Ready! Running YOLOv2..."
cd ~/linux_app

# Pass through YOLO2_* env vars even under sudo (sudo often resets the environment).
YOLO_ENV=()
for v in YOLO2_LAYER_TIMEOUT_MS YOLO2_NO_DUMP YOLO2_DUMP_REGION_RAW YOLO2_DUMP_REGION YOLO2_VERBOSE; do
  if [[ -n "${!v}" ]]; then
    YOLO_ENV+=("$v=${!v}")
  fi
done

sudo env "${YOLO_ENV[@]}" ./yolo2_linux "$@"
