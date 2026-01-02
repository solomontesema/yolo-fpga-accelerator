# Linux App Tests – `linux_app/tests`

Small sanity-check programs that help verify the KV260 runtime environment:

- `test_accel`: checks `/dev/mem` register access to the accelerator address range
- `test_dma`: checks `udmabuf` allocation and physical-address mapping
- `test_pl_ddr`: basic PL↔DDR connectivity test (platform-specific)
- `check_hp_clocks`: prints/validates HP port clocking (platform-specific)

## Build

On the KV260:

```bash
cd /home/ubuntu/linux_app
make test_accel test_dma test_pl_ddr check_hp_clocks
```

## Run

```bash
cd /home/ubuntu/linux_app

# Make sure your overlay/bitstream is loaded first (see ../README.md).
sudo ./test_accel

# Make sure u-dma-buf is loaded first (or run via start_yolo.sh once).
sudo ./test_dma
```

