# Vivado (KV260)

This folder contains:
- `bd/kv260_yolov2_int16_bd.tcl`: an exported IP Integrator Block Design (generated in Vivado 2024.2)
- `build_from_bd.sh` + `build_from_bd.tcl`: a **no-GUI** build flow that recreates a Vivado project from the BD TCL and produces a **bitstream + XSA**

Why this exists:
- Committing the full `vivado/<project>/` directory to GitHub is usually a bad idea (huge + tool-generated).
- A BD TCL + batch build script makes the project reproducible.

## Prerequisites

- Vivado **2024.2** (the BD TCL has a strict version check)
- KV260 board files installed in Vivado (board part `xilinx.com:kv260_som:part0:1.4`)
- The custom HLS IP `xilinx.com:hls:YOLO2_FPGA:1.0` must be available as an IP repository
  - To generate/export it, run the INT16 Vitis HLS flow (see `vitis/README.md`):
    - `HLS_RUN_COSIM=0 vitis-run --mode hls --tcl vitis/yolo2_int16_cli.tcl`
  - Typical output location (used with `--ip-repo`): `yolo2_int16/solution1/impl/ip` (also packaged as `yolo2_int16_ip.zip`)

## Batch Build (no GUI)

From the repo root:

```bash
source /tools/Xilinx/Vivado/2024.2/settings64.sh

vivado/build_from_bd.sh \
  --bd-tcl vivado/bd/kv260_yolov2_int16_bd.tcl \
  --proj-dir vivado/yolov2_int16_autogen \
  --xsa vivado/yolov2_int16_autogen/design_1_wrapper.xsa \
  --ip-repo yolo2_int16/solution1/impl/ip \
  --jobs 8
```

Outputs:
- Vivado project: `vivado/yolov2_int16_autogen/`
- Bitstream: inside `vivado/yolov2_int16_autogen/<project>.runs/impl_1/`
- XSA (includes bitstream): `vivado/yolov2_int16_autogen/design_1_wrapper.xsa`

## Notes

- The `linux_app` uses `.bit.bin` + `.dtbo` via `xmutil`, so you’ll likely still run the packaging steps in `linux_app/accel_package/` after generating a new `.bit`.
