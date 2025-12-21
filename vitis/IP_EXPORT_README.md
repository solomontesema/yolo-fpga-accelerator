# IP Export Guide

This guide explains how to export the YOLO2_FPGA accelerator as IP for Vivado integration.

## Export Process

The TCL build scripts (`yolo2_cli.tcl` and `yolo2_int16_cli.tcl`) automatically export IP after synthesis. The exported IP is placed in:

- `yolo2_fp32/yolo2_fp32_ip/` (FP32 version)
- `yolo2_int16/yolo2_int16_ip/` (INT16 version)

## Using Exported IP in Vivado

1. **Add IP Repository**:
   - In Vivado, go to `Settings -> IP -> Repository`
   - Click `+` and browse to the exported IP directory (e.g., `yolo2_fp32/yolo2_fp32_ip/`)
   - Click `Apply` and `OK`

2. **Add IP to Block Design**:
   - In Block Design, click `+` to add IP
   - Search for `YOLO2_FPGA`
   - Add the IP instance to your design

3. **Connect AXI Interfaces**:
   - Connect `m_axi_Input` and `m_axi_Output` to AXI Interconnect (HP port recommended)
   - Connect `m_axi_Weight` and `m_axi_Beta` to AXI Interconnect
   - Connect `s_axi_CTRL_BUS` to AXI Interconnect (GP port for control)

4. **Clock and Reset**:
   - Connect `ap_clk` to your system clock (typically 200MHz for ZynqMP)
   - Connect `ap_rst_n` to system reset

## IP Configuration

The IP supports the following AXI interfaces:

- **M_AXI Input/Output**: 64-bit width, burst length 64
- **M_AXI Weight/Beta**: 64-bit width, burst length 128
- **S_AXI Control**: 32-bit width for register access

## Resource Estimates

Check the synthesis reports in `yolo2_fp32/solution1/syn/report/` for:
- LUT/FF usage
- BRAM/DSP usage
- Estimated Fmax
- Initiation Interval (II)

Compare FP32 vs INT16 to understand resource trade-offs.

## Next Steps

After IP export:
1. Create Vivado block design with ZynqMP PS
2. Connect YOLO2_FPGA IP to AXI Interconnect
3. Generate bitstream
4. Create Vitis platform project for PS-side software

