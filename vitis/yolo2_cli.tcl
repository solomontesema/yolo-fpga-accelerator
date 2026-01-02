# Simple Vitis HLS batch flow for YOLOv2 FP32 (csim → csynth → cosim → c-impl → IP export).
# Run from repo root:
#   vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
# To skip stages: HLS_RUN_CSIM=0, HLS_RUN_COSIM=0, HLS_RUN_IMPL=0, or HLS_RUN_EXPORT=0 in env.
# For INT16 version, use: vitis/yolo2_int16_cli.tcl

proc norm {p} { file normalize $p }

set script_dir [file dirname [norm [info script]]]
set proj_root  [norm [file join $script_dir ..]]

set top        "YOLO2_FPGA"
# Target device part for HLS.
#
# Default matches the Vivado KV260/K26 SOM project part.
# You can override at runtime:
#   HLS_PART=<part> vitis-run --mode hls --tcl vitis/yolo2_cli.tcl
#
# If the default part is not installed/recognized in your Vivado/Vitis setup,
# we fall back to a generic ZU5EV part string that commonly works.
set part_default "xck26-sfvc784-2LV-c"
set part_fallback "xczu5ev-sfvc784-1-i"
set part $part_default
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
  set part $::env(HLS_PART)
}
set clk_period 5.0

# Use the full co-simulation testbench
set tb_file [norm [file join $proj_root vitis yolo2_cosim_tb.cpp]]

set include_flags "-std=c++14 -I$proj_root/include -I$proj_root/include/core -I$proj_root/include/models/yolov2 -I$proj_root/hls -I$proj_root/hls/core -I$proj_root/hls/models/yolov2"

set design_files [list                            \
  [norm [file join $proj_root hls core core_io.cpp]]        \
  [norm [file join $proj_root hls core core_compute.cpp]]   \
  [norm [file join $proj_root hls core core_scheduler.cpp]] \
  [norm [file join $proj_root hls models yolov2 yolo2_accel.cpp]]]

# Host-side support for full co-sim TB (Darknet-lite stack)
set tb_support_files [list                               \
  [norm [file join $proj_root src core yolo_cfg.cpp]]    \
  [norm [file join $proj_root src core yolo_image.cpp]]  \
  [norm [file join $proj_root src core yolo_layers.cpp]] \
  [norm [file join $proj_root src core yolo_math.cpp]]   \
  [norm [file join $proj_root src core yolo_net.cpp]]    \
  [norm [file join $proj_root src core yolo_post.cpp]]   \
  [norm [file join $proj_root src core yolo_region.cpp]] \
  [norm [file join $proj_root src core yolo_utils.cpp]]  \
  [norm [file join $proj_root hls models yolov2 model_config.cpp]] \
  [norm [file join $proj_root hls models yolov2 yolo2_model.cpp]]]

proc build_project {proj_name} {
  global top part clk_period design_files include_flags tb_file tb_support_files part_fallback

  open_project -reset $proj_name
  set_top $top

  foreach f $design_files {
    add_files $f -cflags $include_flags
  }
  if {[file exists $tb_file]} {
    add_files -tb $tb_file -cflags $include_flags
    foreach f $tb_support_files {
      add_files -tb $f -cflags $include_flags
    }
  }

  open_solution -reset solution1
  if {[catch { set_part $part } err]} {
    puts "WARNING: set_part '$part' failed: $err"
    puts "WARNING: Falling back to '$part_fallback' (override with HLS_PART=... to silence this)."
    set part $part_fallback
    set_part $part
  }
  puts "Using HLS target part: $part"
  create_clock -period $clk_period -name default

  # CRITICAL: Set AXI depths BEFORE synthesis to ensure wrapc generation uses correct values
  # These depths are in 32-bit words and MUST match the pragmas in yolo2_accel.cpp
  # Input: 416*416*32 + 208*208*32 = 6,922,240 words = 27,688,960 bytes
  # Output: 416*416*32 = 5,537,792 words = 22,151,168 bytes
  # Weight: weights_reorg.bin = 50,941,792 words = 203,767,168 bytes
  # Beta: bias.bin = 10,761 words = 43,044 bytes
  # Note: TCL directive overrides pragma, so these values take precedence
  set_directive_interface -mode m_axi   -depth 6922240   -bundle DATA_BUS_IN  $top Input
  set_directive_interface -mode m_axi   -depth 5537792   -bundle DATA_BUS_OUT $top Output
  set_directive_interface -mode m_axi   -depth 50941792  -bundle DATA_BUS1    $top Weight
  set_directive_interface -mode m_axi   -depth 10761     -bundle DATA_BUS1    $top Beta

  if {![info exists ::env(HLS_RUN_CSIM)] || $::env(HLS_RUN_CSIM) != 0} {
    csim_design
  }

  csynth_design

  if {![info exists ::env(HLS_RUN_COSIM)] || $::env(HLS_RUN_COSIM) != 0} {
    cosim_design -rtl verilog -trace_level all
  }

  # C Implementation (RTL synthesis and place & route)
  if {![info exists ::env(HLS_RUN_IMPL)] || $::env(HLS_RUN_IMPL) != 0} {
    puts "Running C implementation (RTL synthesis and P&R)..."
    csynth_design
    # Note: Full implementation (place & route) is typically done in Vivado
    # HLS only does RTL synthesis. For full P&R, export IP and run in Vivado.
  }

  # Export IP for Vivado integration
  if {![info exists ::env(HLS_RUN_EXPORT)] || $::env(HLS_RUN_EXPORT) != 0} {
    puts "Exporting IP catalog..."
    set export_dir [file join [pwd] "${proj_name}_ip"]
    export_design -format ip_catalog -rtl verilog -output $export_dir
    puts "IP exported to: $export_dir"
    puts "To use in Vivado: Add IP Repository -> $export_dir"
  }
}

build_project yolo2_fp32
