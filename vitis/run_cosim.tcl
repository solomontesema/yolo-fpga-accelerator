# Standalone co-simulation run script for YOLO2_FPGA
# Usage: vitis_hls -f vitis/run_cosim.tcl
# Or: cd yolo2_fp32_cosim/YOLO2_FPGA/hls && vitis_hls -f ../../../vitis/run_cosim.tcl

# Resolve project root
set script_path [info script]
if {$script_path eq ""} {
    set script_dir [pwd]
} else {
    if {[file pathtype $script_path] eq "absolute"} {
        set script_dir [file dirname [file normalize $script_path]]
    } else {
        set script_dir [file dirname [file normalize [file join [pwd] $script_path]]]
    }
}
set proj_root [file normalize [file join $script_dir ".."]]

# Default project name (can be overridden)
set proj_name [expr {[info exists ::env(HLS_PROJ_NAME)] ? $::env(HLS_PROJ_NAME) : "yolo2_fp32_cosim"}]

puts "Opening project: $proj_name"
open_project "$proj_name"

open_solution "solution1"

# Configure co-simulation
puts "Configuring co-simulation..."
config_cosim -tool xsim -rtl verilog
# Increase timeout for long-running inference (default is 1000 seconds)
# Note: Full YOLOv2 inference can take 10-30 minutes in co-simulation
config_cosim -setup -timeout 3600

# Run co-simulation
puts "Starting co-simulation..."
puts "This may take 10-30 minutes for a full inference..."
cosim_design -rtl verilog -trace_level all

puts "Co-simulation completed!"
close_solution
close_project

