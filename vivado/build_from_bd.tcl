# KV260 YOLOv2 INT16 - Non-interactive Vivado build
#
# This TCL is meant to be run by `vivado -mode batch -source vivado/build_from_bd.tcl -tclargs ...`.
#
# It:
# - Creates a Vivado project (no GUI)
# - Adds the HLS IP repo (YOLO2_FPGA) to the catalog
# - Sources an exported BD TCL (e.g. vivado/bd/kv260_yolov2_int16_bd.tcl)
# - Generates HDL wrapper
# - Runs synthesis + implementation + bitstream
# - Exports an XSA that includes the bitstream
#
# Note: The BD TCL in this repo was generated with Vivado 2024.2, so you should
# run this script with Vivado 2024.2 to avoid version-mismatch failures.

proc usage {} {
  puts "Usage:"
  puts "  vivado -mode batch -source vivado/build_from_bd.tcl -tclargs \\"
  puts "    --bd-tcl <path/to/bd.tcl> \\"
  puts "    --proj-dir <output/project/dir> \\"
  puts "    --xsa <output.xsa> \\"
  puts "    [--ip-repo <ip_repo_dir>]... \\"
  puts "    [--proj-name <name>] \\"
  puts "    [--design-name <bd_design_name>] \\"
  puts "    [--top <top_module_name>] \\"
  puts "    [--part <part>] \\"
  puts "    [--board-part <board_part>] \\"
  puts "    [--board-connections <list>] \\"
  puts "    [--jobs <n>]"
  puts ""
  puts "Example:"
  puts "  vivado -mode batch -source vivado/build_from_bd.tcl -tclargs \\"
  puts "    --bd-tcl vivado/bd/kv260_yolov2_int16_bd.tcl \\"
  puts "    --proj-dir vivado/yolov2_int16_autogen \\"
  puts "    --xsa vivado/yolov2_int16_autogen/design_1_wrapper.xsa \\"
  puts "    --ip-repo yolo2_int16/solution1/impl/ip \\"
  puts "    --jobs 8"
}

proc require_opt {opts name} {
  if {![info exists opts($name)] || $opts($name) eq ""} {
    error "Missing required option: --$name"
  }
}

proc check_run_complete {run_name} {
  set run_obj [get_runs -quiet $run_name]
  if {$run_obj eq ""} {
    error "Vivado run '$run_name' not found"
  }
  set status [get_property STATUS $run_obj]
  puts "INFO: Run '$run_name' status: $status"
  if {[string match "*Complete*" $status] == 0 && [string match "*complete*" $status] == 0} {
    error "Run '$run_name' did not complete successfully (status: $status)"
  }
}

array set opts {
  bd_tcl ""
  proj_dir ""
  xsa ""
  proj_name ""
  design_name "design_1"
  top "design_1_wrapper"
  jobs 8
  part "xck26-sfvc784-2LV-c"
  board_part "xilinx.com:kv260_som:part0:1.4"
  board_connections "som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:1.3"
}
set ip_repos {}

for {set i 0} {$i < [llength $argv]} {incr i} {
  set arg [lindex $argv $i]
  switch -- $arg {
    --bd-tcl {
      incr i
      set opts(bd_tcl) [lindex $argv $i]
    }
    --proj-dir {
      incr i
      set opts(proj_dir) [lindex $argv $i]
    }
    --xsa {
      incr i
      set opts(xsa) [lindex $argv $i]
    }
    --ip-repo {
      incr i
      lappend ip_repos [lindex $argv $i]
    }
    --proj-name {
      incr i
      set opts(proj_name) [lindex $argv $i]
    }
    --design-name {
      incr i
      set opts(design_name) [lindex $argv $i]
    }
    --top {
      incr i
      set opts(top) [lindex $argv $i]
    }
    --jobs {
      incr i
      set opts(jobs) [lindex $argv $i]
    }
    --part {
      incr i
      set opts(part) [lindex $argv $i]
    }
    --board-part {
      incr i
      set opts(board_part) [lindex $argv $i]
    }
    --board-connections {
      incr i
      set opts(board_connections) [lindex $argv $i]
    }
    -h -
    --help {
      usage
      exit 0
    }
    default {
      usage
      error "Unknown argument: $arg"
    }
  }
}

require_opt opts bd_tcl
require_opt opts proj_dir
require_opt opts xsa

set opts(bd_tcl) [file normalize $opts(bd_tcl)]
set opts(proj_dir) [file normalize $opts(proj_dir)]
set opts(xsa) [file normalize $opts(xsa)]

if {$opts(proj_name) eq ""} {
  set opts(proj_name) [file tail $opts(proj_dir)]
}

if {![file exists $opts(bd_tcl)]} {
  error "BD TCL not found: $opts(bd_tcl)"
}

puts "INFO: Vivado version: [version -short]"
puts "INFO: BD TCL:         $opts(bd_tcl)"
puts "INFO: Project dir:    $opts(proj_dir)"
puts "INFO: Project name:   $opts(proj_name)"
puts "INFO: Output XSA:     $opts(xsa)"
puts "INFO: Part:           $opts(part)"
puts "INFO: Board part:     $opts(board_part)"
puts "INFO: Jobs:           $opts(jobs)"
if {[llength $ip_repos] > 0} {
  puts "INFO: IP repo(s):     $ip_repos"
} else {
  puts "INFO: IP repo(s):     (none provided)"
}

file mkdir $opts(proj_dir)
file mkdir [file dirname $opts(xsa)]

create_project $opts(proj_name) $opts(proj_dir) -part $opts(part) -force

catch { set_property BOARD_PART $opts(board_part) [current_project] }
catch { set_property BOARD_CONNECTIONS $opts(board_connections) [current_project] }

if {[llength $ip_repos] > 0} {
  set normalized_repos {}
  foreach repo $ip_repos {
    set repo_norm [file normalize $repo]
    if {![file exists $repo_norm]} {
      error "IP repo not found: $repo_norm"
    }
    lappend normalized_repos $repo_norm
  }
  set_property ip_repo_paths $normalized_repos [current_project]
  update_ip_catalog
}

puts "INFO: Sourcing BD TCL..."
if {[catch { source $opts(bd_tcl) } err]} {
  puts "ERROR: Failed to source BD TCL: $opts(bd_tcl)"
  puts "ERROR: $err"
  puts ""
  puts "Hint: if the BD TCL reports missing IPs, ensure you built/exported the HLS IP and passed --ip-repo."
  exit 1
}

set bd_file [get_files -quiet "${opts(design_name)}.bd"]
if {$bd_file eq ""} {
  set all_bds [get_files -quiet *.bd]
  if {[llength $all_bds] == 1} {
    set bd_file [lindex $all_bds 0]
    puts "WARNING: --design-name not found; using detected BD: $bd_file"
  } else {
    error "Could not find BD file '${opts(design_name)}.bd' (found: $all_bds)"
  }
}

puts "INFO: Generating BD output products..."
generate_target all $bd_file

puts "INFO: Creating HDL wrapper..."
set wrapper_files [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_files
update_compile_order -fileset sources_1
set_property top $opts(top) [current_fileset]

puts "INFO: Launching synth_1..."
launch_runs synth_1 -jobs $opts(jobs)
wait_on_run synth_1
check_run_complete synth_1

puts "INFO: Launching impl_1 (through write_bitstream)..."
launch_runs impl_1 -to_step write_bitstream -jobs $opts(jobs)
wait_on_run impl_1
check_run_complete impl_1

puts "INFO: Opening implemented design..."
open_run impl_1

puts "INFO: Exporting XSA (include bitstream)..."
if {[llength [info commands write_hw_platform]] > 0} {
  write_hw_platform -fixed -include_bit -force -file $opts(xsa)
} elseif {[llength [info commands write_xsa]] > 0} {
  write_xsa -include_bit -force $opts(xsa)
} else {
  error "Neither write_hw_platform nor write_xsa is available in this Vivado build"
}

puts "INFO: Done."
puts "INFO: XSA written to: $opts(xsa)"

exit 0

