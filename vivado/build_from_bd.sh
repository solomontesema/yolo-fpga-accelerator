#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a Vivado project from an exported Block Design TCL (no GUI), then generate bitstream + XSA.

Usage:
  vivado/build_from_bd.sh \
    --bd-tcl <path/to/bd.tcl> \
    --proj-dir <output/project/dir> \
    --xsa <output.xsa> \
    [--ip-repo <ip_repo_dir>]... \
    [--jobs <n>] \
    [--vivado <vivado_bin>] \
    [--proj-name <name>] \
    [--design-name <bd_design_name>] \
    [--top <top_module_name>] \
    [--part <part>] \
    [--board-part <board_part>] \
    [--board-connections <value>]

Notes:
  - The BD TCL in this repo (kv260_yolov2_int16_bd.tcl) was generated with Vivado 2024.2,
    so use Vivado 2024.2 to run this script.
  - The design uses a custom HLS IP: xilinx.com:hls:YOLO2_FPGA:1.0.
    You must provide an IP repository that contains this IP (or let this script auto-detect it).

Examples:
  # Auto-detect IP repo from yolo2_int16/solution1/impl/ip (if it exists)
  vivado/build_from_bd.sh \
    --bd-tcl vivado/bd/kv260_yolov2_int16_bd.tcl \
    --proj-dir vivado/yolov2_int16_autogen \
    --xsa vivado/yolov2_int16_autogen/design_1_wrapper.xsa

  # Explicit IP repo + more parallel jobs
  vivado/build_from_bd.sh \
    --bd-tcl vivado/bd/kv260_yolov2_int16_bd.tcl \
    --proj-dir vivado/yolov2_int16_autogen \
    --xsa vivado/yolov2_int16_autogen/design_1_wrapper.xsa \
    --ip-repo yolo2_int16/solution1/impl/ip \
    --jobs 8
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

abs_path() {
  local path="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$path"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' "$path"
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
    return
  fi
  # Fallback (best effort): prefix relative paths with current directory.
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    *) printf '%s/%s\n' "$(pwd)" "$path" ;;
  esac
}

bd_tcl=""
proj_dir=""
xsa_out=""
vivado_bin="${VIVADO_BIN:-vivado}"
jobs=""
proj_name=""
design_name=""
top_name=""
part=""
board_part=""
board_connections=""
ip_repos=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bd-tcl) bd_tcl="$2"; shift 2 ;;
    --proj-dir) proj_dir="$2"; shift 2 ;;
    --xsa) xsa_out="$2"; shift 2 ;;
    --ip-repo) ip_repos+=("$2"); shift 2 ;;
    --vivado) vivado_bin="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --proj-name) proj_name="$2"; shift 2 ;;
    --design-name) design_name="$2"; shift 2 ;;
    --top) top_name="$2"; shift 2 ;;
    --part) part="$2"; shift 2 ;;
    --board-part) board_part="$2"; shift 2 ;;
    --board-connections) board_connections="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$bd_tcl" || -z "$proj_dir" || -z "$xsa_out" ]]; then
  echo "ERROR: --bd-tcl, --proj-dir, and --xsa are required." >&2
  usage
  exit 2
fi

if ! command -v "$vivado_bin" >/dev/null 2>&1; then
  echo "ERROR: Vivado executable not found: $vivado_bin" >&2
  echo "Hint: source the Vivado settings script, e.g.: source /tools/Xilinx/Vivado/2024.2/settings64.sh" >&2
  exit 2
fi

if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    jobs="$(nproc)"
  else
    jobs="8"
  fi
fi

bd_tcl="$(abs_path "$bd_tcl")"
proj_dir="$(abs_path "$proj_dir")"
xsa_out="$(abs_path "$xsa_out")"

if [[ ${#ip_repos[@]} -eq 0 ]]; then
  default_repo="${repo_root}/yolo2_int16/solution1/impl/ip"
  if [[ -f "${default_repo}/component.xml" ]]; then
    ip_repos+=("$default_repo")
  fi
fi

work_dir="${proj_dir}/_vivado_batch"
mkdir -p "$work_dir"

echo "INFO: Vivado:         $vivado_bin"
echo "INFO: BD TCL:         $bd_tcl"
echo "INFO: Project dir:    $proj_dir"
echo "INFO: Output XSA:     $xsa_out"
echo "INFO: Jobs:           $jobs"
if [[ ${#ip_repos[@]} -gt 0 ]]; then
  printf "INFO: IP repo(s):     %s\n" "${ip_repos[*]}"
else
  echo "INFO: IP repo(s):     (none)  <-- build will fail if YOLO2_FPGA IP is not available"
fi
echo "INFO: Work dir:       $work_dir"

tcl_args=(--bd-tcl "$bd_tcl" --proj-dir "$proj_dir" --xsa "$xsa_out" --jobs "$jobs")

if [[ -n "$proj_name" ]]; then tcl_args+=(--proj-name "$proj_name"); fi
if [[ -n "$design_name" ]]; then tcl_args+=(--design-name "$design_name"); fi
if [[ -n "$top_name" ]]; then tcl_args+=(--top "$top_name"); fi
if [[ -n "$part" ]]; then tcl_args+=(--part "$part"); fi
if [[ -n "$board_part" ]]; then tcl_args+=(--board-part "$board_part"); fi
if [[ -n "$board_connections" ]]; then tcl_args+=(--board-connections "$board_connections"); fi
for r in "${ip_repos[@]}"; do tcl_args+=(--ip-repo "$r"); done

pushd "$work_dir" >/dev/null
"$vivado_bin" -mode batch -source "${repo_root}/vivado/build_from_bd.tcl" -tclargs "${tcl_args[@]}"
popd >/dev/null

echo "INFO: Build finished."
echo "INFO: XSA: $xsa_out"
echo "INFO: Bitstream should be under: ${proj_dir}/${proj_name:-$(basename "$proj_dir")}.runs/impl_1/"
