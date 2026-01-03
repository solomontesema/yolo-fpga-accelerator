#!/usr/bin/env bash
#
# Create (or refresh) a Kria `xmutil loadapp` firmware package.
#
# This script packages:
#   - <accel>.bit.bin  (required)
#   - <accel>.dtbo     (recommended; can be generated separately via XSCT/createdts)
#   - shell.json
#
# It does NOT modify your Vivado outputs (.xsa/.bit). It reads them and writes into a package folder.
#
# Recommended DTBO generation method:
#   linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md
#

set -euo pipefail

usage() {
  cat <<'EOF'
Create/refresh an xmutil firmware package folder.

Usage:
  linux_app/accel_package/create_accel_package.sh [options]

Options:
  --accel-name <name>   Accelerator/package name (default: yolov2_accel)
  --out-dir <dir>       Output package directory (default: linux_app/accel_package/<accel-name>)
  --xsa <file.xsa>      Input XSA (default: vivado/yolov2_int16/design_1_wrapper.xsa)
  --bit <file.bit>      Input bitstream (optional; auto-detected from --xsa if not provided)
  --bit-bin <file>      Pre-generated .bit.bin (optional; auto-detected if present)
  --dtbo <file.dtbo>    Copy an already-generated DTBO into the package
  --help                Show this help

Notes:
  - For .bit.bin, this script prefers an existing Vivado-generated '<top>.bit.bin' (if present).
    Otherwise it uses 'bootgen' from Vivado to convert '<top>.bit' → '<top>.bit.bin'.
  - DTBO generation is platform-dependent; the most reliable method is XSCT/createdts from the XSA.

EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"

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
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    *) printf '%s/%s\n' "$(pwd)" "$path" ;;
  esac
}

accel_name="yolov2_accel"
out_dir=""
xsa_file=""
bit_file=""
bit_bin_file=""
dtbo_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accel-name) accel_name="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --xsa) xsa_file="$2"; shift 2 ;;
    --bit) bit_file="$2"; shift 2 ;;
    --bit-bin) bit_bin_file="$2"; shift 2 ;;
    --dtbo) dtbo_file="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$xsa_file" ]]; then
  xsa_file="${project_root}/vivado/yolov2_int16/design_1_wrapper.xsa"
fi
if [[ -z "$out_dir" ]]; then
  out_dir="${script_dir}/${accel_name}"
fi

xsa_file="$(abs_path "$xsa_file")"
out_dir="$(abs_path "$out_dir")"
if [[ -n "$bit_file" ]]; then bit_file="$(abs_path "$bit_file")"; fi
if [[ -n "$bit_bin_file" ]]; then bit_bin_file="$(abs_path "$bit_bin_file")"; fi
if [[ -n "$dtbo_file" ]]; then dtbo_file="$(abs_path "$dtbo_file")"; fi

echo "========================================="
echo "KV260 Accelerator Package Creator"
echo "========================================="
echo "ACCEL: $accel_name"
echo "OUT:   $out_dir"
echo "XSA:   $xsa_file"
if [[ -n "$bit_file" ]]; then echo "BIT:   $bit_file"; fi
if [[ -n "$bit_bin_file" ]]; then echo "BIN:   $bit_bin_file"; fi
if [[ -n "$dtbo_file" ]]; then echo "DTBO:  $dtbo_file"; fi
echo ""

if [[ ! -f "$xsa_file" ]]; then
  echo "ERROR: XSA file not found: $xsa_file" >&2
  exit 1
fi

vivado_dir="$(dirname "$xsa_file")"
top_name="$(basename "$xsa_file" .xsa)"
proj_name="$(basename "$vivado_dir")"

detect_first_existing() {
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

if [[ -z "$bit_bin_file" ]]; then
  if [[ -n "$bit_file" && -f "${bit_file}.bin" ]]; then
    bit_bin_file="${bit_file}.bin"
  else
    bit_bin_file="$(detect_first_existing \
      "${vivado_dir}/${top_name}.bit.bin" \
      "${vivado_dir}/${proj_name}.runs/impl_1/${top_name}.bit.bin" \
      "$(find "$vivado_dir" -path "*/impl_1/${top_name}.bit.bin" -print -quit 2>/dev/null || true)" \
    )" || true
  fi
fi

if [[ -z "$bit_file" ]]; then
  bit_file="$(detect_first_existing \
    "${vivado_dir}/${top_name}.bit" \
    "${vivado_dir}/${proj_name}.runs/impl_1/${top_name}.bit" \
    "$(find "$vivado_dir" -path "*/impl_1/${top_name}.bit" -print -quit 2>/dev/null || true)" \
  )" || true
fi

if [[ -z "$bit_bin_file" && -z "$bit_file" ]]; then
  echo "ERROR: Could not find a Vivado bitstream for top '${top_name}' under: $vivado_dir" >&2
  echo "Hint: run the Vivado build stage first (impl_1/write_bitstream) so .bit/.bit.bin exist." >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${script_dir}/.${accel_name}.pkg.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir" 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/3] Creating package files in temp dir: $tmp_dir"

# 1) Bitstream .bit.bin
echo "[2/3] Preparing ${accel_name}.bit.bin..."
if [[ -n "$bit_bin_file" && -f "$bit_bin_file" ]]; then
  cp "$bit_bin_file" "$tmp_dir/${accel_name}.bit.bin"
  echo "      Using existing bit.bin: $bit_bin_file"
else
  if [[ -z "$bit_file" || ! -f "$bit_file" ]]; then
    echo "ERROR: BIT file not found (needed for bootgen): $bit_file" >&2
    exit 1
  fi

  bootgen_cmd=""
  if command -v bootgen >/dev/null 2>&1; then
    bootgen_cmd="bootgen"
  elif [[ -n "${XILINX_VIVADO:-}" && -x "${XILINX_VIVADO}/bin/bootgen" ]]; then
    bootgen_cmd="${XILINX_VIVADO}/bin/bootgen"
  fi

  if [[ -z "$bootgen_cmd" ]]; then
    echo "ERROR: bootgen not found (and no pre-generated .bit.bin was found)." >&2
    echo "Hint: source Vivado settings, e.g.: source /tools/Xilinx/Vivado/2024.2/settings64.sh" >&2
    exit 1
  fi

  bif_file="$tmp_dir/bitstream.bif"
  cat > "$bif_file" <<EOF
all:
{
  $bit_file
}
EOF

  echo "      Running: $bootgen_cmd -image $bif_file -arch zynqmp -process_bitstream bin -o $tmp_dir/${accel_name}.bit.bin"
  "$bootgen_cmd" -image "$bif_file" -arch zynqmp -process_bitstream bin -o "$tmp_dir/${accel_name}.bit.bin"
fi

if [[ ! -s "$tmp_dir/${accel_name}.bit.bin" ]]; then
  echo "ERROR: Failed to produce ${accel_name}.bit.bin" >&2
  exit 1
fi

# 2) DTBO (optional here; recommended via XSCT)
if [[ -n "$dtbo_file" ]]; then
  if [[ ! -f "$dtbo_file" ]]; then
    echo "ERROR: DTBO file not found: $dtbo_file" >&2
    exit 1
  fi
  cp "$dtbo_file" "$tmp_dir/${accel_name}.dtbo"
  echo "INFO: Included DTBO from: $dtbo_file"
elif [[ -f "${out_dir}/${accel_name}.dtbo" ]]; then
  cp "${out_dir}/${accel_name}.dtbo" "$tmp_dir/${accel_name}.dtbo"
  echo "INFO: Preserved existing DTBO from: ${out_dir}/${accel_name}.dtbo"
else
  echo "WARNING: No DTBO provided/found in package output dir."
  echo "         Recommended: generate DTBO from XSA using XSCT/createdts:"
  echo "         linux_app/accel_package/GENERATE_DTBO_FROM_XSA.md"
fi

# Preserve any tracked/manual overlay templates from the existing package dir (if present).
shopt -s nullglob
for f in "${out_dir}"/*.dtsi; do
  if [[ -f "$f" ]]; then
    cp "$f" "$tmp_dir/"
  fi
done
shopt -u nullglob

# 3) shell.json (preserve existing for stability if present)
echo "[3/3] Writing shell.json + deploy script..."
if [[ -f "${out_dir}/shell.json" ]]; then
  cp "${out_dir}/shell.json" "$tmp_dir/shell.json"
else
  cat > "$tmp_dir/shell.json" <<EOF
{
  "shell_type": "XRT_FLAT",
  "num_slots": 1,
  "uuid": "$(uuidgen 2>/dev/null || echo "00000000-0000-0000-0000-000000000001")",
  "pcie_config": {
    "device_id": "0x0001",
    "vendor_id": "0x10ee"
  }
}
EOF
fi

cat > "$tmp_dir/deploy_to_kv260.sh" <<DEPLOY_EOF
#!/usr/bin/env bash
set -euo pipefail

ACCEL_NAME="${accel_name}"
FIRMWARE_DIR="/lib/firmware/xilinx/\$ACCEL_NAME"

echo "Deploying \$ACCEL_NAME to KV260..."

if [ "\$EUID" -ne 0 ]; then
  echo "Please run as root (sudo)" >&2
  exit 1
fi

mkdir -p "\$FIRMWARE_DIR"

cp "\${ACCEL_NAME}.bit.bin" "\$FIRMWARE_DIR/"
if [[ -f "\${ACCEL_NAME}.dtbo" ]]; then
  cp "\${ACCEL_NAME}.dtbo" "\$FIRMWARE_DIR/"
else
  echo "WARNING: \${ACCEL_NAME}.dtbo not found in package folder (xmutil may fail)" >&2
fi
cp shell.json "\$FIRMWARE_DIR/"

echo ""
echo "Files deployed to \$FIRMWARE_DIR:"
ls -la "\$FIRMWARE_DIR/"
echo ""
echo "To load the accelerator:"
echo "  sudo xmutil unloadapp"
echo "  sudo xmutil loadapp \$ACCEL_NAME"
DEPLOY_EOF
chmod +x "$tmp_dir/deploy_to_kv260.sh"

# Swap into place (only after the package files are ready).
chmod -R u=rwX,go=rX "$tmp_dir"

mkdir -p "$(dirname "$out_dir")"
if [[ -d "$out_dir" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  backup_root="$(dirname "$out_dir")/_backups"
  mkdir -p "$backup_root"
  backup="${backup_root}/$(basename "$out_dir").${ts}"
  mv "$out_dir" "$backup"
  echo "INFO: Backed up existing package to: $backup"
fi
mv "$tmp_dir" "$out_dir"
trap - EXIT

echo ""
echo "========================================="
echo "Package ready"
echo "========================================="
echo "Output directory: $out_dir"
ls -la "$out_dir/"
