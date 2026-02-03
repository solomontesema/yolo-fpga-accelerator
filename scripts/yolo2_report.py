#!/usr/bin/env python3
"""
YOLO2 Performance Report Tool

A lightweight CLI for tracking HLS/Vivado/KV260 performance across iterations.
Uses only Python stdlib (no external dependencies required for core features).

Usage:
    python3 scripts/yolo2_report.py --help
    python3 scripts/yolo2_report.py init
    python3 scripts/yolo2_report.py run --label baseline --hls-report-dir yolo2_int16/solution1/syn/report
    python3 scripts/yolo2_report.py list
    python3 scripts/yolo2_report.py compare reports/<runA>/ reports/<runB>/
"""

import argparse
import copy
import fcntl
import getpass
import json
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_FILE = "yolo2_report.local.json"
DEFAULT_REPORTS_DIR = "reports"

CONFIG_TEMPLATE = {
    "reports_dir": DEFAULT_REPORTS_DIR,
    "hls_report_dir": "yolo2_int16/solution1/syn/report",
    "vivado_report_dir": "",
    "kv260": {
        "enabled": False,
        "host": "ubuntu@kria",
        "port": 22,
        "identity_file": "",
        "remote_cmd_template": "./linux_app/start_yolo.sh -v 1 -i /home/ubuntu/test_images/dog.jpg --max-frames 10",
        "force_tty": True,
        "password_env": "",
        "timeout_s": 300
    }
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_git_info() -> Dict[str, Any]:
    """Get current git commit, branch, and dirty state."""
    info = {"commit": None, "branch": None, "dirty": None}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:12]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0
    except Exception:
        pass
    return info


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def format_number(value: Any, precision: int = 2) -> str:
    """Format a number for display, handling None/invalid."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)


# ---------------------------------------------------------------------------
# HLS XML Parser
# ---------------------------------------------------------------------------

def parse_hls_csynth_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse a Vitis HLS *_csynth.xml file and extract key metrics."""
    result = {
        "file": str(xml_path),
        "parsed": False,
        "error": None,
        "timing": {},
        "utilization": {},
        "available_resources": {},
        "user_assignments": {},
        "axi_ports": {}
    }

    if not xml_path.exists():
        result["error"] = f"File not found: {xml_path}"
        return result

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # User assignments
        user_assign = root.find("UserAssignments")
        if user_assign is not None:
            result["user_assignments"] = {
                "part": user_assign.findtext("Part", ""),
                "top_model": user_assign.findtext("TopModelName", ""),
                "target_clock_ns": safe_float(user_assign.findtext("TargetClockPeriod")),
                "clock_uncertainty_ns": safe_float(user_assign.findtext("ClockUncertainty")),
                "product_family": user_assign.findtext("ProductFamily", "")
            }

        # Performance estimates - timing
        perf = root.find("PerformanceEstimates")
        if perf is not None:
            timing = perf.find("SummaryOfTimingAnalysis")
            if timing is not None:
                result["timing"] = {
                    "estimated_clock_ns": safe_float(timing.findtext("EstimatedClockPeriod"))
                }

        # Area estimates
        area = root.find("AreaEstimates")
        if area is not None:
            resources = area.find("Resources")
            if resources is not None:
                result["utilization"] = {
                    "BRAM_18K": safe_int(resources.findtext("BRAM_18K")),
                    "DSP": safe_int(resources.findtext("DSP")),
                    "FF": safe_int(resources.findtext("FF")),
                    "LUT": safe_int(resources.findtext("LUT")),
                    "URAM": safe_int(resources.findtext("URAM"))
                }
            available = area.find("AvailableResources")
            if available is not None:
                result["available_resources"] = {
                    "BRAM_18K": safe_int(available.findtext("BRAM_18K")),
                    "DSP": safe_int(available.findtext("DSP")),
                    "FF": safe_int(available.findtext("FF")),
                    "LUT": safe_int(available.findtext("LUT")),
                    "URAM": safe_int(available.findtext("URAM"))
                }

        # Interface summary - extract AXI port widths
        interface = root.find("InterfaceSummary")
        if interface is not None:
            for port in interface.findall("RtlPorts"):
                name = port.findtext("name", "")
                bits = safe_int(port.findtext("Bits"))
                protocol = port.findtext("IOProtocol", "")
                obj = port.findtext("Object", "")

                # Track m_axi data ports
                if protocol == "m_axi" and ("RDATA" in name or "WDATA" in name):
                    if obj not in result["axi_ports"]:
                        result["axi_ports"][obj] = {}
                    if "RDATA" in name:
                        result["axi_ports"][obj]["read_width"] = bits
                    elif "WDATA" in name:
                        result["axi_ports"][obj]["write_width"] = bits

        result["parsed"] = True

    except ET.ParseError as e:
        result["error"] = f"XML parse error: {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result


def parse_hls_report_dir(report_dir: Path) -> Dict[str, Any]:
    """Parse all HLS csynth.xml files in a directory."""
    result = {
        "directory": str(report_dir),
        "parsed": False,
        "error": None,
        "top_level": None,
        "modules": {},
        "summary": {}
    }

    if not report_dir.exists():
        result["error"] = f"Directory not found: {report_dir}"
        return result

    # Find top-level report (YOLO2_FPGA_csynth.xml or similar)
    top_candidates = list(report_dir.glob("YOLO2_FPGA_csynth.xml"))
    if not top_candidates:
        top_candidates = list(report_dir.glob("*_csynth.xml"))

    if top_candidates:
        # Parse the top-level (largest/first match by convention)
        for candidate in sorted(top_candidates, key=lambda p: p.name):
            if "Pipeline" not in candidate.name:
                parsed = parse_hls_csynth_xml(candidate)
                if parsed["parsed"]:
                    result["top_level"] = parsed
                    break

        # If no non-pipeline found, use first
        if result["top_level"] is None and top_candidates:
            result["top_level"] = parse_hls_csynth_xml(top_candidates[0])

    # Parse other modules (optional, for detailed analysis)
    for xml_file in sorted(report_dir.glob("*_csynth.xml")):
        module_name = xml_file.stem.replace("_csynth", "")
        if result["top_level"] and xml_file.name == Path(result["top_level"]["file"]).name:
            continue
        result["modules"][module_name] = parse_hls_csynth_xml(xml_file)

    # Build summary from top-level
    if result["top_level"] and result["top_level"]["parsed"]:
        top = result["top_level"]
        result["summary"] = {
            "target_clock_ns": top["user_assignments"].get("target_clock_ns"),
            "estimated_clock_ns": top["timing"].get("estimated_clock_ns"),
            "target_clock_mhz": 1000.0 / top["user_assignments"].get("target_clock_ns", 1) if top["user_assignments"].get("target_clock_ns") else None,
            "estimated_clock_mhz": 1000.0 / top["timing"].get("estimated_clock_ns", 1) if top["timing"].get("estimated_clock_ns") else None,
            "part": top["user_assignments"].get("part"),
            "utilization": top["utilization"],
            "available": top["available_resources"],
            "axi_ports": top["axi_ports"]
        }
        result["parsed"] = True

    return result


# ---------------------------------------------------------------------------
# Vivado Report Parsers
# ---------------------------------------------------------------------------

def parse_vivado_timing_summary(rpt_path: Path) -> Dict[str, Any]:
    """Parse Vivado report_timing_summary output."""
    result = {
        "file": str(rpt_path),
        "parsed": False,
        "error": None,
        "wns_ns": None,
        "tns_ns": None,
        "whs_ns": None,
        "ths_ns": None,
        "target_clock_ns": None,
        "clock_name": None,
        "target_clock_mhz": None
    }

    if not rpt_path.exists():
        result["error"] = f"File not found: {rpt_path}"
        return result

    try:
        content = rpt_path.read_text(errors="replace")
        lines = content.splitlines()

        # Vivado 2024.x uses table formats. Prefer those, fall back to older key:value style.
        for i, line in enumerate(lines):
            if "Design Timing Summary" not in line:
                continue

            header_idx = None
            for j in range(i, min(i + 80, len(lines))):
                if "WNS(ns)" in lines[j] and "TNS(ns)" in lines[j] and "WHS(ns)" in lines[j]:
                    header_idx = j
                    break
            if header_idx is None:
                continue

            for k in range(header_idx + 1, min(header_idx + 12, len(lines))):
                tokens = lines[k].split()
                if not tokens:
                    continue
                if not re.match(r"^-?\d", tokens[0]):
                    continue

                # Expected token order:
                # WNS, TNS, TNS_Fail, TNS_Total, WHS, THS, ...
                if len(tokens) >= 6:
                    result["wns_ns"] = safe_float(tokens[0], default=None)
                    result["tns_ns"] = safe_float(tokens[1], default=None)
                    result["whs_ns"] = safe_float(tokens[4], default=None)
                    result["ths_ns"] = safe_float(tokens[5], default=None)
                break

            break

        # Fallback: key:value format
        if result["wns_ns"] is None:
            wns_match = re.search(r"WNS\(ns\)\s*:\s*([-\d.]+)", content)
            if wns_match:
                result["wns_ns"] = safe_float(wns_match.group(1), default=None)
        if result["tns_ns"] is None:
            tns_match = re.search(r"TNS\(ns\)\s*:\s*([-\d.]+)", content)
            if tns_match:
                result["tns_ns"] = safe_float(tns_match.group(1), default=None)
        if result["whs_ns"] is None:
            whs_match = re.search(r"WHS\(ns\)\s*:\s*([-\d.]+)", content)
            if whs_match:
                result["whs_ns"] = safe_float(whs_match.group(1), default=None)
        if result["ths_ns"] is None:
            ths_match = re.search(r"THS\(ns\)\s*:\s*([-\d.]+)", content)
            if ths_match:
                result["ths_ns"] = safe_float(ths_match.group(1), default=None)

        # Clock Summary table (preferred)
        in_clock_summary = False
        clock_candidates: List[Tuple[str, Optional[float], Optional[float]]] = []
        for line in lines:
            if re.search(r"\bClock Summary\b", line):
                in_clock_summary = True
                continue
            if not in_clock_summary:
                continue
            if not line.strip():
                if clock_candidates:
                    break
                continue
            m = re.match(r"^\s*(\S+)\s+\{[^}]*\}\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m:
                clk = m.group(1)
                period_ns = safe_float(m.group(2), default=None)
                freq_mhz = safe_float(m.group(3), default=None)
                clock_candidates.append((clk, period_ns, freq_mhz))

        # Choose clock: prefer clk_pl_0, else first.
        if clock_candidates:
            chosen = None
            for cand in clock_candidates:
                if cand[0] == "clk_pl_0":
                    chosen = cand
                    break
            if chosen is None:
                chosen = clock_candidates[0]
            result["clock_name"] = chosen[0]
            result["target_clock_ns"] = chosen[1]
            result["target_clock_mhz"] = chosen[2] if chosen[2] is not None else (
                (1000.0 / chosen[1]) if (chosen[1] is not None and chosen[1] > 0) else None
            )

        # Older format: "Requirement: Xns"
        if result["target_clock_ns"] is None:
            clk_match = re.search(r"Requirement:\s*([\d.]+)ns", content)
            if clk_match:
                result["target_clock_ns"] = safe_float(clk_match.group(1), default=None)
                if result["target_clock_ns"] and result["target_clock_ns"] > 0:
                    result["target_clock_mhz"] = 1000.0 / result["target_clock_ns"]

        result["parsed"] = any(
            v is not None for v in [
                result["wns_ns"],
                result["tns_ns"],
                result["whs_ns"],
                result["ths_ns"],
                result["target_clock_ns"],
                result["target_clock_mhz"],
            ]
        )

    except Exception as e:
        result["error"] = f"Parse error: {e}"

    return result


def parse_vivado_utilization(rpt_path: Path) -> Dict[str, Any]:
    """Parse Vivado report_utilization output."""
    result = {
        "file": str(rpt_path),
        "parsed": False,
        "error": None,
        "utilization": {},
        "available": {},
        "util_pct": {}
    }

    if not rpt_path.exists():
        result["error"] = f"File not found: {rpt_path}"
        return result

    try:
        content = rpt_path.read_text(errors="replace")
        lines = content.splitlines()

        # Table format (Vivado 2024.x):
        # | Site Type | Used | Fixed | Prohibited | Available | Util% |
        row_to_key = {
            "CLB LUTs": "LUT",
            "CLB Registers": "FF",
            "Block RAM Tile": "BRAM",
            "DSPs": "DSP",
            "URAM": "URAM",
        }

        for line in lines:
            if not line.lstrip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if len(cells) < 5:
                continue
            name = cells[0]
            if name not in row_to_key:
                continue

            key = row_to_key[name]
            used = safe_float(cells[1], default=None)
            available = safe_float(cells[4], default=None) if len(cells) >= 5 else None
            util_pct = safe_float(cells[5], default=None) if len(cells) >= 6 else None

            if used is not None:
                result["utilization"][key] = used
            if available is not None:
                result["available"][key] = available
            if util_pct is not None:
                result["util_pct"][key] = util_pct

        # Pattern: | Resource | Used | Available | Util% |
        # Look for CLB LUTs, CLB Registers, Block RAM, DSPs
        patterns = {
            "LUT": r"\|\s*CLB LUTs[^|]*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|",
            "FF": r"\|\s*CLB Registers[^|]*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|",
            "BRAM": r"\|\s*Block RAM Tile[^|]*\|\s*([\d.]+)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|",
            "DSP": r"\|\s*DSPs[^|]*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|",
            "URAM": r"\|\s*URAM[^|]*\|\s*([\d.]+)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|"
        }

        if not result["utilization"]:
            for resource, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    result["utilization"][resource] = safe_float(match.group(1), default=None)
                    result["available"][resource] = safe_float(match.group(2), default=None)

        # Alternative simpler patterns
        if not result["utilization"]:
            # Try simpler line-based parsing
            for line in content.split("\n"):
                if "LUT" in line and "%" in line:
                    nums = re.findall(r"(\d+)", line)
                    if len(nums) >= 2:
                        result["utilization"]["LUT"] = safe_int(nums[0])
                        result["available"]["LUT"] = safe_int(nums[1])

        result["parsed"] = bool(result["utilization"])

    except Exception as e:
        result["error"] = f"Parse error: {e}"

    return result


def parse_vivado_power(rpt_path: Path) -> Dict[str, Any]:
    """Parse Vivado report_power output."""
    result = {
        "file": str(rpt_path),
        "parsed": False,
        "error": None,
        "total_power_w": None,
        "dynamic_power_w": None,
        "static_power_w": None,
        "ip_name": "YOLO2_FPGA_0",
        "ip_power_w": None
    }

    if not rpt_path.exists():
        result["error"] = f"File not found: {rpt_path}"
        return result

    try:
        content = rpt_path.read_text(errors="replace")
        lines = content.splitlines()

        def parse_cell_number(value: str) -> Optional[float]:
            if value is None:
                return None
            v = value.strip()
            if not v:
                return None
            v = v.lstrip("<").rstrip("*").strip()
            try:
                return float(v)
            except ValueError:
                return None

        # Table format (Vivado 2024.x)
        table_patterns = {
            "total_power_w": r"^\|\s*Total On-Chip Power\s*\(W\)\s*\|\s*([^|]+?)\s*\|",
            "dynamic_power_w": r"^\|\s*Dynamic\s*\(W\)\s*\|\s*([^|]+?)\s*\|",
            "static_power_w": r"^\|\s*Device Static\s*\(W\)\s*\|\s*([^|]+?)\s*\|",
        }
        for line in lines:
            for key, pattern in table_patterns.items():
                m = re.match(pattern, line)
                if m:
                    result[key] = parse_cell_number(m.group(1))

        # By Hierarchy table for IP-level power (if present)
        target_ip = result.get("ip_name") or "YOLO2_FPGA_0"
        for line in lines:
            if not line.lstrip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().split("|")[1:-1]]
            if len(cells) != 2:
                continue
            name, power = cells[0], cells[1]
            if not name:
                continue
            if target_ip in name:
                result["ip_power_w"] = parse_cell_number(power)
                break

        # Fallback regex for older formats
        if result["total_power_w"] is None:
            total_match = re.search(r"Total On-Chip Power[^:]*:\s*([\d.]+)", content)
            if total_match:
                result["total_power_w"] = safe_float(total_match.group(1), default=None)
        if result["dynamic_power_w"] is None:
            dynamic_match = re.search(r"Dynamic[^:]*:\s*([\d.]+)", content)
            if dynamic_match:
                result["dynamic_power_w"] = safe_float(dynamic_match.group(1), default=None)
        if result["static_power_w"] is None:
            static_match = re.search(r"Device Static[^:]*:\s*([\d.]+)", content)
            if static_match:
                result["static_power_w"] = safe_float(static_match.group(1), default=None)

        result["parsed"] = any(
            v is not None for v in [
                result["total_power_w"],
                result["dynamic_power_w"],
                result["static_power_w"],
                result["ip_power_w"],
            ]
        )

    except Exception as e:
        result["error"] = f"Parse error: {e}"

    return result


def parse_vivado_report_dir(report_dir: Path) -> Dict[str, Any]:
    """Parse all Vivado reports in a directory."""
    result = {
        "directory": str(report_dir),
        "parsed": False,
        "error": None,
        "timing": None,
        "utilization": None,
        "power": None,
        "summary": {}
    }

    if not report_dir.exists():
        result["error"] = f"Directory not found: {report_dir}"
        return result

    def pick_first(preferred_globs: List[str], exclude_substrings: Optional[List[str]] = None) -> Optional[Path]:
        exclude_substrings = exclude_substrings or []
        for pattern in preferred_globs:
            candidates = [p for p in report_dir.glob(pattern)]
            candidates = [
                p for p in candidates
                if not any(excl in p.name for excl in exclude_substrings)
            ]
            candidates = sorted(candidates)
            if candidates:
                return candidates[0]
        return None

    # Find timing report
    timing_rpt = pick_first([
        "*timing_summary*_routed.rpt",
        "*timing_summary*.rpt",
        "*timing*_routed.rpt",
        "*timing*.rpt",
    ])
    if timing_rpt:
        result["timing"] = parse_vivado_timing_summary(timing_rpt)

    # Find utilization report
    util_rpt = pick_first(
        [
            "*utilization*_routed.rpt",
            "*utilization*_placed.rpt",
            "*utilization*.rpt",
        ],
        exclude_substrings=["clock_utilization"],
    )
    if util_rpt:
        result["utilization"] = parse_vivado_utilization(util_rpt)

    # Find power report
    power_rpt = pick_first([
        "*power*_routed.rpt",
        "*power*.rpt",
    ])
    if power_rpt:
        result["power"] = parse_vivado_power(power_rpt)

    # Build summary
    if result["timing"] and result["timing"]["parsed"]:
        result["summary"]["wns_ns"] = result["timing"]["wns_ns"]
        result["summary"]["tns_ns"] = result["timing"]["tns_ns"]
        result["summary"]["whs_ns"] = result["timing"]["whs_ns"]
        result["summary"]["ths_ns"] = result["timing"]["ths_ns"]
        result["summary"]["clock_name"] = result["timing"]["clock_name"]
        result["summary"]["target_clock_ns"] = result["timing"]["target_clock_ns"]
        result["summary"]["target_clock_mhz"] = result["timing"]["target_clock_mhz"]
        if result["timing"]["target_clock_ns"] is not None and result["timing"]["wns_ns"] is not None:
            min_period = result["timing"]["target_clock_ns"] - result["timing"]["wns_ns"]
            result["summary"]["fmax_est_mhz"] = (1000.0 / min_period) if min_period and min_period > 0 else None
        result["parsed"] = True

    if result["utilization"] and result["utilization"]["parsed"]:
        result["summary"]["utilization"] = result["utilization"]["utilization"]
        result["summary"]["available"] = result["utilization"]["available"]
        result["summary"]["util_pct"] = result["utilization"].get("util_pct", {})
        result["parsed"] = True

    if result["power"] and result["power"]["parsed"]:
        result["summary"]["total_power_w"] = result["power"]["total_power_w"]
        result["summary"]["dynamic_power_w"] = result["power"]["dynamic_power_w"]
        result["summary"]["static_power_w"] = result["power"]["static_power_w"]
        result["summary"]["ip_name"] = result["power"].get("ip_name")
        result["summary"]["ip_power_w"] = result["power"].get("ip_power_w")
        result["parsed"] = True

    return result


# ---------------------------------------------------------------------------
# KV260 Log Parser
# ---------------------------------------------------------------------------

def parse_kv260_log(log_content: str) -> Dict[str, Any]:
    """Parse yolo2_linux stdout for inference timing lines."""
    result = {
        "parsed": False,
        "error": None,
        "inference_times_ms": [],
        "frame_count": 0,
        "stats": {}
    }

    if not log_content:
        result["error"] = "Empty log content"
        return result

    # Match examples:
    # - "Inference time: 1234.56 ms"
    # - "Frame 1 (infer 1) inference time: 1234.56 ms"
    matches = re.findall(r"inference time:\s*([\d.]+)\s*ms", log_content, flags=re.IGNORECASE)
    times: List[float] = []
    for m in matches:
        val = safe_float(m, default=None)
        if val is not None:
            times.append(val)

    if times:
        result["inference_times_ms"] = times
        result["frame_count"] = len(times)

        # Compute stats
        times_sorted = sorted(times)
        result["stats"] = {
            "count": len(times),
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "median_ms": times_sorted[len(times) // 2],
            "p90_ms": times_sorted[int(len(times) * 0.9)] if len(times) >= 10 else times_sorted[-1]
        }
        result["stats"]["fps_from_mean"] = 1000.0 / result["stats"]["mean_ms"] if result["stats"]["mean_ms"] > 0 else 0
        result["stats"]["fps_from_median"] = 1000.0 / result["stats"]["median_ms"] if result["stats"]["median_ms"] > 0 else 0
        result["parsed"] = True
    else:
        result["error"] = "No inference timing lines found"

    return result


def run_kv260_ssh(config: Dict[str, Any], remote_cmd: str) -> Tuple[str, str, int]:
    """Run a command on KV260 via SSH and return (stdout, stderr, returncode).

    By default this uses BatchMode=yes (key-based auth only). To support password auth,
    pass a password via config kv260.password / kv260.password_env or via CLI options.
    """
    kv260_cfg = config.get("kv260", {})
    host = kv260_cfg.get("host", "ubuntu@kria")
    port = kv260_cfg.get("port", 22)
    identity = kv260_cfg.get("identity_file", "")
    password = kv260_cfg.get("password") or None
    password_env = kv260_cfg.get("password_env") or None
    timeout_s = int(kv260_cfg.get("timeout_s", 300))
    force_tty = bool(kv260_cfg.get("force_tty", True))

    if password is None and password_env:
        password = os.environ.get(str(password_env)) or None

    ssh_cmd = ["ssh"]
    if port != 22:
        ssh_cmd.extend(["-p", str(port)])
    if identity:
        ssh_cmd.extend(["-i", os.path.expanduser(identity)])
    if force_tty:
        # Needed for commands that use sudo and prompt for a password.
        ssh_cmd.append("-tt")
    ssh_cmd.extend([
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
    ])
    if password is None:
        ssh_cmd.extend(["-o", "BatchMode=yes"])
    else:
        ssh_cmd.extend(["-o", "BatchMode=no"])
    ssh_cmd.append(host)
    ssh_cmd.append(remote_cmd)

    try:
        if password is None:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return result.stdout, result.stderr, result.returncode

        stdout, stderr, rc = run_ssh_with_password(
            ssh_cmd,
            password=password,
            timeout_s=timeout_s,
            redact=password,
        )
        return stdout, stderr, rc
    except subprocess.TimeoutExpired:
        return "", "SSH command timed out", -1
    except Exception as e:
        return "", str(e), -1


def run_ssh_with_password(
    ssh_cmd: List[str],
    password: str,
    timeout_s: int = 300,
    redact: Optional[str] = None,
    max_password_prompts: int = 10,
) -> Tuple[str, str, int]:
    """Run `ssh_cmd` via a local PTY, answering password prompts programmatically.

    Notes:
    - stdout/stderr are merged due to PTY use.
    - `redact` (if provided) is stripped from the captured output as a safety net.
    """
    if not password:
        raise ValueError("Empty password provided")

    try:
        master_fd, slave_fd = pty.openpty()
    except OSError as e:
        return "", f"Failed to allocate PTY for password auth ({e}). Use key-based SSH or install sshpass.", -1

    def preexec() -> None:
        # Make the PTY the controlling TTY for the child, so ssh/sudo read prompts from it
        # instead of trying to use our real terminal (/dev/tty).
        os.setsid()
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        except Exception:
            pass

    proc = subprocess.Popen(
        ssh_cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        preexec_fn=preexec,
    )
    os.close(slave_fd)

    out = bytearray()
    tail = b""
    # Match common interactive prompts:
    # - ssh:   "ubuntu@kria's password:"
    # - sudo:  "[sudo] password for ubuntu:"
    # - keys:  "Enter passphrase for key ..."
    prompt_re = re.compile(br"(?i)(?:password|passphrase)[^:\n]*:")
    prompts_seen = 0

    start = time.monotonic()
    try:
        while True:
            if time.monotonic() - start > timeout_s:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                text = out.decode(errors="replace")
                if redact:
                    text = text.replace(redact, "<redacted>")
                return text, "SSH command timed out", -1

            r, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""
                if data:
                    out.extend(data)
                    tail = (tail + data)[-2048:]
                    if prompts_seen < max_password_prompts and prompt_re.search(tail):
                        os.write(master_fd, (password + "\n").encode())
                        prompts_seen += 1

            if proc.poll() is not None:
                # Drain remaining output quickly.
                while True:
                    r2, _, _ = select.select([master_fd], [], [], 0)
                    if master_fd not in r2:
                        break
                    try:
                        data2 = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not data2:
                        break
                    out.extend(data2)
                break

        rc = proc.wait(timeout=2)
        text = out.decode(errors="replace")
        if redact:
            text = text.replace(redact, "<redacted>")
        return text, "", rc
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        text = out.decode(errors="replace")
        if redact:
            text = text.replace(redact, "<redacted>")
        return text, "Interrupted by user", 130
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Report Bundle Management
# ---------------------------------------------------------------------------

def create_bundle_dir(reports_dir: Path, label: str) -> Path:
    """Create a new report bundle directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_label = re.sub(r"[^\w\-]", "_", label)
    bundle_name = f"{timestamp}_{safe_label}"
    bundle_path = reports_dir / bundle_name

    bundle_path.mkdir(parents=True, exist_ok=True)
    (bundle_path / "hls").mkdir(exist_ok=True)
    (bundle_path / "vivado").mkdir(exist_ok=True)
    (bundle_path / "kv260").mkdir(exist_ok=True)

    return bundle_path


def write_meta_json(bundle_path: Path, label: str, note: str, input_paths: Dict[str, str]) -> Dict[str, Any]:
    """Write meta.json with run metadata."""
    git_info = get_git_info()

    meta = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "note": note,
        "git": git_info,
        "input_paths": input_paths,
        "tool_version": "1.0.0"
    }

    meta_path = bundle_path / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def write_metrics_json(bundle_path: Path, metrics: Dict[str, Any]) -> None:
    """Write metrics.json with all parsed metrics."""
    metrics_path = bundle_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def write_summary_md(bundle_path: Path, meta: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Write summary.md with human-readable summary."""
    lines = []
    lines.append(f"# Report: {meta['label']}")
    lines.append("")
    lines.append(f"**Date:** {meta['timestamp'][:19]}")
    if meta["git"]["commit"]:
        dirty = " (dirty)" if meta["git"]["dirty"] else ""
        lines.append(f"**Git:** {meta['git']['commit']}{dirty} on {meta['git']['branch']}")
    if meta["note"]:
        lines.append(f"**Note:** {meta['note']}")
    lines.append("")

    # KV260 section
    kv260 = metrics.get("kv260") or {}
    if kv260.get("parsed"):
        stats = kv260.get("stats", {})
        lines.append("## KV260 Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Frames | {stats.get('count', 'N/A')} |")
        lines.append(f"| Mean | {format_number(stats.get('mean_ms'))} ms |")
        lines.append(f"| Median | {format_number(stats.get('median_ms'))} ms |")
        lines.append(f"| P90 | {format_number(stats.get('p90_ms'))} ms |")
        lines.append(f"| FPS (median) | {format_number(stats.get('fps_from_median'))} |")
        lines.append("")

    # HLS section
    hls = metrics.get("hls") or {}
    if hls.get("parsed"):
        summary = hls.get("summary", {})
        lines.append("## HLS Synthesis")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Target Clock | {format_number(summary.get('target_clock_ns'))} ns ({format_number(summary.get('target_clock_mhz'))} MHz) |")
        lines.append(f"| Est. Clock | {format_number(summary.get('estimated_clock_ns'))} ns ({format_number(summary.get('estimated_clock_mhz'))} MHz) |")
        lines.append(f"| Part | {summary.get('part', 'N/A')} |")
        lines.append("")

        util = summary.get("utilization", {})
        avail = summary.get("available", {})
        if util:
            lines.append("### Utilization")
            lines.append("")
            lines.append("| Resource | Used | Available | % |")
            lines.append("|----------|------|-----------|---|")
            for res in ["LUT", "FF", "DSP", "BRAM_18K", "URAM"]:
                used = util.get(res, 0)
                total = avail.get(res, 1)
                pct = (used / total * 100) if total > 0 else 0
                lines.append(f"| {res} | {used} | {total} | {pct:.1f}% |")
            lines.append("")

        axi = summary.get("axi_ports", {})
        if axi:
            lines.append("### AXI Port Widths")
            lines.append("")
            lines.append("| Port | Read Width | Write Width |")
            lines.append("|------|------------|-------------|")
            for port, info in axi.items():
                lines.append(f"| {port} | {info.get('read_width', 'N/A')} bits | {info.get('write_width', 'N/A')} bits |")
            lines.append("")

    # Vivado section
    vivado = metrics.get("vivado") or {}
    if vivado.get("parsed"):
        summary = vivado.get("summary", {})
        lines.append("## Vivado Implementation")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        if summary.get("clock_name"):
            lines.append(f"| Clock | {summary.get('clock_name')} |")
        if summary.get("target_clock_ns") is not None or summary.get("target_clock_mhz") is not None:
            lines.append(f"| Target Clock | {format_number(summary.get('target_clock_ns'), 3)} ns ({format_number(summary.get('target_clock_mhz'), 3)} MHz) |")
        if summary.get("wns_ns") is not None:
            lines.append(f"| WNS | {format_number(summary.get('wns_ns'), 3)} ns |")
        if summary.get("tns_ns") is not None:
            lines.append(f"| TNS | {format_number(summary.get('tns_ns'), 3)} ns |")
        if summary.get("whs_ns") is not None:
            lines.append(f"| WHS | {format_number(summary.get('whs_ns'), 3)} ns |")
        if summary.get("ths_ns") is not None:
            lines.append(f"| THS | {format_number(summary.get('ths_ns'), 3)} ns |")
        if summary.get("fmax_est_mhz") is not None:
            lines.append(f"| Fmax (est) | {format_number(summary.get('fmax_est_mhz'), 2)} MHz |")
        if summary.get("total_power_w") is not None:
            lines.append(f"| Total Power | {format_number(summary.get('total_power_w'))} W |")
        if summary.get("dynamic_power_w") is not None:
            lines.append(f"| Dynamic Power | {format_number(summary.get('dynamic_power_w'))} W |")
        if summary.get("static_power_w") is not None:
            lines.append(f"| Static Power | {format_number(summary.get('static_power_w'))} W |")
        if summary.get("ip_power_w") is not None:
            ip_name = summary.get("ip_name") or "YOLO2_FPGA_0"
            lines.append(f"| {ip_name} Power | {format_number(summary.get('ip_power_w'))} W |")
        lines.append("")

        util = summary.get("utilization", {})
        avail = summary.get("available", {})
        if util:
            lines.append("### Post-Implementation Utilization")
            lines.append("")
            lines.append("| Resource | Used | Available | % |")
            lines.append("|----------|------|-----------|---|")
            for res in ["LUT", "FF", "DSP", "BRAM", "URAM"]:
                used = util.get(res)
                total = avail.get(res)
                if used is None and total is None:
                    continue
                pct = (used / total * 100.0) if (used is not None and total not in (None, 0)) else None
                pct_str = f"{pct:.2f}%" if pct is not None else "N/A"
                used_str = format_number(used, 0) if used is not None else "N/A"
                total_str = format_number(total, 0) if total is not None else "N/A"
                lines.append(f"| {res} | {used_str} | {total_str} | {pct_str} |")
            lines.append("")

    summary_path = bundle_path / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))


def load_bundle(bundle_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load meta.json and metrics.json from a bundle."""
    meta_path = bundle_path / "meta.json"
    metrics_path = bundle_path / "metrics.json"

    meta = {}
    metrics = {}

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return meta, metrics


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize config file and reports directory."""
    config_path = Path(args.config)

    if config_path.exists() and not args.force:
        print(f"Config file already exists: {config_path}")
        print("Use --force to overwrite.")
        return 1

    with open(config_path, "w") as f:
        json.dump(CONFIG_TEMPLATE, f, indent=2)

    reports_dir = Path(CONFIG_TEMPLATE["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    gitkeep = reports_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()

    print(f"Created config: {config_path}")
    print(f"Created reports directory: {reports_dir}")
    print("")
    print("Edit the config file to customize paths and KV260 SSH settings.")
    print("Then run: python3 scripts/yolo2_report.py run --label <label>")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Create a new report bundle."""
    # Load config if exists
    config = copy.deepcopy(CONFIG_TEMPLATE)
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            if key == "kv260" and isinstance(value, dict):
                config.setdefault("kv260", {}).update(value)
            else:
                config[key] = value

    # Override with CLI args
    reports_dir = Path(args.reports_dir or config["reports_dir"])
    hls_report_dir = args.hls_report_dir or config.get("hls_report_dir", "")
    vivado_report_dir = args.vivado_report_dir or config.get("vivado_report_dir", "")
    kv260_log_file = args.kv260_log

    label = args.label or "run"
    note = args.note or ""

    # Create bundle
    reports_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = create_bundle_dir(reports_dir, label)
    print(f"Creating report bundle: {bundle_path}")

    input_paths = {}
    metrics = {"hls": None, "vivado": None, "kv260": None}

    # Parse HLS reports
    if hls_report_dir:
        hls_path = Path(hls_report_dir)
        input_paths["hls_report_dir"] = str(hls_path)
        print(f"Parsing HLS reports from: {hls_path}")
        metrics["hls"] = parse_hls_report_dir(hls_path)

        # Copy top-level XML to bundle
        if metrics["hls"].get("top_level"):
            src = Path(metrics["hls"]["top_level"]["file"])
            if src.exists():
                shutil.copy(src, bundle_path / "hls" / src.name)

        # Save parsed output
        with open(bundle_path / "hls" / "parsed_hls.json", "w") as f:
            json.dump(metrics["hls"], f, indent=2)

    # Parse Vivado reports
    if vivado_report_dir:
        vivado_path = Path(vivado_report_dir)
        input_paths["vivado_report_dir"] = str(vivado_path)
        print(f"Parsing Vivado reports from: {vivado_path}")
        metrics["vivado"] = parse_vivado_report_dir(vivado_path)

        # Copy raw reports used (timing/util/power) into bundle for self-contained runs.
        for key in ("timing", "utilization", "power"):
            rpt_file = ((metrics.get("vivado") or {}).get(key) or {}).get("file")
            if rpt_file:
                src = Path(rpt_file)
                if src.exists():
                    shutil.copy(src, bundle_path / "vivado" / src.name)

        # Save parsed output
        with open(bundle_path / "vivado" / "parsed_vivado.json", "w") as f:
            json.dump(metrics["vivado"], f, indent=2)

    # Parse KV260 log (from file or SSH)
    kv260_log_content = None

    if kv260_log_file:
        kv260_log_path = Path(kv260_log_file)
        if kv260_log_path.exists():
            input_paths["kv260_log"] = str(kv260_log_path)
            print(f"Parsing KV260 log from: {kv260_log_path}")
            kv260_log_content = kv260_log_path.read_text()
            shutil.copy(kv260_log_path, bundle_path / "kv260" / "stdout.log")
    elif args.kv260_ssh:
        kv260_cfg = config.get("kv260", {})
        remote_cmd = args.kv260_cmd or kv260_cfg.get("remote_cmd_template", "")
        if remote_cmd:
            input_paths["kv260_ssh"] = config.get("kv260", {}).get("host", "")
            input_paths["kv260_cmd"] = remote_cmd
            print(f"Running KV260 command via SSH: {remote_cmd[:60]}...")

            # Password auth support (optional).
            # Default behavior remains key-based (BatchMode=yes) unless a password is provided.
            if getattr(args, "kv260_password_prompt", False):
                host = config.get("kv260", {}).get("host", "ubuntu@kria")
                config.setdefault("kv260", {})["password"] = getpass.getpass(f"KV260 SSH password for {host}: ")
            elif getattr(args, "kv260_password_env", None):
                env_key = str(args.kv260_password_env)
                pw = os.environ.get(env_key)
                if not pw:
                    print(f"Error: environment variable {env_key} is not set (needed for --kv260-password-env).")
                    return 2
                config.setdefault("kv260", {})["password"] = pw

            stdout, stderr, rc = run_kv260_ssh(config, remote_cmd)

            # Save command and output
            with open(bundle_path / "kv260" / "run_cmd.txt", "w") as f:
                f.write(remote_cmd)
            with open(bundle_path / "kv260" / "stdout.log", "w") as f:
                f.write(stdout)
            if stderr:
                with open(bundle_path / "kv260" / "stderr.log", "w") as f:
                    f.write(stderr)

            if rc != 0:
                print(f"Warning: SSH command returned {rc}")
                if "Permission denied" in (stderr or ""):
                    print("Hint: if your KV260 requires password auth, rerun with --kv260-password-prompt.")

            kv260_log_content = stdout

    if kv260_log_content:
        metrics["kv260"] = parse_kv260_log(kv260_log_content)
        with open(bundle_path / "kv260" / "parsed_kv260.json", "w") as f:
            json.dump(metrics["kv260"], f, indent=2)

    # Write meta and metrics
    meta = write_meta_json(bundle_path, label, note, input_paths)
    write_metrics_json(bundle_path, metrics)
    write_summary_md(bundle_path, meta, metrics)

    print(f"\nReport bundle created: {bundle_path}")
    print(f"  - meta.json")
    print(f"  - metrics.json")
    print(f"  - summary.md")

    # Print quick summary
    print("\n--- Quick Summary ---")
    if (metrics.get("kv260") or {}).get("parsed"):
        stats = metrics["kv260"]["stats"]
        print(f"KV260: {stats['count']} frames, median {stats['median_ms']:.2f} ms, ~{stats['fps_from_median']:.2f} FPS")
    if (metrics.get("hls") or {}).get("parsed"):
        summary = metrics["hls"]["summary"]
        util = summary.get("utilization", {})
        print(f"HLS: {summary.get('estimated_clock_ns', 'N/A')} ns est. clock, DSP={util.get('DSP', 'N/A')}, LUT={util.get('LUT', 'N/A')}")
    if (metrics.get("vivado") or {}).get("parsed"):
        summary = metrics["vivado"]["summary"]
        wns = summary.get("wns_ns")
        fmax = summary.get("fmax_est_mhz")
        pwr = summary.get("total_power_w")
        parts = [f"WNS={format_number(wns, 3)} ns"]
        if fmax is not None:
            parts.append(f"Fmax(est)={format_number(fmax, 2)} MHz")
        if pwr is not None:
            parts.append(f"P={format_number(pwr, 3)} W")
        print("Vivado: " + ", ".join(parts))

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List existing report bundles."""
    config = copy.deepcopy(CONFIG_TEMPLATE)
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            loaded = json.load(f)
        for key, value in loaded.items():
            if key == "kv260" and isinstance(value, dict):
                config.setdefault("kv260", {}).update(value)
            else:
                config[key] = value

    reports_dir = Path(args.reports_dir or config["reports_dir"])

    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        return 1

    bundles = []
    for d in sorted(reports_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "meta.json").exists():
            meta, metrics = load_bundle(d)
            bundles.append((d.name, meta, metrics))

    if not bundles:
        print("No report bundles found.")
        return 0

    # Print table header
    print(f"{'Bundle':<45} {'Label':<20} {'Git':<15} {'FPS':<10} {'WNS (ns)':<10}")
    print("-" * 100)

    for name, meta, metrics in bundles:
        label = meta.get("label", "")[:18]
        git = meta.get("git", {}).get("commit", "")[:12]
        if meta.get("git", {}).get("dirty"):
            git += "*"

        fps = "N/A"
        if (metrics.get("kv260") or {}).get("parsed"):
            fps_val = metrics["kv260"].get("stats", {}).get("fps_from_median")
            if fps_val:
                fps = f"{fps_val:.2f}"

        wns = "N/A"
        if (metrics.get("vivado") or {}).get("parsed"):
            wns_val = metrics["vivado"].get("summary", {}).get("wns_ns")
            if wns_val is not None:
                wns = f"{wns_val:.3f}"
        elif (metrics.get("hls") or {}).get("parsed"):
            # Show HLS slack as fallback
            pass

        print(f"{name:<45} {label:<20} {git:<15} {fps:<10} {wns:<10}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two report bundles."""
    bundle_a = Path(args.bundle_a)
    bundle_b = Path(args.bundle_b)

    if not bundle_a.exists():
        print(f"Bundle A not found: {bundle_a}")
        return 1
    if not bundle_b.exists():
        print(f"Bundle B not found: {bundle_b}")
        return 1

    meta_a, metrics_a = load_bundle(bundle_a)
    meta_b, metrics_b = load_bundle(bundle_b)

    label_a = meta_a.get("label", bundle_a.name)
    label_b = meta_b.get("label", bundle_b.name)

    print(f"# Comparison: {label_a} vs {label_b}")
    print("")

    rows = []

    # KV260 metrics
    kv260_a = (metrics_a.get("kv260") or {}).get("stats", {})
    kv260_b = (metrics_b.get("kv260") or {}).get("stats", {})

    if kv260_a or kv260_b:
        fps_a = kv260_a.get("fps_from_median")
        fps_b = kv260_b.get("fps_from_median")
        if fps_a is not None and fps_b is not None:
            delta = fps_b - fps_a
            pct = (delta / fps_a * 100) if fps_a != 0 else 0
            rows.append(("FPS (median)", f"{fps_a:.2f}", f"{fps_b:.2f}", f"{delta:+.2f} ({pct:+.1f}%)"))

        ms_a = kv260_a.get("median_ms")
        ms_b = kv260_b.get("median_ms")
        if ms_a is not None and ms_b is not None:
            delta = ms_b - ms_a
            pct = (delta / ms_a * 100) if ms_a != 0 else 0
            rows.append(("ms/infer (median)", f"{ms_a:.2f}", f"{ms_b:.2f}", f"{delta:+.2f} ({pct:+.1f}%)"))

    # HLS metrics
    hls_a = (metrics_a.get("hls") or {}).get("summary", {})
    hls_b = (metrics_b.get("hls") or {}).get("summary", {})

    if hls_a or hls_b:
        util_a = hls_a.get("utilization", {})
        util_b = hls_b.get("utilization", {})

        for res in ["DSP", "LUT", "FF", "BRAM_18K"]:
            val_a = util_a.get(res)
            val_b = util_b.get(res)
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                pct = (delta / val_a * 100) if val_a != 0 else 0
                rows.append((f"HLS {res}", str(val_a), str(val_b), f"{delta:+d} ({pct:+.1f}%)"))

        clk_a = hls_a.get("estimated_clock_ns")
        clk_b = hls_b.get("estimated_clock_ns")
        if clk_a is not None and clk_b is not None:
            delta = clk_b - clk_a
            rows.append(("HLS Est. Clock (ns)", f"{clk_a:.3f}", f"{clk_b:.3f}", f"{delta:+.3f}"))

    # Vivado metrics
    vivado_a = (metrics_a.get("vivado") or {}).get("summary", {})
    vivado_b = (metrics_b.get("vivado") or {}).get("summary", {})

    if vivado_a or vivado_b:
        wns_a = vivado_a.get("wns_ns")
        wns_b = vivado_b.get("wns_ns")
        if wns_a is not None and wns_b is not None:
            delta = wns_b - wns_a
            rows.append(("Vivado WNS (ns)", f"{wns_a:.3f}", f"{wns_b:.3f}", f"{delta:+.3f}"))

        pwr_a = vivado_a.get("total_power_w")
        pwr_b = vivado_b.get("total_power_w")
        if pwr_a is not None and pwr_b is not None:
            delta = pwr_b - pwr_a
            rows.append(("Vivado Power (W)", f"{pwr_a:.3f}", f"{pwr_b:.3f}", f"{delta:+.3f}"))

    if not rows:
        print("No comparable metrics found.")
        return 0

    # Print table
    print(f"| {'Metric':<25} | {label_a[:15]:<15} | {label_b[:15]:<15} | {'Delta':<20} |")
    print(f"|{'-'*27}|{'-'*17}|{'-'*17}|{'-'*22}|")
    for metric, val_a, val_b, delta in rows:
        print(f"| {metric:<25} | {val_a:<15} | {val_b:<15} | {delta:<20} |")

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO2 Performance Report Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize config and reports directory
  python3 scripts/yolo2_report.py init

  # Create a report from existing HLS reports
  python3 scripts/yolo2_report.py run --label baseline \\
    --hls-report-dir yolo2_int16/solution1/syn/report

  # Create a report with KV260 log file
  python3 scripts/yolo2_report.py run --label test1 \\
    --hls-report-dir yolo2_int16/solution1/syn/report \\
    --kv260-log /path/to/kv260_output.log

  # List all reports
  python3 scripts/yolo2_report.py list

  # Compare two reports
  python3 scripts/yolo2_report.py compare reports/<runA>/ reports/<runB>/
"""
    )

    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_FILE,
        help=f"Config file path (default: {DEFAULT_CONFIG_FILE})"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize config and reports directory")
    init_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing config")

    # run command
    run_parser = subparsers.add_parser("run", help="Create a new report bundle")
    run_parser.add_argument("--label", "-l", required=True, help="Label for this run")
    run_parser.add_argument("--note", "-n", default="", help="Optional note about this run")
    run_parser.add_argument("--reports-dir", help="Reports output directory")
    run_parser.add_argument("--hls-report-dir", help="HLS csynth.xml directory")
    run_parser.add_argument("--vivado-report-dir", help="Vivado .rpt files directory")
    run_parser.add_argument("--kv260-log", help="Path to KV260 log file")
    run_parser.add_argument("--kv260-ssh", action="store_true", help="Run command on KV260 via SSH")
    run_parser.add_argument("--kv260-cmd", help="Remote command to run on KV260")
    run_parser.add_argument(
        "--kv260-password-prompt",
        action="store_true",
        help="Prompt for KV260 SSH password (enables password auth; otherwise key-based only).",
    )
    run_parser.add_argument(
        "--kv260-password-env",
        help="Read KV260 SSH password from this environment variable (enables password auth).",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List existing report bundles")
    list_parser.add_argument("--reports-dir", help="Reports directory")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two report bundles")
    compare_parser.add_argument("bundle_a", help="First bundle path")
    compare_parser.add_argument("bundle_b", help="Second bundle path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "init":
        return cmd_init(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "compare":
        return cmd_compare(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
