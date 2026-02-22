#!/usr/bin/env python3
"""
KV260 YOLOv2 INT16 end-to-end pipeline runner (staged).

This script reads a YAML config (default: ./pipeline.yaml) and runs the repo's
"from scratch → detections" flow in clearly named stages.

Design goals:
  - Student-friendly: clear stage names, clear failures, easy to rerun partial flows
  - Reproducible: prefers batch/no-GUI flows when available
  - Minimal magic: stages call the repo's existing scripts/commands
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import yaml  # type: ignore
except Exception:
    print(
        "ERROR: PyYAML is required. Install with: python3 -m pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parents[1]


class PipelineError(RuntimeError):
    pass


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _require_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    v = cfg.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise PipelineError(f"Config '{key}' must be a mapping/object")
    return v


def _require_list(cfg: dict[str, Any], key: str) -> list[Any]:
    v = cfg.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise PipelineError(f"Config '{key}' must be a list")
    return v


def _path(repo_root: Path, p: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(p))
    path = Path(expanded)
    return path if path.is_absolute() else (repo_root / path)


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run_local(
    *,
    repo_root: Path,
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    source_scripts: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    cwd = cwd or repo_root
    source_scripts = source_scripts or []
    env = env or {}

    # Avoid `set -u` here: vendor environment scripts (Vivado/Vitis) are often not `nounset` clean.
    lines: list[str] = ["set -e", "set -o pipefail"]
    for s in source_scripts:
        s = s.strip()
        if not s:
            continue
        lines.append(f"source {shlex.quote(s)}")
    lines.append(f"cd {shlex.quote(str(cwd))}")
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    cmd_str = _quote_cmd(cmd)
    lines.append(f"{prefix} {cmd_str}".strip())

    script = "\n".join(lines)
    printable = cmd_str if not source_scripts and not env and cwd == repo_root else f"bash -lc {shlex.quote(script)}"
    print(f"$ {printable}", flush=True)

    if dry_run:
        return

    subprocess.run(["bash", "-lc", script], check=True)


def _ssh_base_args(ssh_cfg: dict[str, Any]) -> list[str]:
    args: list[str] = ["ssh"]
    port = int(ssh_cfg.get("port", 22))
    args += ["-p", str(port)]
    identity = str(ssh_cfg.get("identity_file", "") or "").strip()
    if identity:
        args += ["-i", os.path.expanduser(identity)]
    for opt in _require_list(ssh_cfg, "extra_opts"):
        if not isinstance(opt, str):
            raise PipelineError("deploy_kv260.ssh.extra_opts must be a list of strings")
        args.append(opt)
    return args


def run_ssh(
    *,
    ssh_cfg: dict[str, Any],
    remote_cmd: str,
    dry_run: bool = False,
) -> None:
    host = str(ssh_cfg.get("host", "")).strip()
    user = str(ssh_cfg.get("user", "")).strip()
    if not host or not user:
        raise PipelineError("deploy_kv260.ssh.host and deploy_kv260.ssh.user are required")

    args = _ssh_base_args(ssh_cfg)
    # `sudo` often requires a TTY to prompt for a password. Auto-allocate a pseudo-tty when the
    # remote command contains `sudo`, unless the user already provided -t/-tt in extra_opts.
    auto_tty_for_sudo = _as_bool(ssh_cfg.get("auto_tty_for_sudo"), default=True)
    allocate_tty = _as_bool(ssh_cfg.get("allocate_tty"), default=False)
    if not any(a in {"-t", "-tt"} for a in args):
        needs_tty = allocate_tty
        if auto_tty_for_sudo:
            try:
                needs_tty = needs_tty or ("sudo" in shlex.split(remote_cmd))
            except ValueError:
                needs_tty = needs_tty or (" sudo " in f" {remote_cmd} ")
        if needs_tty:
            args.insert(1, "-tt")
    args.append(f"{user}@{host}")
    args.append(remote_cmd)
    print(f"$ {_quote_cmd(args)}", flush=True)
    if dry_run:
        return
    subprocess.run(args, check=True)


def run_scp(
    *,
    ssh_cfg: dict[str, Any],
    local_paths: list[Path],
    remote_dir: str,
    recursive: bool = False,
    dry_run: bool = False,
) -> None:
    host = str(ssh_cfg.get("host", "")).strip()
    user = str(ssh_cfg.get("user", "")).strip()
    if not host or not user:
        raise PipelineError("deploy_kv260.ssh.host and deploy_kv260.ssh.user are required")

    args: list[str] = ["scp"]
    port = int(ssh_cfg.get("port", 22))
    args += ["-P", str(port)]
    identity = str(ssh_cfg.get("identity_file", "") or "").strip()
    if identity:
        args += ["-i", os.path.expanduser(identity)]
    for opt in _require_list(ssh_cfg, "extra_opts"):
        if not isinstance(opt, str):
            raise PipelineError("deploy_kv260.ssh.extra_opts must be a list of strings")
        args.append(opt)
    if recursive:
        args.append("-r")
    args += [str(p) for p in local_paths]
    args.append(f"{user}@{host}:{remote_dir.rstrip('/')}/")

    print(f"$ {_quote_cmd(args)}", flush=True)
    if dry_run:
        return
    subprocess.run(args, check=True)


def _find_first(root: Path, filename: str) -> Path | None:
    for p in root.rglob(filename):
        return p
    return None


def _format_shell_env_assignments(env_map: dict[str, Any], *, error_prefix: str) -> str:
    parts: list[str] = []
    for k, v in env_map.items():
        key = str(k)
        if not _ENV_KEY_RE.match(key):
            raise PipelineError(f"{error_prefix}: invalid environment variable name: {key}")
        if v is None:
            continue
        if isinstance(v, bool):
            value = "1" if v else "0"
        else:
            value = str(v)
        parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(parts)


def _cmd_exists_in_shell(cmd: str, *, source_scripts: list[str]) -> bool:
    if shutil.which(cmd) is not None:
        return True
    if not source_scripts:
        return False
    try:
        lines = ["set -e"]
        for s in source_scripts:
            s = s.strip()
            if not s:
                continue
            lines.append(f"source {shlex.quote(s)}")
        lines.append(f"command -v {shlex.quote(cmd)} >/dev/null 2>&1")
        subprocess.run(["bash", "-lc", "\n".join(lines)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


@dataclass(frozen=True)
class Stage:
    name: str
    description: str


def stage_host_sanity(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    env_cfg = _require_dict(cfg, "env")
    vivado_settings = str(env_cfg.get("vivado_settings", "") or "").strip()
    vitis_settings = str(env_cfg.get("vitis_settings", "") or "").strip()
    xsct_settings = str(env_cfg.get("xsct_settings", "") or "").strip()

    def _check_settings(label: str, value: str) -> None:
        if not value:
            return
        p = _path(repo_root, value)
        if not p.exists():
            raise PipelineError(f"{label} not found: {p}")

    _check_settings("env.vivado_settings", vivado_settings)
    _check_settings("env.vitis_settings", vitis_settings)
    _check_settings("env.xsct_settings", xsct_settings)

    effective_stages = cfg.get("__effective_stages")
    if isinstance(effective_stages, list) and all(isinstance(s, str) for s in effective_stages):
        stages = list(effective_stages)
    else:
        stages = _require_dict(cfg, "pipeline").get("stages", [])
        if not isinstance(stages, list):
            stages = []
        stages = [str(s) for s in stages]

    required_cmds: set[str] = {"bash", "python3"}
    if "host_quickstart" in stages:
        required_cmds |= {"make", "g++"}
    if "hls_ip" in stages:
        required_cmds |= {"vitis-run"}
    if "vivado_build" in stages:
        required_cmds |= {"vivado"}
    if "package_firmware" in stages:
        pkg_cfg = _require_dict(cfg, "package_firmware")
        dtbo_cfg = _require_dict(pkg_cfg, "dtbo")
    if "deploy_kv260" in stages or "run_kv260" in stages:
        required_cmds |= {"ssh", "scp"}

    print("== Host sanity ==")
    print(f"Repo root: {repo_root}")

    def _cmd_exists(cmd: str) -> bool:
        if shutil.which(cmd) is not None:
            return True
        if cmd in {"vivado", "bootgen"} and vivado_settings:
            try:
                subprocess.run(
                    ["bash", "-lc", f"set -e; source {shlex.quote(str(_path(repo_root, vivado_settings)))}; command -v {shlex.quote(cmd)}"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        if cmd == "vitis-run" and vitis_settings:
            try:
                subprocess.run(
                    ["bash", "-lc", f"set -e; source {shlex.quote(str(_path(repo_root, vitis_settings)))}; command -v vitis-run"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        if cmd == "xsct" and (xsct_settings or vitis_settings):
            s = xsct_settings or vitis_settings
            try:
                subprocess.run(
                    ["bash", "-lc", f"set -e; source {shlex.quote(str(_path(repo_root, s)))}; command -v xsct"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except subprocess.CalledProcessError:
                return False
        return False

    missing = [c for c in sorted(required_cmds) if not _cmd_exists(c)]
    if missing:
        raise PipelineError(f"Missing required command(s) in PATH: {', '.join(missing)}")

    # Optional-but-important tool checks for packaging (can be satisfied by Vivado-generated *.bit.bin).
    if "package_firmware" in stages:
        viv_cfg = _require_dict(cfg, "vivado_build")
        xsa = str(_require_dict(cfg, "package_firmware").get("xsa", "") or viv_cfg.get("xsa", "") or "").strip()
        bitbin_found = False
        if xsa:
            xsa_path = _path(repo_root, xsa)
            stem = xsa_path.stem
            xsa_dir = xsa_path.parent
            candidates = [
                xsa_dir / f"{stem}.bit.bin",
                xsa_dir / f"{xsa_dir.name}.runs/impl_1/{stem}.bit.bin",
            ]
            bitbin_found = any(p.exists() for p in candidates)

        if shutil.which("bootgen") is None and not bitbin_found:
            print("WARNING: bootgen not found and no Vivado-generated '*.bit.bin' was detected.")
            print("         Packaging will fail unless you source Vivado settings (so bootgen is in PATH) or generate a .bit.bin.")
            print("")

        if str(dtbo_cfg.get("method", "")).strip().lower() == "xsct" and not _cmd_exists("xsct"):
            print("WARNING: DTBO method is 'xsct' but xsct was not found in PATH.")
            print("         If a DTBO already exists in your package folder, the pipeline can keep using it; otherwise DTBO generation will fail.")
            print("")

    # Quick file presence checks (fail early on obvious misconfig).
    hls_cfg = _require_dict(cfg, "hls_ip")
    if "hls_ip" in stages:
        tcl = str(hls_cfg.get("tcl", "") or "").strip()
        if not tcl:
            raise PipelineError("hls_ip.tcl is required")
        if not _path(repo_root, tcl).exists():
            raise PipelineError(f"HLS TCL not found: {tcl}")

    viv_cfg = _require_dict(cfg, "vivado_build")
    if "vivado_build" in stages:
        bd_tcl = str(viv_cfg.get("bd_tcl", "") or "").strip()
        if not bd_tcl:
            raise PipelineError("vivado_build.bd_tcl is required")
        if not _path(repo_root, bd_tcl).exists():
            raise PipelineError(f"Vivado BD TCL not found: {bd_tcl}")

    pkg_cfg = _require_dict(cfg, "package_firmware")
    if "package_firmware" in stages:
        pkg_script = str(pkg_cfg.get("script", "") or "").strip()
        if not pkg_script:
            raise PipelineError("package_firmware.script is required")
        if not _path(repo_root, pkg_script).exists():
            raise PipelineError(f"Packaging script not found: {pkg_script}")

    print("OK\n")


def stage_host_quickstart(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    print("== Host quickstart (smoke test) ==")
    hqs = _require_dict(cfg, "host_quickstart")
    fp32 = _require_dict(hqs, "fp32")
    int16 = _require_dict(hqs, "int16")

    fp32_enabled = _as_bool(fp32.get("enabled"), default=True)
    int16_enabled = _as_bool(int16.get("enabled"), default=True)

    run_local(repo_root=repo_root, cmd=["make", "test"], dry_run=dry_run)
    run_local(repo_root=repo_root, cmd=["make", "gen"], dry_run=dry_run)

    if fp32_enabled:
        run_local(repo_root=repo_root, cmd=["./yolov2_weight_gen"], dry_run=dry_run)
        run_local(
            repo_root=repo_root,
            cmd=[
                "./yolov2_detect",
                "--cfg",
                "config/yolov2.cfg",
                "--names",
                "config/coco.names",
                "--input",
                str(fp32.get("input_image", "examples/test_images/dog.jpg")),
                "--output",
                str(fp32.get("output_dir", "predictions")),
                "--thresh",
                str(fp32.get("thresh", 0.5)),
                "--nms",
                str(fp32.get("nms", 0.45)),
                "--backend",
                "hls",
            ],
            dry_run=dry_run,
        )

    if int16_enabled:
        run_local(repo_root=repo_root, cmd=["make", "test-int16"], dry_run=dry_run)
        run_local(repo_root=repo_root, cmd=["./yolov2_weight_gen", "--precision", "int16"], dry_run=dry_run)
        run_local(
            repo_root=repo_root,
            cmd=[
                "./yolov2_detect",
                "--precision",
                "int16",
                "--cfg",
                "config/yolov2.cfg",
                "--names",
                "config/coco.names",
                "--input",
                str(int16.get("input_image", "examples/test_images/dog.jpg")),
            ],
            dry_run=dry_run,
        )

    print("OK\n")


def stage_hls_ip(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    print("== HLS IP export ==")
    env_cfg = _require_dict(cfg, "env")
    vitis_settings = str(env_cfg.get("vitis_settings", "") or "").strip()

    hls_cfg = _require_dict(cfg, "hls_ip")
    tcl = str(hls_cfg.get("tcl", "") or "").strip()
    if not tcl:
        raise PipelineError("hls_ip.tcl is required")
    tcl_path = _path(repo_root, tcl)
    if not tcl_path.exists():
        raise PipelineError(f"HLS TCL not found: {tcl_path}")

    extra_env: dict[str, str] = {}
    for k, v in _require_dict(hls_cfg, "env").items():
        extra_env[str(k)] = str(v)

    run_local(
        repo_root=repo_root,
        cmd=["vitis-run", "--mode", "hls", "--tcl", str(tcl_path)],
        env=extra_env,
        source_scripts=[vitis_settings] if vitis_settings else [],
        dry_run=dry_run,
    )

    print("OK\n")


def stage_vivado_build(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    print("== Vivado build (no GUI) ==")
    env_cfg = _require_dict(cfg, "env")
    vivado_settings = str(env_cfg.get("vivado_settings", "") or "").strip()

    viv = _require_dict(cfg, "vivado_build")
    bd_tcl = str(viv.get("bd_tcl", "") or "").strip()
    proj_dir = str(viv.get("proj_dir", "") or "").strip()
    xsa = str(viv.get("xsa", "") or "").strip()
    ip_repo = str(viv.get("ip_repo", "") or "").strip()
    jobs = int(viv.get("jobs", 8))

    if not bd_tcl or not proj_dir or not xsa:
        raise PipelineError("vivado_build.bd_tcl, vivado_build.proj_dir, and vivado_build.xsa are required")

    bd_tcl_path = _path(repo_root, bd_tcl)
    if not bd_tcl_path.exists():
        raise PipelineError(f"BD TCL not found: {bd_tcl_path}")

    ip_repo_path = _path(repo_root, ip_repo) if ip_repo else None
    if ip_repo_path is not None and not (ip_repo_path / "component.xml").exists():
        raise PipelineError(
            "Vivado IP repo not found or missing component.xml. "
            "Run the HLS IP stage first (hls_ip) or fix vivado_build.ip_repo."
        )

    run_local(
        repo_root=repo_root,
        cmd=[
            str((repo_root / "vivado/build_from_bd.sh").resolve()),
            "--bd-tcl",
            str(bd_tcl_path),
            "--proj-dir",
            str(_path(repo_root, proj_dir)),
            "--xsa",
            str(_path(repo_root, xsa)),
            "--ip-repo",
            str(ip_repo_path) if ip_repo_path is not None else "",
            "--jobs",
            str(jobs),
        ]
        if ip_repo_path is not None
        else [
            str((repo_root / "vivado/build_from_bd.sh").resolve()),
            "--bd-tcl",
            str(bd_tcl_path),
            "--proj-dir",
            str(_path(repo_root, proj_dir)),
            "--xsa",
            str(_path(repo_root, xsa)),
            "--jobs",
            str(jobs),
        ],
        source_scripts=[vivado_settings] if vivado_settings else [],
        dry_run=dry_run,
    )

    xsa_path = _path(repo_root, xsa)
    if not dry_run and not xsa_path.exists():
        raise PipelineError(f"Vivado build finished but XSA was not found: {xsa_path}")

    print("OK\n")


def stage_package_firmware(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    print("== Package firmware for KV260 (xmutil) ==")
    env_cfg = _require_dict(cfg, "env")
    vivado_settings = str(env_cfg.get("vivado_settings", "") or "").strip()
    xsct_settings = str(env_cfg.get("xsct_settings", "") or "").strip() or str(env_cfg.get("vitis_settings", "") or "").strip()

    pkg = _require_dict(cfg, "package_firmware")
    pkg_script = str(pkg.get("script", "") or "").strip()
    if not pkg_script:
        raise PipelineError("package_firmware.script is required")
    pkg_script_path = _path(repo_root, pkg_script)
    if not pkg_script_path.exists():
        raise PipelineError(f"Packaging script not found: {pkg_script_path}")

    accel_name = str(pkg.get("accel_name", "yolov2_accel") or "yolov2_accel").strip()
    if not accel_name:
        raise PipelineError("package_firmware.accel_name must be non-empty")
    output_dir = str(pkg.get("output_dir", f"linux_app/accel_package/{accel_name}") or "").strip()
    if not output_dir:
        raise PipelineError("package_firmware.output_dir must be non-empty")
    output_dir_path = _path(repo_root, output_dir)

    # Prefer explicit inputs, otherwise use the Vivado stage outputs.
    viv_cfg = _require_dict(cfg, "vivado_build")
    xsa = str(pkg.get("xsa", "") or viv_cfg.get("xsa", "") or "").strip()
    if not xsa:
        raise PipelineError("package_firmware.xsa (or vivado_build.xsa) is required")
    xsa_path = _path(repo_root, xsa)
    bit = str(pkg.get("bit", "") or "").strip()
    bit_bin = str(pkg.get("bit_bin", "") or "").strip()

    # Choose an explicit bitstream by default so packaging does not accidentally pick a stale
    # pre-generated .bit.bin from a previous run.
    if not bit:
        xsa_stem = xsa_path.stem
        xsa_dir = xsa_path.parent
        bit_candidates = [
            xsa_dir / f"{xsa_stem}.bit",
            xsa_dir / f"{xsa_dir.name}.runs/impl_1/{xsa_stem}.bit",
        ]
        found_bits = [p for p in bit_candidates if p.exists()]
        if found_bits:
            selected_bit = max(found_bits, key=lambda p: p.stat().st_mtime)
            bit = str(selected_bit)
            print(f"INFO: Auto-selected bitstream for packaging: {selected_bit}")
            if xsa_path.exists():
                try:
                    if selected_bit.stat().st_mtime + 1e-6 < xsa_path.stat().st_mtime:
                        print(
                            "WARNING: Selected .bit appears older than .xsa; verify vivado_build completed for this run."
                        )
                except OSError:
                    pass

    dtbo_cfg = _require_dict(pkg, "dtbo")
    method = str(dtbo_cfg.get("method", "manual") or "manual").strip().lower()
    if method not in {"manual", "xsct", "skip"}:
        raise PipelineError("package_firmware.dtbo.method must be one of: manual, xsct, skip")

    dtbo_to_include: Path | None = None
    if method == "xsct":
        target_dtbo = output_dir_path / f"{accel_name}.dtbo"
        xsct_is_available = _cmd_exists_in_shell("xsct", source_scripts=[xsct_settings] if xsct_settings else [])

        if not xsct_is_available and not dry_run:
            if target_dtbo.exists():
                print(f"WARNING: xsct not found; keeping existing DTBO: {target_dtbo}")
                print("         To regenerate DTBO, set env.xsct_settings (or env.vitis_settings) and ensure xsct is in PATH.")
                print("")
            else:
                raise PipelineError("xsct not found in PATH (needed for DTBO generation). Set env.xsct_settings or source Vitis.")
        else:
            xsct_cfg = _require_dict(dtbo_cfg, "xsct")
            out_dir = _path(repo_root, str(xsct_cfg.get("out_dir", "tmp/dts_output")))
            platform_name = str(xsct_cfg.get("platform_name", "yolov2_kv260"))
            git_branch = str(xsct_cfg.get("git_branch", "xlnx_rel_v2024.1"))
            compile_flag = _as_bool(xsct_cfg.get("compile"), default=True)

            if not xsa_path.exists() and not dry_run:
                raise PipelineError(f"XSA not found for DTBO generation: {xsa_path}")

            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)

            # Generate a small XSCT TCL file so students can read/modify it if needed.
            tcl_text = textwrap.dedent(
                f"""\
                set xsa_file "{xsa_path}"
                set out_dir "{out_dir}"
                set platform_name "{platform_name}"
                set git_branch "{git_branch}"

                puts "INFO: XSA: $xsa_file"
                puts "INFO: OUT: $out_dir"
                puts "INFO: platform: $platform_name"
                puts "INFO: git-branch: $git_branch"

                # NOTE: `createdts -hw <xsa>` opens the design internally; avoid opening it twice.
                set args [list -hw $xsa_file -platform-name $platform_name -overlay -out $out_dir -git-branch $git_branch]
                lappend args -zocl
                """
            )
            if compile_flag:
                tcl_text += "\nlappend args -compile\n"
            tcl_text += "eval createdts $args\n"
            tcl_text += "puts \"INFO: createdts finished\"\n"

            tcl_path = repo_root / f"tmp/xsct_createdts_{accel_name}.tcl"
            if not dry_run:
                tcl_path.parent.mkdir(parents=True, exist_ok=True)
                tcl_path.write_text(tcl_text, encoding="utf-8")

            run_local(
                repo_root=repo_root,
                cmd=["xsct", "-nodisp", str(tcl_path)],
                source_scripts=[xsct_settings] if xsct_settings else [],
                dry_run=dry_run,
            )

            # Prefer the canonical path, but fall back to a recursive search.
            canonical_bsp = out_dir / platform_name / "psu_cortexa53_0" / "device_tree_domain" / "bsp"
            canonical_dtbo = canonical_bsp / "pl.dtbo"
            canonical_dtsi = canonical_bsp / "pl.dtsi"

            pl_dtbo = canonical_dtbo
            pl_dtsi = canonical_dtsi

            if not dry_run:
                if canonical_dtbo.exists():
                    pl_dtbo = canonical_dtbo
                elif canonical_dtsi.exists():
                    pl_dtsi = canonical_dtsi
                    pl_dtbo = canonical_dtbo
                else:
                    found_dtsi = _find_first(out_dir, "pl.dtsi")
                    if found_dtsi is not None:
                        pl_dtsi = found_dtsi
                        pl_dtbo = found_dtsi.with_suffix(".dtbo")
                    else:
                        found_dtbo = _find_first(out_dir, "pl.dtbo")
                        if found_dtbo is not None:
                            pl_dtbo = found_dtbo
                        else:
                            raise PipelineError(f"Could not find pl.dtsi/pl.dtbo under: {out_dir}")

                if not pl_dtbo.exists() and pl_dtsi.exists():
                    if shutil.which("dtc") is None:
                        raise PipelineError("dtc not found in PATH (needed to compile pl.dtsi → pl.dtbo)")
                    run_local(
                        repo_root=repo_root,
                        cmd=["dtc", "-@", "-O", "dtb", "-o", str(pl_dtbo), str(pl_dtsi)],
                        dry_run=dry_run,
                    )
                    if not pl_dtbo.exists():
                        raise PipelineError(f"dtc completed but DTBO was not created: {pl_dtbo}")

            dtbo_to_include = pl_dtbo

    run_local(
        repo_root=repo_root,
        cmd=[
            str(pkg_script_path),
            "--accel-name",
            accel_name,
            "--out-dir",
            str(output_dir_path),
            "--xsa",
            str(xsa_path),
        ]
        + (["--bit", str(_path(repo_root, bit))] if bit else [])
        + (["--bit-bin", str(_path(repo_root, bit_bin))] if bit_bin else [])
        + (["--dtbo", str(dtbo_to_include)] if dtbo_to_include is not None else []),
        source_scripts=[vivado_settings] if vivado_settings else [],
        dry_run=dry_run,
    )

    expected_bit_bin = output_dir_path / f"{accel_name}.bit.bin"
    if not dry_run and not expected_bit_bin.exists():
        raise PipelineError(
            f"Expected {expected_bit_bin} after packaging, but it was not created. "
            "Hint: ensure you built the Vivado bitstream and/or have bootgen available."
        )

    if method == "xsct" and not dry_run:
        expected_dtbo = output_dir_path / f"{accel_name}.dtbo"
        if not expected_dtbo.exists():
            raise PipelineError(
                f"DTBO method is 'xsct' but DTBO was not created in the package: {expected_dtbo}\n"
                "Hint: ensure xsct+createdts are available and your XSA path is correct."
            )

    print("OK\n")


def stage_deploy_kv260(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    print("== Deploy to KV260 ==")
    d = _require_dict(cfg, "deploy_kv260")
    ssh_cfg = _require_dict(d, "ssh")
    remote = _require_dict(d, "remote")
    copy = _require_dict(d, "copy")
    files = _require_dict(d, "files")

    home = str(remote.get("home", "/home/ubuntu"))
    linux_app_dir = str(remote.get("linux_app_dir", f"{home}/linux_app"))
    accel_dir = str(remote.get("accel_dir", f"{home}/yolov2_accel"))
    weights_dir = str(remote.get("weights_dir", f"{home}/weights"))
    config_dir = str(remote.get("config_dir", f"{home}/config"))
    images_dir = str(remote.get("images_dir", f"{home}/test_images"))

    # 1) Copy linux_app folder (host → board).
    if _as_bool(copy.get("linux_app"), default=True):
        run_scp(
            ssh_cfg=ssh_cfg,
            local_paths=[repo_root / "linux_app"],
            remote_dir=os.path.dirname(linux_app_dir) or home,
            recursive=True,
            dry_run=dry_run,
        )

    # 2) Copy accelerator package folder + deploy into /lib/firmware.
    if _as_bool(copy.get("accel_package"), default=True):
        run_scp(
            ssh_cfg=ssh_cfg,
            local_paths=[repo_root / "linux_app/accel_package/yolov2_accel"],
            remote_dir=os.path.dirname(accel_dir) or home,
            recursive=True,
            dry_run=dry_run,
        )
        run_ssh(
            ssh_cfg=ssh_cfg,
            remote_cmd=f"set -e; cd {shlex.quote(accel_dir)} && sudo ./deploy_to_kv260.sh",
            dry_run=dry_run,
        )

    # 3) Create dirs and copy model/config/images.
    mkdir_cmd = f"set -e; mkdir -p {shlex.quote(weights_dir)} {shlex.quote(config_dir)} {shlex.quote(images_dir)}"
    run_ssh(ssh_cfg=ssh_cfg, remote_cmd=mkdir_cmd, dry_run=dry_run)

    if _as_bool(copy.get("weights"), default=True):
        weight_files: list[str] = [str(x) for x in _require_list(files, "weights")]
        if weight_files:
            run_scp(
                ssh_cfg=ssh_cfg,
                local_paths=[_path(repo_root, p) for p in weight_files],
                remote_dir=weights_dir,
                recursive=False,
                dry_run=dry_run,
            )

    if _as_bool(copy.get("config"), default=True):
        cfg_files: list[str] = [str(x) for x in _require_list(files, "config")]
        if cfg_files:
            run_scp(
                ssh_cfg=ssh_cfg,
                local_paths=[_path(repo_root, p) for p in cfg_files],
                remote_dir=config_dir,
                recursive=False,
                dry_run=dry_run,
            )

    if _as_bool(copy.get("test_image"), default=True):
        img = str(files.get("test_image", "examples/test_images/dog.jpg"))
        run_scp(
            ssh_cfg=ssh_cfg,
            local_paths=[_path(repo_root, img)],
            remote_dir=images_dir,
            recursive=False,
            dry_run=dry_run,
        )

    # 4) Build linux_app on board.
    if _as_bool(d.get("build_linux_app"), default=True):
        run_ssh(
            ssh_cfg=ssh_cfg,
            remote_cmd=f"set -e; cd {shlex.quote(linux_app_dir)} && make clean && make",
            dry_run=dry_run,
        )

    print("OK\n")


def stage_run_kv260(cfg: dict[str, Any], *, repo_root: Path, dry_run: bool) -> None:
    print("== Run on KV260 ==")
    d = _require_dict(cfg, "deploy_kv260")
    ssh_cfg = _require_dict(d, "ssh")
    remote = _require_dict(d, "remote")
    linux_app_dir = str(remote.get("linux_app_dir", "/home/ubuntu/linux_app"))

    r = _require_dict(cfg, "run_kv260")
    env_map = _require_dict(r, "env")
    env_prefix = _format_shell_env_assignments(env_map, error_prefix="run_kv260.env")
    env_prefix = (env_prefix + " ") if env_prefix else ""
    args = [str(a) for a in _require_list(r, "args")]
    if not args:
        raise PipelineError("run_kv260.args must be a non-empty list, e.g. ['-v','1','-i','/home/ubuntu/test_images/dog.jpg']")

    # `start_yolo.sh` uses sudo internally; run `sudo -v` first so the user enters the password once (if needed),
    # and the rest of the stage can proceed without repeated sudo prompts.
    cmd = "set -e; cd " + shlex.quote(linux_app_dir) + " && sudo -v && " + env_prefix + "./start_yolo.sh " + " ".join(shlex.quote(a) for a in args)
    run_ssh(ssh_cfg=ssh_cfg, remote_cmd=cmd, dry_run=dry_run)
    print("OK\n")


STAGES: dict[str, tuple[Stage, Any]] = {
    "host_sanity": (Stage("host_sanity", "Check host commands/files/toolchain paths"), stage_host_sanity),
    "host_quickstart": (Stage("host_quickstart", "Build + run host smoke test (fp32/int16)"), stage_host_quickstart),
    "hls_ip": (Stage("hls_ip", "Build/export HLS IP repo (vitis-run)"), stage_hls_ip),
    "vivado_build": (Stage("vivado_build", "Batch Vivado build from BD TCL"), stage_vivado_build),
    "package_firmware": (Stage("package_firmware", "Create xmutil firmware package (.bit.bin + .dtbo)"), stage_package_firmware),
    "deploy_kv260": (Stage("deploy_kv260", "Copy files to KV260, deploy firmware, build app"), stage_deploy_kv260),
    "run_kv260": (Stage("run_kv260", "Run inference on the KV260"), stage_run_kv260),
}


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise PipelineError("Top-level YAML must be a mapping/object")
    return cfg


def compute_stage_list(cfg: dict[str, Any], cli_from: str | None, cli_to: str | None) -> list[str]:
    pipeline = _require_dict(cfg, "pipeline")
    stages = [str(s) for s in _require_list(pipeline, "stages")]
    if not stages:
        raise PipelineError("pipeline.stages must be a non-empty list")

    for s in stages:
        if s not in STAGES:
            raise PipelineError(f"Unknown stage '{s}'. Known: {', '.join(sorted(STAGES.keys()))}")

    start = cli_from or (str(pipeline.get("from")) if pipeline.get("from") not in (None, "null") else None)
    end = cli_to or (str(pipeline.get("to")) if pipeline.get("to") not in (None, "null") else None)

    if start:
        if start not in stages:
            raise PipelineError(f"pipeline.from '{start}' not found in pipeline.stages")
        stages = stages[stages.index(start) :]
    if end:
        if end not in stages:
            raise PipelineError(f"pipeline.to '{end}' not found in pipeline.stages")
        stages = stages[: stages.index(end) + 1]

    return stages


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run the staged KV260 YOLOv2 INT16 pipeline using a YAML config.",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "pipeline.yaml"),
        help="Path to YAML config (default: ./pipeline.yaml)",
    )
    parser.add_argument("--list-stages", action="store_true", help="List built-in stages and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them (overrides YAML)")
    parser.add_argument("--from", dest="from_stage", default=None, help="Start from this stage (inclusive)")
    parser.add_argument("--to", dest="to_stage", default=None, help="Stop after this stage (inclusive)")

    args = parser.parse_args()

    if args.list_stages:
        for name in sorted(STAGES.keys()):
            st, _ = STAGES[name]
            print(f"- {st.name}: {st.description}")
        return 0

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = load_config(cfg_path)
    pipeline_cfg = _require_dict(cfg, "pipeline")
    dry_run = args.dry_run or _as_bool(pipeline_cfg.get("dry_run"), default=False)

    stages = compute_stage_list(cfg, args.from_stage, args.to_stage)
    cfg["__effective_stages"] = stages

    print("== Pipeline ==", flush=True)
    print(f"Config:  {cfg_path}", flush=True)
    print(f"Stages:  {', '.join(stages)}", flush=True)
    print(f"Dry-run: {dry_run}", flush=True)
    print("", flush=True)

    for name in stages:
        _, fn = STAGES[name]
        fn(cfg, repo_root=REPO_ROOT, dry_run=dry_run)

    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: command failed with exit code {e.returncode}", file=sys.stderr)
        raise SystemExit(e.returncode)
