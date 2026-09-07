#!/usr/bin/env python3
"""Download federated-learning result directories from the ``main`` host.

Uses the existing SSH alias from ~/.ssh/config. No remote Python or Ansible
module is required; transfer is performed with OpenSSH scp.

Examples:
    # newest result directory
    python tools-deployment/scripts/download_results.py --latest

    # newest FedProx result
    python tools-deployment/scripts/download_results.py --latest --algorithm FedProx

    # all result directories not already downloaded
    python tools-deployment/scripts/download_results.py --all

    # one exact result directory
    python tools-deployment/scripts/download_results.py --run 20260907T134242Z_FedProx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def ssh(host: str, command: str) -> str:
    proc = run(["ssh", host, command], capture=True)
    return proc.stdout.strip()


def resolve_remote_root(host: str, remote_root: str) -> str:
    # Resolve ~ remotely so scp receives an absolute path.
    return ssh(host, f"mkdir -p {remote_root} && cd {remote_root} && pwd")


def list_remote_runs(host: str, remote_root: str) -> list[str]:
    out = ssh(
        host,
        f"find {remote_root} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n' | sort",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def choose_runs(
    runs: list[str],
    *,
    exact_run: str | None,
    latest: bool,
    all_runs: bool,
    algorithm: str | None,
) -> list[str]:
    filtered = runs
    if algorithm:
        suffix = f"_{algorithm}"
        filtered = [name for name in filtered if name.endswith(suffix)]

    if exact_run:
        if exact_run not in runs:
            raise SystemExit(f"remote run does not exist: {exact_run}")
        if algorithm and exact_run not in filtered:
            raise SystemExit(f"run {exact_run} does not match --algorithm {algorithm}")
        return [exact_run]

    if latest:
        if not filtered:
            raise SystemExit("no matching result directories found")
        return [filtered[-1]]

    if all_runs:
        return filtered

    raise SystemExit("choose one of --latest, --all, or --run")


def download_one(
    host: str,
    remote_root: str,
    run_name: str,
    destination_root: Path,
    overwrite: bool,
) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    target = destination_root / run_name

    if target.exists():
        if not overwrite:
            print(f"SKIP {run_name}: already exists at {target}")
            return target
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    run(["scp", "-r", f"{host}:{remote_root}/{run_name}", str(destination_root)])

    summary = target / "summary.json"
    if not summary.exists():
        raise RuntimeError(f"downloaded result has no summary.json: {target}")

    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
        algorithm = data.get("algorithm", "?")
        rounds = data.get("completed_rounds", "?")
        print(f"OK   {run_name}: algorithm={algorithm}, completed_rounds={rounds}")
    except Exception:
        print(f"OK   {run_name}")

    return target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FL experiment results from main")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--latest", action="store_true", help="download newest matching result")
    mode.add_argument("--all", action="store_true", help="download all matching results")
    mode.add_argument("--run", help="download one exact result-directory name")
    p.add_argument("--algorithm", choices=["FedAvg", "FedProx", "FedPAQ", "FedMA", "HierFedAvg"])
    p.add_argument("--host", default="main", help="SSH host/alias (default: main)")
    p.add_argument(
        "--remote-results-root",
        default="~/Platform-for-prototyping-fl-in-IoT/tools-deployment/results",
    )
    p.add_argument("--output", default="downloaded-results", help="local destination directory")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for command in ("ssh", "scp"):
        if shutil.which(command) is None:
            raise SystemExit(f"required command not found: {command}")

    remote_root = resolve_remote_root(args.host, args.remote_results_root)
    runs = list_remote_runs(args.host, remote_root)
    selected = choose_runs(
        runs,
        exact_run=args.run,
        latest=args.latest,
        all_runs=args.all,
        algorithm=args.algorithm,
    )

    if not selected:
        print("No matching result directories found.")
        return

    output = Path(args.output).resolve()
    print(f"Remote results: {args.host}:{remote_root}")
    print(f"Local output:   {output}")
    print(f"Selected runs:  {len(selected)}")

    for run_name in selected:
        download_one(args.host, remote_root, run_name, output, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit status {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)
