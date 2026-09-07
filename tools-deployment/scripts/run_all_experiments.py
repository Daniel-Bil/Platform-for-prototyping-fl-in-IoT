#!/usr/bin/env python3
"""Run all deployment FL algorithms sequentially and collect their result folders.

This script is meant to be executed on the operator/laptop machine from the
repository checkout. It reuses the generic Ansible experiment launcher and the
existing ``main`` SSH alias, waits for each experiment to finish, downloads the
new result directory, then produces comparison CSV/Markdown files.

Example:
    python tools-deployment/scripts/run_all_experiments.py \
        --clients 7 --rounds 5 --local-epochs 3 --repetitions 3
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

DEFAULT_ALGORITHMS = ["FedAvg", "FedProx", "FedPAQ", "FedMA", "HierFedAvg"]


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def ssh(command: str, *, capture: bool = True) -> str:
    completed = run(["ssh", "main", command], capture=capture)
    return completed.stdout.strip() if capture and completed.stdout else ""


def remote_run_names(remote_results_root: str) -> set[str]:
    command = (
        f"mkdir -p {remote_results_root} && "
        f"find {remote_results_root} -mindepth 1 -maxdepth 1 -type d -printf '%f\\n'"
    )
    output = ssh(command)
    return {line.strip() for line in output.splitlines() if line.strip()}


def wait_for_server(timeout: float, poll: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = subprocess.run(
            ["ssh", "main", "systemctl", "is-active", "--quiet", "fl-thesis-server.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if status.returncode != 0:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FL server did not finish within {timeout:.0f}s")
        time.sleep(poll)

    result = ssh("systemctl show fl-thesis-server.service -p Result --value || true")
    exec_status = ssh("systemctl show fl-thesis-server.service -p ExecMainStatus --value || true")
    if result and result != "success":
        raise RuntimeError(f"server service finished with Result={result}, ExecMainStatus={exec_status}")
    if exec_status and exec_status != "0":
        raise RuntimeError(f"server service exited with status {exec_status}")


def download_run(remote_results_root: str, run_name: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    run(["scp", "-r", f"main:{remote_results_root}/{run_name}", str(destination)])
    downloaded = destination / run_name
    if not (downloaded / "summary.json").exists():
        raise RuntimeError(f"downloaded run has no summary.json: {downloaded}")
    return downloaded


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_row(run_dir: Path, repetition: int, requested_clients: int) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "config.json")
    metrics = summary.get("final_metrics") or {}
    params = summary.get("algorithm_params") or {}
    return {
        "repetition": repetition,
        "seed": summary.get("seed", config.get("seed")),
        "algorithm": summary.get("algorithm", config.get("algorithm")),
        "requested_clients": requested_clients,
        "completed_rounds": summary.get("completed_rounds"),
        "final_test_loss": metrics.get("test_loss"),
        "final_accuracy": metrics.get("accuracy"),
        "final_precision": metrics.get("precision"),
        "final_recall": metrics.get("recall"),
        "final_f1": metrics.get("f1"),
        "final_macro_f1": metrics.get("macro_f1"),
        "total_network_bytes": summary.get("total_network_bytes"),
        "total_network_mib": (
            None if summary.get("total_network_bytes") is None
            else float(summary["total_network_bytes"]) / (1024.0 * 1024.0)
        ),
        "total_training_network_bytes": summary.get("total_training_network_bytes"),
        "total_evaluation_network_bytes": summary.get("total_evaluation_network_bytes"),
        "total_round_seconds": summary.get("total_round_seconds"),
        "mean_aggregation_seconds": summary.get("mean_aggregation_seconds"),
        "fedprox_mu": params.get("fedprox_mu"),
        "fedpaq_bits": params.get("fedpaq_bits"),
        "remote_run_dir": summary.get("run_dir"),
        "local_run_dir": str(run_dir),
        "git_commit": config.get("git_commit"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "repetition", "seed", "algorithm", "completed_rounds",
        "final_accuracy", "final_f1", "final_macro_f1",
        "total_network_mib", "total_round_seconds", "mean_aggregation_seconds",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        formatted = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                value = f"{value:.6f}"
            formatted.append("" if value is None else str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [
        "final_test_loss", "final_accuracy", "final_f1", "final_macro_f1",
        "total_network_mib", "total_round_seconds", "mean_aggregation_seconds",
    ]
    result: list[dict[str, Any]] = []
    algorithms = []
    for row in rows:
        if row["algorithm"] not in algorithms:
            algorithms.append(row["algorithm"])
    for algorithm in algorithms:
        group = [r for r in rows if r["algorithm"] == algorithm]
        out: dict[str, Any] = {"algorithm": algorithm, "runs": len(group)}
        for field in numeric:
            values = [float(r[field]) for r in group if r.get(field) is not None]
            out[f"{field}_mean"] = statistics.mean(values) if values else None
            out[f"{field}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        result.append(out)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and collect all federated deployment algorithms")
    parser.add_argument("--clients", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--edge-count", type=int, default=2)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--fedpaq-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="base seed; repetition N uses seed + N - 1")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="comma-separated subset/order",
    )
    parser.add_argument("--inventory", default="ansible/inventory.yml")
    parser.add_argument("--playbook", default="ansible/playbooks/02-start-one-client.yml")
    parser.add_argument(
        "--remote-results-root",
        default="~/Platform-for-prototyping-fl-in-IoT/tools-deployment/results",
    )
    parser.add_argument("--output-root", default="benchmark-results")
    parser.add_argument("--experiment-timeout", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clients < 1:
        raise SystemExit("--clients must be >= 1")
    if args.rounds < 1 or args.local_epochs < 1 or args.repetitions < 1:
        raise SystemExit("--rounds, --local-epochs and --repetitions must be >= 1")
    if not 1 <= args.fedpaq_bits <= 8:
        raise SystemExit("--fedpaq-bits must be in 1..8")
    if args.fedprox_mu < 0:
        raise SystemExit("--fedprox-mu must be >= 0")

    algorithms = [value.strip() for value in args.algorithms.split(",") if value.strip()]
    unknown = [value for value in algorithms if value not in DEFAULT_ALGORITHMS]
    if unknown:
        raise SystemExit(f"unsupported algorithms: {', '.join(unknown)}")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    inventory = (repo_root / args.inventory).resolve()
    playbook = (repo_root / args.playbook).resolve()
    if not inventory.exists() or not playbook.exists():
        raise SystemExit("inventory/playbook path does not exist")
    if shutil.which("ansible-playbook") is None or shutil.which("ssh") is None or shutil.which("scp") is None:
        raise SystemExit("ansible-playbook, ssh and scp must be installed")

    # Resolve ~ on the remote once so both find(1) and scp use an absolute path.
    remote_results_root = ssh(
        f"mkdir -p {args.remote_results_root} && cd {args.remote_results_root} && pwd"
    )

    batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = (repo_root / args.output_root / batch_stamp).resolve()
    batch_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithms": algorithms,
        "clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "edge_count": args.edge_count,
        "fedprox_mu": args.fedprox_mu,
        "fedpaq_bits": args.fedpaq_bits,
        "base_seed": args.seed,
        "repetitions": args.repetitions,
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        seed = args.seed + repetition - 1
        rep_dir = batch_dir / f"rep-{repetition:03d}-seed-{seed}"
        rep_dir.mkdir(parents=True)
        for algorithm in algorithms:
            print("\n" + "=" * 88)
            print(f"RUN {repetition}/{args.repetitions}: {algorithm} | clients={args.clients} | seed={seed}")
            print("=" * 88)
            before = remote_run_names(remote_results_root)

            extra_vars = [
                f"fl_client_count={args.clients}",
                f"fl_algorithm={algorithm}",
                f"fl_rounds={args.rounds}",
                f"fl_local_epochs={args.local_epochs}",
                f"fl_batch_size={args.batch_size}",
                f"fl_seed={seed}",
                f"fl_fedprox_mu={args.fedprox_mu}",
                f"fl_fedpaq_bits={args.fedpaq_bits}",
                f"fl_edge_count={args.edge_count}",
            ]
            cmd = ["ansible-playbook", "-i", str(inventory), str(playbook)]
            for value in extra_vars:
                cmd.extend(["-e", value])
            run(cmd, cwd=repo_root)
            wait_for_server(args.experiment_timeout)

            after = remote_run_names(remote_results_root)
            new_runs = sorted(after - before)
            matching = [name for name in new_runs if f"_{algorithm}" in name]
            if not matching:
                raise RuntimeError(
                    f"experiment completed but no new {algorithm} result directory appeared; new={new_runs}"
                )
            run_name = matching[-1]
            downloaded = download_run(remote_results_root, run_name, rep_dir / algorithm)
            row = comparison_row(downloaded, repetition, args.clients)
            rows.append(row)
            write_csv(batch_dir / "comparison.csv", rows)
            write_markdown(batch_dir / "comparison.md", rows)
            write_csv(batch_dir / "aggregate.csv", aggregate_rows(rows))
            print(f"Collected {algorithm}: {downloaded}")

    print("\nAll requested experiments completed.")
    print(f"Results: {batch_dir}")
    print(f"Comparison: {batch_dir / 'comparison.csv'}")
    print(f"Aggregate: {batch_dir / 'aggregate.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        raise SystemExit(130)
