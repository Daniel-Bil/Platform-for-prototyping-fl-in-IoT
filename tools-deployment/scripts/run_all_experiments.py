#!/usr/bin/env python3
"""Run a reproducible thesis benchmark matrix via Ansible and collect results.

This version is hardened for long campaigns:
- retries transient Ansible/SSH failures,
- does not treat a temporary SSH outage as a finished server,
- retries result downloads,
- preserves per-attempt launch logs,
- aborts after repeated infrastructure failures instead of generating hundreds
  of useless failed rows,
- supports resuming an existing campaign without repeating completed runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ALGS = ["FedAvg", "FedProx", "FedPAQ", "FedMA", "HierFedAvg"]


class InfrastructureLaunchError(RuntimeError):
    def __init__(self, message: str, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


def run(cmd, cwd=None, capture=False, check=True):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def run_logged(cmd, log_path: Path, cwd=None) -> int:
    """Run a command, tee combined output to console and to a persistent log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def ssh_once(command: str):
    return subprocess.run(
        ["ssh", "main", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ssh(command: str, retries: int = 5, delay: float = 3.0) -> str:
    last = None
    for attempt in range(1, retries + 1):
        last = ssh_once(command)
        if last.returncode == 0:
            return last.stdout.strip()
        if attempt < retries:
            print(
                f"SSH to main failed (attempt {attempt}/{retries}, rc={last.returncode}); "
                f"retrying in {delay:.0f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise InfrastructureLaunchError(
        f"SSH to main failed after {retries} attempts: "
        f"{(last.stderr if last else '').strip()}",
        returncode=(last.returncode if last else None),
    )


def wait_server(timeout: float, poll: float = 2.0):
    """Wait for the FL server service to finish, tolerating transient SSH outages."""
    end = time.monotonic() + timeout
    consecutive_ssh_failures = 0

    while True:
        if time.monotonic() > end:
            raise TimeoutError("server timeout")

        probe = ssh_once("systemctl is-active --quiet fl-thesis-server.service")
        if probe.returncode == 0:
            consecutive_ssh_failures = 0
            time.sleep(poll)
            continue

        # SSH itself failed. rc=255 is the normal OpenSSH connection-error code.
        # Do NOT interpret this as "the server service stopped".
        if probe.returncode == 255:
            consecutive_ssh_failures += 1
            print(
                f"Temporary SSH failure while waiting for server "
                f"({consecutive_ssh_failures}); retrying...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(10.0, poll * max(1, consecutive_ssh_failures)))
            continue

        # systemctl returns non-zero when the service is inactive/failed.
        break

    result = ssh("systemctl show fl-thesis-server.service -p Result --value || true")
    status = ssh("systemctl show fl-thesis-server.service -p ExecMainStatus --value || true")
    if result and result != "success":
        raise RuntimeError(f"server Result={result} ExecMainStatus={status}")
    if status and status != "0":
        raise RuntimeError(f"server exited {status}")


def run_ansible_with_retries(
    cmd,
    *,
    cwd: Path,
    log_path: Path,
    retries: int,
    retry_delay: float,
):
    """Retry transient Ansible unreachable-host failures (exit code 4)."""
    log_path.write_text("", encoding="utf-8")
    last_rc = None
    for attempt in range(1, retries + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== ANSIBLE ATTEMPT {attempt}/{retries} =====\n")
        rc = run_logged(cmd, log_path, cwd=cwd)
        last_rc = rc
        if rc == 0:
            return

        # Ansible exit code 4 means unreachable host(s). This is infrastructure,
        # not an experiment result, so retry it instead of immediately burning a
        # repetition.
        if rc == 4 and attempt < retries:
            wait = min(60.0, retry_delay * (2 ** (attempt - 1)))
            print(
                f"Ansible reported unreachable host(s) (rc=4). "
                f"Retrying launch in {wait:.0f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
            continue

        break

    raise InfrastructureLaunchError(
        f"ansible-playbook failed with exit status {last_rc}; see {log_path}",
        returncode=last_rc,
    )


def scp_with_retries(source: str, dest: Path, retries: int, delay: float):
    last = None
    for attempt in range(1, retries + 1):
        print(f"\n$ scp -r {source} {dest}", flush=True)
        last = subprocess.run(["scp", "-r", source, str(dest)], text=True)
        if last.returncode == 0:
            return
        if attempt < retries:
            wait = min(30.0, delay * attempt)
            print(
                f"SCP failed (attempt {attempt}/{retries}); retrying in {wait:.0f}s...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    raise InfrastructureLaunchError(
        f"scp failed after {retries} attempts", returncode=(last.returncode if last else None)
    )


def write_status(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_status(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_run(rd: Path, clients: int, rounds: int):
    s = json.loads((rd / "summary.json").read_text())
    if s.get("status") != "completed":
        raise RuntimeError(f"run status={s.get('status')}: {s.get('failure_reason')}")
    if int(s.get("completed_rounds", 0)) != rounds:
        raise RuntimeError(f"completed {s.get('completed_rounds')}/{rounds} rounds")
    with (rd / "rounds.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if int(r.get("logical_client_count") or 0) != clients]
    if bad:
        raise RuntimeError(
            "not all requested logical clients participated in every round: "
            + ", ".join(
                f"r{r['round']}={r.get('logical_client_count')}" for r in bad
            )
        )
    final_metrics = s.get("final_metrics") or {}
    if not final_metrics.get("test_samples"):
        raise RuntimeError("final distributed evaluation is missing")
    if final_metrics.get("selected_threshold") is None:
        raise RuntimeError("validation-selected threshold is missing")
    if not final_metrics.get("validation_samples"):
        raise RuntimeError("distributed validation statistics are missing")


def parse_counts(text):
    vals = []
    for x in text.split(","):
        n = int(x.strip())
        if n < 1:
            raise ValueError
        if n not in vals:
            vals.append(n)
    return vals


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--client-counts", default="2,4,7", help="comma-separated benchmark sizes")
    p.add_argument("--clients", type=int, default=None, help="compatibility shortcut for a single size")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algorithms", default=",".join(ALGS))
    p.add_argument("--model", default="tools-deployment/config/default_model.json",
                   help="model JSON inside the repository (FL Builder/tools2 graph format)")
    p.add_argument("--edge-count", type=int, default=2)
    p.add_argument("--fedprox-mu", type=float, default=.01)
    p.add_argument("--fedpaq-bits", type=int, default=8)
    p.add_argument("--initial-join-window", type=float, default=8.0)
    p.add_argument("--join-window", type=float, default=.25)
    p.add_argument("--threshold-min", type=float, default=0.0)
    p.add_argument("--threshold-max", type=float, default=1.0)
    p.add_argument("--threshold-step", type=float, default=.01)
    p.add_argument("--threshold-preferred", type=float, default=.5)
    p.add_argument("--inventory", default="ansible/inventory.yml")
    p.add_argument("--playbook", default="ansible/playbooks/02-start-one-client.yml")
    p.add_argument("--remote-results-root", default="~/Platform-for-prototyping-fl-in-IoT/tools-deployment/results")
    p.add_argument("--output-root", default="benchmark-results")
    p.add_argument("--experiment-timeout", type=float, default=1800)
    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--resume", default=None)
    p.add_argument("--fail-fast", action="store_true")

    # Long-campaign reliability options.
    p.add_argument("--launch-retries", type=int, default=6,
                   help="retry Ansible unreachable-host failures before marking the run failed")
    p.add_argument("--launch-retry-delay", type=float, default=10.0,
                   help="initial retry delay; exponential backoff is applied")
    p.add_argument("--transfer-retries", type=int, default=5)
    p.add_argument("--infra-abort-threshold", type=int, default=3,
                   help="abort campaign after this many consecutive exhausted infrastructure failures")
    p.add_argument("--analyze-every", type=int, default=10,
                   help="rebuild campaign analysis every N successful experiments; final analysis is always built")
    return p.parse_args()


def main():
    a = args()
    counts = [a.clients] if a.clients else parse_counts(a.client_counts)
    algs = [x.strip() for x in a.algorithms.split(",") if x.strip()]
    if any(x not in ALGS for x in algs):
        raise SystemExit("unsupported algorithm")
    if not (0.0 <= a.threshold_min <= a.threshold_max <= 1.0):
        raise SystemExit("invalid threshold range")
    if a.threshold_step <= 0:
        raise SystemExit("--threshold-step must be > 0")
    if not 0.0 <= a.threshold_preferred <= 1.0:
        raise SystemExit("--threshold-preferred must be in [0,1]")

    root = Path(__file__).resolve().parents[2]
    inv = (root / a.inventory).resolve()
    pb = (root / a.playbook).resolve()

    def resolve_model_path(value: str):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"model JSON not found: {path}")
        try:
            relative = path.relative_to(root)
        except ValueError:
            raise SystemExit(
                "--model must be inside the repository so Ansible can use the same file on main"
            )
        raw = path.read_bytes()
        return path, relative.as_posix(), hashlib.sha256(raw).hexdigest()

    model_path, model_repo_relative, model_sha256 = resolve_model_path(a.model)
    for c in ("ansible-playbook", "ssh", "scp"):
        if not shutil.which(c):
            raise SystemExit(f"missing {c}")

    remote = ssh(f"mkdir -p {a.remote_results_root} && cd {a.remote_results_root} && pwd")

    if a.resume:
        campaign = Path(a.resume).resolve()
        manifest = json.loads((campaign / "manifest.json").read_text())
        campaign_id = manifest["campaign_id"]
        # Resume the exact original matrix, regardless of current CLI defaults.
        counts = [int(x) for x in manifest["client_counts"]]
        algs = list(manifest["algorithms"])
        a.rounds = int(manifest["rounds"])
        a.local_epochs = int(manifest["local_epochs"])
        a.batch_size = int(manifest["batch_size"])
        a.repetitions = int(manifest["repetitions"])
        a.seed = int(manifest["base_seed"])
        a.fedprox_mu = float(manifest["fedprox_mu"])
        a.fedpaq_bits = int(manifest["fedpaq_bits"])
        a.edge_count = int(manifest["edge_count"])
        a.initial_join_window = float(manifest.get("initial_join_window", 8.0))
        a.join_window = float(manifest.get("join_window", .25))
        threshold = manifest.get("threshold_selection") or {}
        a.threshold_min = float(threshold.get("minimum", 0.0))
        a.threshold_max = float(threshold.get("maximum", 1.0))
        a.threshold_step = float(threshold.get("step", .01))
        a.threshold_preferred = float(threshold.get("preferred", .5))
        model_info = manifest.get("model") or {}
        model_repo_relative = model_info.get(
            "repo_relative_path", "tools-deployment/config/default_model.json"
        )
        model_path, model_repo_relative, model_sha256 = resolve_model_path(model_repo_relative)
        expected_model_sha = model_info.get("sha256")
        if expected_model_sha and expected_model_sha != model_sha256:
            raise SystemExit(
                f"model JSON changed since campaign creation: expected {expected_model_sha}, "
                f"got {model_sha256}. Restore the original model before resuming."
            )
    else:
        campaign_id = datetime.now(timezone.utc).strftime("campaign-%Y%m%dT%H%M%SZ")
        campaign = (root / a.output_root / campaign_id).resolve()
        campaign.mkdir(parents=True, exist_ok=False)
        dataset_manifest_path = root / "tools2/data/fl_dataset_real/dataset_manifest.json"
        dataset_info = {"path": str(dataset_manifest_path.relative_to(root))}
        if dataset_manifest_path.exists():
            raw = dataset_manifest_path.read_bytes()
            dataset_info.update(json.loads(raw.decode("utf-8")))
            dataset_info["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = {
            "campaign_id": campaign_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "algorithms": algs,
            "client_counts": counts,
            "rounds": a.rounds,
            "local_epochs": a.local_epochs,
            "batch_size": a.batch_size,
            "repetitions": a.repetitions,
            "base_seed": a.seed,
            "fedprox_mu": a.fedprox_mu,
            "fedpaq_bits": a.fedpaq_bits,
            "edge_count": a.edge_count,
            "initial_join_window": a.initial_join_window,
            "join_window": a.join_window,
            "threshold_selection": {
                "objective": "validation_macro_f1",
                "minimum": a.threshold_min,
                "maximum": a.threshold_max,
                "step": a.threshold_step,
                "preferred": a.threshold_preferred,
            },
            "model": {
                "name": model_path.stem,
                "repo_relative_path": model_repo_relative,
                "sha256": model_sha256,
            },
            "dataset": dataset_info,
        }
        (campaign / "manifest.json").write_text(json.dumps(manifest, indent=2))
        shutil.copy2(model_path, campaign / "model.json")

    status_path = campaign / "campaign_status.csv"
    statuses = load_status(status_path)
    done = {
        (r.get("algorithm"), int(r.get("requested_clients", 0)), int(r.get("repetition", 0)))
        for r in statuses
        if r.get("status") == "completed"
    }

    successful_since_analysis = 0
    consecutive_infra_failures = 0

    try:
        for n in counts:
            for rep in range(1, a.repetitions + 1):
                seed = a.seed + rep - 1
                for alg in algs:
                    key = (alg, n, rep)
                    if key in done:
                        print(f"SKIP completed {key}")
                        continue

                    base_run_id = f"{campaign_id}_c{n:02d}_r{rep:02d}_s{seed}_{alg}"
                    previous_attempts = sum(
                        1
                        for x in statuses
                        if x.get("algorithm") == alg
                        and int(x.get("requested_clients", 0)) == n
                        and int(x.get("repetition", 0)) == rep
                    )
                    run_id = (
                        base_run_id
                        if previous_attempts == 0
                        else f"{base_run_id}-a{previous_attempts + 1}"
                    )
                    dest = campaign / f"clients-{n:02d}" / f"rep-{rep:02d}-seed-{seed}" / alg
                    dest.mkdir(parents=True, exist_ok=True)
                    launch_log = campaign / "launch-logs" / f"{run_id}.log"

                    rec = {
                        "run_id": run_id,
                        "algorithm": alg,
                        "model": model_path.stem,
                        "requested_clients": n,
                        "repetition": rep,
                        "seed": seed,
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "running",
                        "error": "",
                        "launch_log": str(launch_log),
                    }
                    statuses.append(rec)
                    write_status(status_path, statuses)

                    try:
                        ev = [
                            f"fl_client_count={n}",
                            f"fl_requested_clients={n}",
                            f"fl_algorithm={alg}",
                            f"fl_model_repo_relative={model_repo_relative}",
                            f"fl_rounds={a.rounds}",
                            f"fl_local_epochs={a.local_epochs}",
                            f"fl_batch_size={a.batch_size}",
                            f"fl_seed={seed}",
                            f"fl_fedprox_mu={a.fedprox_mu}",
                            f"fl_fedpaq_bits={a.fedpaq_bits}",
                            f"fl_edge_count={min(a.edge_count, n)}",
                            f"fl_initial_join_window={a.initial_join_window}",
                            f"fl_join_window={a.join_window}",
                            f"fl_threshold_min={a.threshold_min}",
                            f"fl_threshold_max={a.threshold_max}",
                            f"fl_threshold_step={a.threshold_step}",
                            f"fl_threshold_preferred={a.threshold_preferred}",
                            f"fl_run_id={run_id}",
                            f"fl_campaign_id={campaign_id}",
                            f"fl_repetition={rep}",
                        ]
                        cmd = ["ansible-playbook", "-i", str(inv), str(pb)]
                        for v in ev:
                            cmd += ["-e", v]

                        run_ansible_with_retries(
                            cmd,
                            cwd=root,
                            log_path=launch_log,
                            retries=a.launch_retries,
                            retry_delay=a.launch_retry_delay,
                        )
                        wait_server(a.experiment_timeout)
                        scp_with_retries(
                            f"main:{remote}/{run_id}",
                            dest,
                            retries=a.transfer_retries,
                            delay=3.0,
                        )
                        rd = dest / run_id
                        validate_run(rd, n, a.rounds)

                        rec["status"] = "completed"
                        rec["result_dir"] = str(rd)
                        done.add(key)
                        consecutive_infra_failures = 0
                        successful_since_analysis += 1
                    except InfrastructureLaunchError as e:
                        rec["status"] = "failed"
                        rec["error"] = f"{type(e).__name__}: {e}"
                        rec["infrastructure_failure"] = "true"
                        consecutive_infra_failures += 1
                        print(f"INFRA FAILED {run_id}: {rec['error']}", file=sys.stderr)
                    except Exception as e:
                        rec["status"] = "failed"
                        rec["error"] = f"{type(e).__name__}: {e}"
                        rec["infrastructure_failure"] = "false"
                        consecutive_infra_failures = 0
                        print(f"FAILED {run_id}: {rec['error']}", file=sys.stderr)

                    rec["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    write_status(status_path, statuses)

                    if rec["status"] == "failed" and a.fail_fast:
                        raise RuntimeError(rec["error"])

                    if (
                        rec.get("infrastructure_failure") == "true"
                        and consecutive_infra_failures >= a.infra_abort_threshold
                    ):
                        raise InfrastructureLaunchError(
                            f"aborting campaign after {consecutive_infra_failures} consecutive "
                            "infrastructure failures. Fix SSH/host availability and resume the "
                            f"same campaign with --resume {campaign}"
                        )

                    if successful_since_analysis >= max(1, a.analyze_every):
                        subprocess.run(
                            [
                                sys.executable,
                                str(root / "tools-deployment/scripts/analyze_campaign.py"),
                                str(campaign),
                            ],
                            check=False,
                        )
                        successful_since_analysis = 0

                    time.sleep(a.cooldown)
    finally:
        # Always leave whatever has completed in an analysis-ready state.
        subprocess.run(
            [
                sys.executable,
                str(root / "tools-deployment/scripts/analyze_campaign.py"),
                str(campaign),
            ],
            check=False,
        )

    print(f"\nCampaign finished: {campaign}")
    print(f"Analysis: {campaign / 'analysis'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
