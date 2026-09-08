#!/usr/bin/env python3
"""Build thesis-ready tables/plots from a benchmark campaign directory."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path
from typing import Any

CORE_RUN_METRICS = [
    "final_test_loss", "final_accuracy", "final_precision", "final_recall",
    "final_f1", "final_macro_f1", "total_network_mib", "total_round_seconds",
    "mean_round_seconds", "mean_training_phase_seconds", "mean_aggregation_seconds",
    "server_wall_seconds", "mean_client_train_seconds", "max_client_train_seconds",
]

def read_json(p: Path) -> dict[str, Any]: return json.loads(p.read_text(encoding="utf-8"))
def read_csv(p: Path) -> list[dict[str,str]]:
    if not p.exists(): return []
    with p.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def num(v: Any) -> float | None:
    if v in (None, ""): return None
    try: return float(v)
    except (TypeError, ValueError): return None

def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def ci95(values: list[float]) -> float:
    if len(values)<2: return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))

def collect(campaign: Path):
    runs=[]; rounds=[]; participants=[]
    for sp in sorted(campaign.rglob("summary.json")):
        rd=sp.parent; summary=read_json(sp); config=read_json(rd/"config.json") if (rd/"config.json").exists() else {}
        req=int(summary.get("requested_clients") or config.get("requested_clients") or 0)
        rep=summary.get("repetition", config.get("repetition")); seed=summary.get("seed", config.get("seed")); alg=summary.get("algorithm", config.get("algorithm"))
        pr=read_csv(rd/"participants.csv")
        client_train=[num(r.get("train_seconds")) for r in pr if r.get("role")=="client"]
        client_train=[v for v in client_train if v is not None]
        fm=summary.get("final_metrics") or {}
        rr={
            "run_id": summary.get("run_id", rd.name), "algorithm": alg, "requested_clients": req,
            "repetition": rep, "seed": seed, "status": summary.get("status"), "failure_reason": summary.get("failure_reason"),
            "completed_rounds": summary.get("completed_rounds"), "git_commit": config.get("git_commit"),
            "final_test_loss": fm.get("test_loss"), "final_accuracy": fm.get("accuracy"), "final_precision": fm.get("precision"),
            "final_recall": fm.get("recall"), "final_f1": fm.get("f1"), "final_macro_f1": fm.get("macro_f1"),
            "final_tp": fm.get("tp"), "final_tn": fm.get("tn"), "final_fp": fm.get("fp"), "final_fn": fm.get("fn"),
            "total_network_bytes": summary.get("total_network_bytes"),
            "total_network_mib": None if summary.get("total_network_bytes") is None else float(summary["total_network_bytes"])/(1024**2),
            "total_training_network_bytes": summary.get("total_training_network_bytes"), "total_evaluation_network_bytes": summary.get("total_evaluation_network_bytes"),
            "total_round_seconds": summary.get("total_round_seconds"), "mean_round_seconds": summary.get("mean_round_seconds"),
            "mean_training_phase_seconds": summary.get("mean_training_phase_seconds"), "mean_aggregation_seconds": summary.get("mean_aggregation_seconds"),
            "server_wall_seconds": summary.get("server_wall_seconds"),
            "mean_client_train_seconds": statistics.mean(client_train) if client_train else None,
            "max_client_train_seconds": max(client_train) if client_train else None,
            "fedprox_mu": (summary.get("algorithm_params") or {}).get("fedprox_mu"),
            "fedpaq_bits": (summary.get("algorithm_params") or {}).get("fedpaq_bits"), "run_dir": str(rd),
        }
        if rr["final_f1"] is not None and rr["total_network_mib"] not in (None,0): rr["f1_per_network_mib"] = float(rr["final_f1"])/float(rr["total_network_mib"])
        runs.append(rr)
        dims={"run_id":rr["run_id"],"algorithm":alg,"requested_clients":req,"repetition":rep,"seed":seed}
        for r in read_csv(rd/"rounds.csv"): rounds.append({**dims, **r})
        for r in pr: participants.append({**dims, **r})
    return runs, rounds, participants

def group_summary(runs):
    out=[]
    keys=sorted({(r["algorithm"],int(r["requested_clients"])) for r in runs if r.get("status")=="completed"}, key=lambda x:(x[1],x[0]))
    for alg,c in keys:
        g=[r for r in runs if r.get("status")=="completed" and r["algorithm"]==alg and int(r["requested_clients"])==c]
        row={"algorithm":alg,"requested_clients":c,"runs":len(g)}
        for m in CORE_RUN_METRICS:
            vals=[num(r.get(m)) for r in g]; vals=[v for v in vals if v is not None]
            row[m+"_mean"]=statistics.mean(vals) if vals else None
            row[m+"_stdev"]=statistics.stdev(vals) if len(vals)>1 else (0.0 if vals else None)
            row[m+"_ci95"]=ci95(vals) if vals else None
        out.append(row)
    return out

def round_summary(rows):
    out=[]
    keys=sorted({(r["algorithm"],int(r["requested_clients"]),int(r["round"])) for r in rows}, key=lambda x:(x[1],x[0],x[2]))
    metrics=["f1","macro_f1","accuracy","test_loss","round_seconds","training_phase_seconds","aggregation_seconds","bytes_total"]
    for alg,c,rnd in keys:
        g=[r for r in rows if r["algorithm"]==alg and int(r["requested_clients"])==c and int(r["round"])==rnd]
        row={"algorithm":alg,"requested_clients":c,"round":rnd,"runs":len(g)}
        for m in metrics:
            vals=[num(x.get(m)) for x in g]; vals=[v for v in vals if v is not None]
            row[m+"_mean"]=statistics.mean(vals) if vals else None; row[m+"_stdev"]=statistics.stdev(vals) if len(vals)>1 else (0.0 if vals else None)
        out.append(row)
    return out

def participant_summary(rows):
    out=[]; client_rows=[r for r in rows if r.get("role")=="client"]
    keys=sorted({(r["algorithm"],int(r["requested_clients"]),r.get("participant_id","")) for r in client_rows}, key=lambda x:(x[1],x[0],x[2]))
    for alg,c,pid in keys:
        g=[r for r in client_rows if r["algorithm"]==alg and int(r["requested_clients"])==c and r.get("participant_id")==pid]
        row={"algorithm":alg,"requested_clients":c,"participant_id":pid,"rows":len(g),"dataset_name":next((x.get("dataset_name") for x in g if x.get("dataset_name")),None)}
        for m in ["train_seconds","final_loss","final_accuracy","test_accuracy","train_positive_rate","test_positive_rate","update_wire_bytes"]:
            vals=[num(x.get(m)) for x in g]; vals=[v for v in vals if v is not None]
            row[m+"_mean"]=statistics.mean(vals) if vals else None; row[m+"_stdev"]=statistics.stdev(vals) if len(vals)>1 else (0.0 if vals else None)
        out.append(row)
    return out

def plots(outdir: Path, groups: list[dict[str,Any]], rounds: list[dict[str,Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception: return []
    created=[]; algs=sorted({r["algorithm"] for r in groups})
    for metric,label,name in [("final_f1_mean","Final anomaly F1","final_f1_vs_clients.png"),("total_network_mib_mean","Network traffic [MiB]","network_vs_clients.png"),("mean_aggregation_seconds_mean","Mean aggregation time [s]","aggregation_time_vs_clients.png")]:
        plt.figure()
        for alg in algs:
            g=sorted([r for r in groups if r["algorithm"]==alg], key=lambda x:int(x["requested_clients"]))
            xs=[int(r["requested_clients"]) for r in g if r.get(metric) is not None]; ys=[float(r[metric]) for r in g if r.get(metric) is not None]
            if xs: plt.plot(xs,ys,marker="o",label=alg)
        plt.xlabel("Number of clients"); plt.ylabel(label); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(outdir/name,dpi=180); plt.close(); created.append(name)
    for c in sorted({int(r["requested_clients"]) for r in rounds}):
        plt.figure()
        for alg in algs:
            g=sorted([r for r in rounds if r["algorithm"]==alg and int(r["requested_clients"])==c], key=lambda x:int(x["round"]))
            xs=[int(r["round"]) for r in g if r.get("f1_mean") is not None]; ys=[float(r["f1_mean"]) for r in g if r.get("f1_mean") is not None]
            if xs: plt.plot(xs,ys,marker="o",label=alg)
        plt.xlabel("Federated round"); plt.ylabel("Anomaly F1"); plt.title(f"Convergence — {c} clients"); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); name=f"convergence_f1_clients_{c}.png"; plt.savefig(outdir/name,dpi=180); plt.close(); created.append(name)
    return created

def rebuild(campaign: Path) -> None:
    runs,rounds,participants=collect(campaign); out=campaign/"analysis"; out.mkdir(exist_ok=True)
    write_csv(out/"runs.csv",runs); write_csv(out/"rounds.csv",rounds); write_csv(out/"participants.csv",participants)
    gs=group_summary(runs); rs=round_summary(rounds); ps=participant_summary(participants)
    write_csv(out/"summary_by_algorithm_clients.csv",gs); write_csv(out/"round_summary.csv",rs); write_csv(out/"participant_summary.csv",ps)
    created=plots(out,gs,rs)
    completed=sum(1 for r in runs if r.get("status")=="completed"); failed=len(runs)-completed
    lines=["# Benchmark campaign analysis","",f"Runs discovered: **{len(runs)}** (completed: **{completed}**, failed/incomplete: **{failed}**).","", "Primary files:", "- `runs.csv` — one row per experiment repetition.", "- `rounds.csv` — all global rounds with campaign dimensions.", "- `participants.csv` — all client/edge measurements.", "- `summary_by_algorithm_clients.csv` — mean, standard deviation and 95% CI.", "- `round_summary.csv` — convergence statistics by round.", "- `participant_summary.csv` — per-device timing/data-distribution summary."]
    if created: lines += ["", "Generated plots:"] + [f"- `{x}`" for x in created]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Analysis rebuilt: {out} ({len(runs)} runs)")

def main():
    p=argparse.ArgumentParser(); p.add_argument("campaign"); a=p.parse_args(); rebuild(Path(a.campaign).resolve())
if __name__=="__main__": main()
