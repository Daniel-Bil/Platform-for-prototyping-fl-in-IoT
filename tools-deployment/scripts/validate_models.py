#!/usr/bin/env python3
"""Build every FL Builder JSON model with the deployment parser before a campaign."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "tools-deployment"
if str(DEPLOYMENT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT))

from common.model import build_model_from_config, load_model_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="tools-deployment/config/models")
    p.add_argument("--seq-len", type=int, default=6)
    p.add_argument("--features", type=int, default=4)
    a = p.parse_args()

    models_dir = (ROOT / a.models_dir).resolve()
    files = sorted(models_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no JSON models found in {models_dir}")

    failed = 0
    for path in files:
        try:
            cfg = load_model_config(path)
            model = build_model_from_config(cfg, a.seq_len, a.features)
            print(f"OK   {path.name:32s} params={model.count_params():,} output={model.output_shape}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {path.name:32s} {type(exc).__name__}: {exc}")

    if failed:
        raise SystemExit(f"{failed}/{len(files)} model(s) failed validation")
    print(f"Validated {len(files)} model(s).")


if __name__ == "__main__":
    main()
