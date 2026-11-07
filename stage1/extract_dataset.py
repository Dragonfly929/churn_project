"""Steps 1.6-1.7 — Run THE QUERY on the DWH and save the dataset.

  python stage1/extract_dataset.py                       # main dataset (cutoff 2025-12-31)
  python stage1/extract_dataset.py --source branch       # external branch, same query
  python stage1/extract_dataset.py --leaky               # WRONG cutoff (leakage demo)
  python stage1/extract_dataset.py --cutoff 2025-09-30 --label-cutoff 2025-12-29 --name oot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from stage1 import db  # noqa: E402


def run_query(cutoff=config.CUTOFF, label_cutoff=config.LABEL_CUTOFF, feature_cutoff=None,
              denoise=1, source="bank") -> pd.DataFrame:
    params = {"cutoff": cutoff, "label_cutoff": label_cutoff,
              "feature_cutoff": feature_cutoff or cutoff, "denoise": int(denoise)}
    df = db.read_sql(db.sql_file("query_features"), params, source)
    for c in df.columns:
        if c not in ("gender", "country"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def dataset_path(name: str) -> Path:
    return config.DATASETS / f"{name}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bank")
    ap.add_argument("--cutoff", default=config.CUTOFF)
    ap.add_argument("--label-cutoff", default=config.LABEL_CUTOFF)
    ap.add_argument("--leaky", action="store_true", help="features computed AFTER the departures")
    ap.add_argument("--raw-text", action="store_true", help="sentiment without denoising")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    fcut = config.LEAKY_FEATURE_CUTOFF if a.leaky else a.cutoff
    df = run_query(a.cutoff, a.label_cutoff, fcut, 0 if a.raw_text else 1, a.source)
    name = a.name or (f"{a.source}_leaky" if a.leaky else f"{a.source}_{a.cutoff}")
    df.to_csv(dataset_path(name), index=False)
    print(f"dataset {name}: {df.shape[0]} rows x {df.shape[1] - 2} features "
          f"(+client_id, churn); churn share = {df['churn'].mean():.3f} -> {dataset_path(name)}")


if __name__ == "__main__":
    main()
