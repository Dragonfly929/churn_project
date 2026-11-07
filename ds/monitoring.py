"""Step 9 — monitoring: data drift (PSI), performance decay, explanation drift, fairness re-check,
retraining rule, model card."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

import config
from ds import plots
from ds.modeling import add_groups, fairness_gaps, fairness_table
from ds.xai import global_importance, psi_table, shap_frame
from stage1.extract_dataset import run_query

MONTHS = [("2025-07-31", "2025-10-29"), ("2025-08-31", "2025-11-29"), ("2025-09-30", "2025-12-29"),
          ("2025-10-31", "2026-01-29"), ("2025-11-30", "2026-02-28"), ("2025-12-31", "2026-03-31"),
          ("2026-01-31", "2026-05-01"), ("2026-02-28", "2026-05-29"), ("2026-03-31", "2026-06-29")]


def monitor(champion: dict, train_ref: pd.DataFrame, months=MONTHS) -> dict:
    rows, drift_rows = [], []
    feats = champion["features"]
    for cut, lab in months:
        d = add_groups(run_query(cut, lab))
        p = champion["pipe"].predict_proba(d[feats])[:, 1]
        pred = (p >= champion["threshold"]).astype(int)
        ps = psi_table(train_ref, d, feats)
        sdf, _ = shap_frame(champion["pipe"], d, feats, max_rows=300)
        ft = fairness_table(d, d["churn"].values, pred, p)
        gaps = fairness_gaps(ft)
        rows.append({"month_cutoff": cut, "clients": len(d), "churn_rate": d["churn"].mean(),
                     "roc_auc": roc_auc_score(d["churn"], p), "recall": recall_score(d["churn"], pred),
                     "max_psi": ps["psi"].max(), "feature_max_psi": ps.iloc[0]["feature"],
                     "shap_top5": ", ".join(global_importance(sdf).index[:5]),
                     "recall_gap_age": gaps.get("age_group", np.nan)})
        for _, r in ps.iterrows():
            drift_rows.append({"month_cutoff": cut, "feature": r["feature"], "psi": r["psi"]})
    t = pd.DataFrame(rows)
    ref_top = set(t.loc[t.month_cutoff == config.CUTOFF, "shap_top5"].iloc[0].split(", "))
    t["shap_top5_jaccard_vs_training"] = [len(ref_top & set(s.split(", "))) / len(ref_top | set(s.split(", "))) for s in t["shap_top5"]]
    t["RETRAIN?"] = np.where((t["max_psi"] >= 0.2) | (t["recall"] < config.CONTRACT["M1_recall_min"] - 0.1) |
                             (t["recall_gap_age"] > config.CONTRACT["fairness_gap_max"] + 0.1), "yes", "no")
    f, axes = plots.fig(12, 3.6, 3, 1)
    axes[0].plot(t["month_cutoff"], t["roc_auc"], marker="o", color=plots.PALETTE[0], label="ROC-AUC")
    axes[0].plot(t["month_cutoff"], t["recall"], marker="o", color=plots.PALETTE[1], label="recall")
    axes[0].set_title("performance by month"); axes[0].legend()
    axes[1].plot(t["month_cutoff"], t["max_psi"], marker="o", color=plots.PALETTE[2])
    axes[1].axhline(0.2, color=plots.GRAY, ls="--"); axes[1].set_title("max PSI (data drift)")
    axes[2].plot(t["month_cutoff"], t["recall_gap_age"], marker="o", color=plots.PALETTE[3])
    axes[2].set_title("recall gap by age group")
    for a in axes:
        a.tick_params(axis="x", rotation=60)
    plots.save(f, "m9_monitoring")
    return {"table": t, "drift": pd.DataFrame(drift_rows)}


def model_card(champion: dict, contract: pd.DataFrame, owner: str = "student (ȘAD)") -> str:
    m = champion.get("metrics", {})
    lines = [
        f"# Model card — churn early warning ({champion.get('name', 'champion')})",
        f"- Date: {date.today().isoformat()}   - Owner: {owner}",
        "- Purpose: rank active clients by risk of leaving in the next 90 days and propose a retention action. "
        "The model only PROPOSES; an employee decides (GDPR Art. 22).",
        "- Data: simulated bank DWH (star schema), Stage 1 query at cutoff "
        f"{config.CUTOFF}, label window to {config.LABEL_CUTOFF}. Synthetic data: no real client is processed.",
        f"- Model: {champion.get('config', {}).get('model')} with balancing "
        f"'{champion.get('config', {}).get('balancing')}', {len(champion['features'])} features: "
        + ", ".join(champion["features"]),
        f"- Decision threshold (from the cost matrix FN={config.COST_FN:.0f} €, FP={config.COST_FP:.0f} €): "
        f"{champion['threshold']:.2f}; per-group thresholds: {champion.get('group_thresholds') or 'none'}",
        f"- Test metrics: ROC-AUC {m.get('roc_auc', float('nan')):.3f}, recall {m.get('recall', float('nan')):.3f}, "
        f"precision {m.get('precision', float('nan')):.3f}, expected cost {m.get('expected_cost', float('nan')):.0f} € "
        f"vs {m.get('cost_no_model', float('nan')):.0f} € without model",
        "- Contract matrix:",
        "",
        contract[["requirement", "threshold", "measured", "verdict"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "- Limits: simulated clients and simulated reactions to offers; the sentiment lexicon was built for the "
        "message templates of the simulator; fairness measured only for age group, gender, country.",
        "- Monitoring: monthly PSI per feature (alert >= 0.2), recall and lead time when exits are known, SHAP top-5 "
        "drift, recall gap by group; retrain when any crosses its threshold, then champion vs challenger by the "
        "contract matrix.",
    ]
    txt = "\n".join(lines)
    (config.REPORTS / "model_card.md").write_text(txt, encoding="utf-8")
    return txt
