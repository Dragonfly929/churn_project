"""Step 8 — the cycle: one iteration = train -> XAI -> external check -> contract matrix -> verdict,
saved to the iteration log so iterations can be compared side by side."""
from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import config
from ds.catalog import validate_contract
from ds.modeling import RunConfig, contract_matrix, make_pipeline, run
from ds.xai import compare_importance, global_importance, psi_table, shap_frame, stability

LOG = config.ARTIFACTS / "iterations.json"


def evaluate_iteration(cfg: RunConfig, df: pd.DataFrame, ext: pd.DataFrame | None = None, n_stab: int = 4,
                       thresholds: dict | None = None) -> dict:
    res = run(cfg, df)
    sdf, _ = shap_frame(res.pipe, res.test, cfg.features, background=res.train, max_rows=600)
    imp = global_importance(sdf)
    jac, tops = stability(cfg, res.train, n=n_stab)
    extra = {"jaccard": jac, "data_contract": bool(validate_contract(df)["pass"].all())}
    ext_info = {}
    if ext is not None and len(ext):
        p_ext = res.pipe.predict_proba(ext[cfg.features])[:, 1]
        from sklearn.metrics import roc_auc_score, recall_score
        s_ext, _ = shap_frame(res.pipe, ext, cfg.features, background=res.train, max_rows=600)
        imp_ext = global_importance(s_ext)
        cmp = compare_importance(imp, imp_ext)
        ps = psi_table(res.train, ext, cfg.features)
        extra.update({"rank_corr": cmp["rank_corr"], "ext_auc": roc_auc_score(ext["churn"], p_ext),
                      "max_psi": float(ps["psi"].max())})
        ext_info = {"auc": extra["ext_auc"], "recall": recall_score(ext["churn"], (p_ext >= res.threshold).astype(int)),
                    "rank_corr": cmp["rank_corr"], "top5_ext": cmp["top5_ext"], "psi": ps, "imp_ext": imp_ext}
    cm = contract_matrix(res, extra, thresholds)
    return {"res": res, "shap": sdf, "importance": imp, "stability_tops": tops, "jaccard": jac, "ext": ext_info,
            "contract": cm, "verdict": "PASS" if cm["pass"].all() else "FAIL"}


def load_log() -> list:
    if LOG.exists():
        return json.loads(LOG.read_text(encoding="utf-8"))
    return []


def save_iteration(it: dict, reason: str, name: str | None = None) -> dict:
    log = load_log()
    r = it["res"]
    entry = {
        "id": len(log) + 1, "name": name or f"iteration {len(log) + 1}",
        "saved_at": datetime.now().isoformat(timespec="seconds"), "reason_for_change": reason,
        "config": {k: v for k, v in r.cfg.to_dict().items()},
        "threshold": r.threshold, "group_thresholds": r.group_thr,
        "metrics": {k: float(v) for k, v in r.metrics.items()},
        "fairness_gaps": r.gaps, "jaccard_top5": it["jaccard"],
        "shap_top5": it["importance"].index[:5].tolist(),
        "external": {k: v for k, v in it["ext"].items() if k in ("auc", "recall", "rank_corr", "top5_ext")},
        "contract": it["contract"].assign(measured=it["contract"]["measured"].astype(float)).to_dict("records"),
        "verdict": it["verdict"],
    }
    entry = json.loads(json.dumps(entry, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log.append(entry)
    LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def save_champion(it: dict, name: str):
    r = it["res"]
    champ = {"pipe": r.pipe, "features": r.cfg.features, "threshold": r.threshold, "group_thresholds": r.group_thr,
             "name": name, "config": r.cfg.to_dict(), "metrics": r.metrics}
    joblib.dump(champ, config.ARTIFACTS / "champion.joblib")
    # challengers for shadow mode (lesson 4): L1 / L2 / ElasticNet logistic regression
    ch = {}
    for nm, pen, l1r in [("L1", "l1", 1.0), ("L2", "l2", 0.0), ("ElasticNet", "elasticnet", 0.5)]:
        c = RunConfig(**{**r.cfg.to_dict(), "model": "logreg", "balancing": "none",
                         "params": {"C": 0.1, "penalty": pen, "l1_ratio": l1r}})
        pipe = make_pipeline(c).fit(r.train[c.features], r.train["churn"])
        from ds.modeling import best_threshold
        p = pipe.predict_proba(r.train[c.features])[:, 1]
        ch[nm] = {"pipe": pipe, "features": c.features, "threshold": best_threshold(r.train["churn"].values, p)}
    joblib.dump(ch, config.ARTIFACTS / "challengers.joblib")
    return champ


def compare_table(entries: list) -> pd.DataFrame:
    rows = {}
    for e in entries:
        col = f"#{e['id']} {e['name']}"
        cfg = e["config"]
        d = {"reason for change": e["reason_for_change"], "model": cfg["model"], "balancing": cfg["balancing"],
             "n features": len(cfg["features"]),
             "sensitive attributes used": ", ".join(f for f in config.SENSITIVE if f in cfg["features"]) or "none",
             "per-group thresholds": cfg["per_group_thresholds"], "threshold": round(e["threshold"], 3)}
        for k in ("recall", "precision", "roc_auc", "expected_cost"):
            d[k] = round(e["metrics"][k], 3)
        for k, v in e["fairness_gaps"].items():
            d[f"recall gap {k}"] = round(v, 3)
        d["Jaccard top-5 (F1)"] = round(e["jaccard_top5"], 3)
        d["external AUC"] = round(e["external"].get("auc", np.nan), 3) if e["external"] else None
        d["external SHAP rank corr (F2)"] = round(e["external"].get("rank_corr", np.nan), 3) if e["external"] else None
        d["SHAP top-5"] = ", ".join(e["shap_top5"])
        for c in e["contract"]:
            d[f"[contract] {c['requirement']}"] = c["verdict"]
        d["VERDICT"] = e["verdict"]
        rows[col] = d
    return pd.DataFrame(rows).astype(str)


def features_entered_left(a: dict, b: dict) -> dict:
    fa, fb = set(a["config"]["features"]), set(b["config"]["features"])
    sa, sb = set(a["shap_top5"]), set(b["shap_top5"])
    return {"features removed": sorted(fa - fb), "features added": sorted(fb - fa),
            "entered SHAP top-5": sorted(sb - sa), "left SHAP top-5": sorted(sa - sb)}
