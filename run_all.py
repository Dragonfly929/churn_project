"""ONE COMMAND rebuilds everything (Step 9.2 reproducibility), fixed seeds:

  python run_all.py            # full run (~6-10 min on a laptop)
  python run_all.py --fast     # fewer repetitions (~4-5 min)
  python run_all.py --skip-data  # reuse the DWH already built

Order = the research cycle: problem/KPIs -> source systems -> staging -> ETL -> DWH -> query ->
dataset -> EDA -> contract matrix -> iterations (loop) -> models -> XAI (pass 1, pass 2) ->
six fundamentals -> lab experiments -> monitoring -> model card -> reports/results_summary.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PY = sys.executable
MD: list[str] = []


def sh(*args):
    print("$", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=config.ROOT)


def section(title):
    print(f"\n=== {title} ===", flush=True)
    MD.append(f"\n## {title}\n")


def table(df, name, index=False, floatfmt=".3f"):
    df.to_csv(config.TABLES / f"{name}.csv", index=index)
    MD.append(f"*Table: {name}*\n\n" + df.to_markdown(index=index, floatfmt=floatfmt) + "\n")


def text(s):
    MD.append(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args()
    reps = 10 if a.fast else 100
    t0 = time.time()

    from ds.catalog import KPIS, PROBLEM, FEATURES_DF, DATA_CONTRACT, validate_contract
    MD.append(f"# Results summary — „Loialitate, tăcere și plecare”\n\nEngine: **{config.DB_BACKEND}**, "
              f"cutoff {config.CUTOFF}, label window to {config.LABEL_CUTOFF}, seed {config.SEED}. "
              "All numbers below were MEASURED by run_all.py — use these in the report, never example numbers.\n")
    section("1.1-1.2 Problem, goal, objectives, KPIs (defined before any data)")
    text(PROBLEM["problem"] + "\n\n**Goal:** " + PROBLEM["goal"] + "\n\n" + "\n".join(f"- {o}" for o in PROBLEM["objectives"]))
    table(KPIS, "kpis")
    section("1.3-1.4 Features and data contract")
    table(FEATURES_DF, "features_catalog")
    table(DATA_CONTRACT, "data_contract")

    # ------------------------------------------------------------ Stage 1 data
    if not a.skip_data:
        section("1.5 Source systems -> staging -> ETL -> DWH")
        sh("stage1/generator.py"); sh("stage1/generator.py", "--branch")
        sh("stage1/etl.py"); sh("stage1/etl.py", "--source", "branch")
    sh("stage1/extract_dataset.py")
    sh("stage1/extract_dataset.py", "--leaky")
    sh("stage1/extract_dataset.py", "--raw-text", "--name", "bank_rawtext")
    sh("stage1/extract_dataset.py", "--cutoff", config.OOT_CUTOFF, "--label-cutoff", config.OOT_LABEL_CUTOFF, "--name", "bank_oot")
    sh("stage1/extract_dataset.py", "--source", "branch")
    sh("realtime/parity_test.py")
    for s in ("bank", "branch"):
        p = config.TABLES / f"cleaning_report_{s}.csv"
        if p.exists():
            table(pd.read_csv(p), f"cleaning_report_{s}")

    from ds.modeling import (RunConfig, add_groups, load_dataset, run, make_pipeline, split, cost_matrix_table,
                             metrics, best_threshold)
    from ds import eda, fundamentals as F, fundamentals_text as FT, experiments as E, xai, rtap, streaming_sim, monitoring, cycle
    df = load_dataset(f"bank_{config.CUTOFF}")
    ext = load_dataset(f"branch_{config.CUTOFF}")
    section("1.7 The dataset (output of the query)")
    text(f"Shape: **{df.shape[0]} clients × {len([c for c in df.columns if c not in ('client_id','churn','marketing_opt_out','age_group','silent')])} features**, "
         f"churn share **{df.churn.mean():.3f}**. External branch: {ext.shape[0]} clients, churn share {ext.churn.mean():.3f}.")
    table(validate_contract(df), "data_contract_check")

    # ------------------------------------------------------------ EDA
    section("2. EDA (course steps)")
    text(f"Step 0 — unique identifier: {eda.step0_unique_id(df)}")
    table(eda.step1_missing(df), "eda_missing", index=True)
    table(eda.step2_types(df), "eda_types")
    seg, low = eda.step3_shape(df)
    for k, v in seg.items():
        table(v.reset_index(), f"eda_churn_by_{k}")
    text(f"Near-zero-variance features: {low}")
    eda.fig_distributions(df); eda.fig_segments(df)
    tgt, corr, prox = eda.step4_correlations(df)
    eda.fig_corr(corr)
    table(tgt.reset_index().rename(columns={"index": "feature"}), "eda_corr_with_churn")
    if len(prox):
        table(prox, "eda_proxies")
    table(eda.step5_outliers(df), "eda_outliers")
    table(pd.DataFrame([eda.kpis_and_hypotheses(df)]).T.reset_index().rename(columns={"index": "KPI / hypothesis", 0: "value"}),
          "eda_kpis_hypotheses")
    section("2. Leakage trap (wrong cutoff)")
    table(eda.leakage_demo(df, load_dataset("bank_leaky")).reset_index(), "leakage_demo")

    # ------------------------------------------------------------ contract + loop
    section("3.1 Contract matrix: cost matrix")
    table(cost_matrix_table().reset_index(), "cost_matrix")
    section("3.5 / 8 The loop: iterations")
    log = cycle.LOG
    if log.exists():
        log.unlink()
    from ds.catalog import ALL_FEATURES
    def failed(it):
        return it["contract"][~it["contract"]["pass"]]["requirement"].tolist()

    it1_cfg = RunConfig(model="gb", balancing="none", features=list(ALL_FEATURES))
    it1 = cycle.evaluate_iteration(it1_cfg, df, ext)
    cycle.save_iteration(it1, "baseline: all features incl. sensitive attributes, cost-optimal threshold", "iteration 1")
    best = it1
    if it1["verdict"] == "FAIL":
        nonsens = [f for f in ALL_FEATURES if f not in config.SENSITIVE + ["digital_share_90d"]]
        it2_cfg = RunConfig(model="gb", balancing="none", features=nonsens)
        it2 = cycle.evaluate_iteration(it2_cfg, df, ext)
        cycle.save_iteration(it2, f"iteration 1 failed {failed(it1)}: removed age/gender/country (non-discrimination) and "
                             "digital_share_90d (proxy for age AND the feature drifting on the external branch)",
                             "iteration 2")
        best = it2
        if it2["verdict"] == "FAIL":
            it3_cfg = RunConfig(**{**it2_cfg.to_dict(), "threshold_mode": "cost+recall"})
            it3 = cycle.evaluate_iteration(it3_cfg, df, ext)
            cycle.save_iteration(it3, f"iteration 2 failed {failed(it2)}: one threshold for all clients = the cheapest "
                                 f"one whose out-of-fold recall is at least {it3_cfg.min_recall:.2f} (contract 0.70 + "
                                 "safety margin)", "iteration 3")
            best = it3 if it3["contract"]["pass"].sum() >= it2["contract"]["pass"].sum() else it2
    entries = cycle.load_log()
    table(cycle.compare_table(entries).reset_index().rename(columns={"index": "item"}), "iterations_side_by_side")
    text(f"Feature changes 1 -> 2: {cycle.features_entered_left(entries[0], entries[1])}")
    for i, e in enumerate(entries):
        table(pd.DataFrame(e["contract"]), f"contract_matrix_iteration_{i + 1}")
    best_id = [i for i, e in enumerate(entries) if e["verdict"] == best["verdict"]][-1] + 1
    champ = cycle.save_champion(best, f"iteration {best_id}")
    text(f"Champion = iteration {best_id}, verdict **{best['verdict']}**")
    base = RunConfig(**{**best["res"].cfg.to_dict(), "per_group_thresholds": False})

    # ------------------------------------------------------------ models, tuning, OOT
    section("3.2-3.3 Models compared, tuning, out-of-time")
    B = F.boundaries(df, RunConfig(**{**base.to_dict(), "features": list(ALL_FEATURES)}))
    table(B["table"], "models_comparison")
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    tr, te = split(df)
    rs = RandomizedSearchCV(make_pipeline(base), {"clf__learning_rate": [0.02, 0.05, 0.1], "clf__max_depth": [2, 3, 4, 6],
                                                   "clf__max_iter": [150, 300, 500], "clf__l2_regularization": [0.0, 0.5, 2.0]},
                            n_iter=8 if a.fast else 20, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED),
                            random_state=config.SEED, n_jobs=-1).fit(tr[base.features], tr["churn"])
    text(f"Hyperparameter tuning (CV on training only): best {rs.best_params_}, CV AUC {rs.best_score_:.3f}")
    oot = load_dataset("bank_oot")
    pipe = make_pipeline(base).fit(oot[base.features], oot["churn"])
    p = pipe.predict_proba(df[base.features])[:, 1]
    po = pipe.predict_proba(oot[base.features])[:, 1]
    thr = best_threshold(oot["churn"].values, po)
    table(pd.DataFrame([{"setup": "random split (main cutoff)", **{k: best['res'].metrics[k] for k in ('roc_auc', 'recall', 'precision')}},
                        {"setup": f"out-of-time: train {config.OOT_CUTOFF} -> test {config.CUTOFF}",
                         **{k: metrics(df['churn'], p, thr)[k] for k in ('roc_auc', 'recall', 'precision')}}]), "out_of_time")
    table(B["complexity"], "complexity_curve")

    # ------------------------------------------------------------ XAI
    section("4. XAI pass 1 (our data) and pass 2 (external branch)")
    r = best["res"]
    imp = best["importance"]
    table(imp.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}), "shap_global")
    import shap
    import matplotlib.pyplot as plt
    from ds import plots
    Xb = r.test.loc[best["shap"].index, base.features].copy()
    for c in config.CATEGORICAL:
        if c in Xb:
            Xb[c] = Xb[c].astype("category").cat.codes
    plt.figure(); shap.summary_plot(best["shap"].values, Xb, show=False)
    plots.save(plt.gcf(), "x4_shap_beeswarm"); plt.close("all")
    clients = xai.pick_contrasting_clients(r.test, r.p_test)
    for lab, idx in clients.items():
        s, _ = xai.shap_frame(r.pipe, r.test.loc[[idx]], base.features)
        row = s.iloc[0]
        lime_df = xai.lime_explain(r.pipe, r.train, r.test.loc[idx], base.features)
        reason, _ = xai.reason_code(row)
        agree = xai.shap_lime_agreement(row, lime_df)
        text(f"**{lab}** (client {r.test.loc[idx, 'client_id']}, risk {r.p_test[list(r.test.index).index(idx)]:.2f}) — reason code: **{reason}**; "
             f"SHAP–LIME agreement: {agree}")
        table(row.sort_values(key=np.abs, ascending=False).head(8).reset_index().rename(columns={"index": "feature", row.name: "shap"}),
              f"shap_local_{lab.replace(' ', '_')}")
        table(lime_df, f"lime_{lab.replace(' ', '_')}")
    text(f"Stability (KPI F1): mean Jaccard of top-5 = **{best['jaccard']:.3f}**; tops = {best['stability_tops']}")
    if best["ext"]:
        text(f"External branch (KPI F2): AUC {best['ext']['auc']:.3f}, recall {best['ext']['recall']:.3f}, "
             f"SHAP rank correlation {best['ext']['rank_corr']:.3f}, top-5 external {best['ext']['top5_ext']}")
        table(best["ext"]["psi"], "psi_external")

    # ------------------------------------------------------------ six fundamentals
    section("5. Six fundamentals")
    results = {}
    results["bias"] = F.bias(df, RunConfig(**{**base.to_dict(), "features": list(ALL_FEATURES)}))
    results["noise"] = F.noise(df, load_dataset("bank_rawtext"), base, reps=1 if a.fast else 3)
    results["boundaries"] = B
    results["smote"] = F.smote(df, RunConfig(**{**base.to_dict(), "features": list(ALL_FEATURES)}))  # with gender/country so SMOTE-NC differs from SMOTE
    results["streaming"] = streaming_sim.run_streaming_fundamental(champ)
    results["rtap"] = rtap.rtap_fundamental(results["streaming"]["alerts"], results["streaming"]["truth"])
    for k in FT.ORDER:  # the report template's order, the five parts of each fundamental
        o = results[k]
        text(f"### {FT.TITLES[k]}")
        text(f"**1. How it appears.** {FT.APPEARS[k]}")
        text("**2. Measured.**")
        table(o["table"], f"fund_{k}")
        for extra in ("text_table", "text_model", "confusion", "deciles", "realism", "subset_noise", "learned", "fair"):
            if extra in o:
                table(o[extra].reset_index() if extra in ("text_table", "deciles", "fair") else o[extra], f"fund_{k}_{extra}")
        text("**3. Observed.**\n" + "\n".join(f"- {s}" for s in FT.observed(k, o)))
        text("**4. Mitigation.**\n" + "\n".join(f"- {m}" for m in o["measures"]))
        text("**5. Răspunsuri.**\n" + "\n".join(f"- *{q}* {a}" for q, a in FT.ANSWERS[k]))

    # ------------------------------------------------------------ lab experiments
    section("6. Lab experiments")
    table(E.experiment_1_2(df, reps=reps), "exp12_noise_bias_xy")
    text(f"Text embeddings + PCA: {E.pca_text()}")
    text(f"MLP hidden layer + PCA: {E.pca_mlp_hidden(df)}")
    from stage1.extract_dataset import run_query
    batches = {c: add_groups(run_query(c, (pd.Timestamp(c) + pd.Timedelta(days=90)).strftime('%Y-%m-%d')))
               for c in ["2025-06-30", "2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31"]}
    reg = E.regularization(df, batches)
    table(reg["coefficients"].reset_index().rename(columns={"index": "feature"}), "reg_coefficients")
    text(f"Features set to 0 by L1 (C=0.05): {reg['zeros_L1']}")
    table(reg["shap_top5"], "reg_shap_top5")
    table(reg["sweep"], "reg_sweep")
    table(E.experiment_a(), "expA_toy")
    from stage1 import github_load
    srcs = {"simulated bank": df[["client_id"] + github_load.COMMON + ["churn"]],
            "external branch": ext[["client_id"] + github_load.COMMON + ["churn"]]}
    if github_load.available():
        k = github_load.load()
        srcs["github (real)"] = k
        resb, syn = E.experiment_b(k.drop(columns=["geography"]), "github")
    else:
        text("github file not found in data/external/ -> Experiment B runs on the simulated bank's profile features.")
        resb, syn = E.experiment_b(srcs["simulated bank"], "simulated bank")
    srcs["learned synthetic"] = syn
    table(pd.DataFrame([resb]), "expB_synthetic")
    mat, keep, common = E.sources_matrix(srcs)
    table(mat.reset_index(), "sources_matrix")
    table(keep.reset_index().rename(columns={"index": "source"}), "sources_features")
    text(f"Common features used across sources: {common}")

    # ------------------------------------------------------------ monitoring + card
    section("9. Monitoring and model card")
    mon = monitoring.monitor(champ, r.train)
    table(mon["table"], "monitoring")
    monitoring.model_card(champ, best["contract"])
    text("Model card written to reports/model_card.md")

    (config.REPORTS / "results_summary.md").write_text("\n".join(MD), encoding="utf-8")
    json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - t0) / 60, 1)},
              open(config.ARTIFACTS / "run_info.json", "w", encoding="utf-8"))
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min -> reports/results_summary.md, reports/figures/, reports/tables/")


if __name__ == "__main__":
    main()
