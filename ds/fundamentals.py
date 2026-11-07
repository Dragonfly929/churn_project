"""Step 5 — the six fundamentals (Stage 2): Bias, Noise, Decision Boundaries, SMOTE
(+ Streaming and RTAP in ds/streaming_sim.py and ds/rtap.py).

Every function returns a dict: {"table": DataFrame, "figs": [...], "measures": [...]}
where "measures" are the sentences for the report: "S-a efectuat ..., deoarece ...".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
                             f1_score, recall_score)
from sklearn.model_selection import StratifiedKFold

import config
from ds import plots
from ds.catalog import ALL_FEATURES
from ds.modeling import (RunConfig, apply_thresholds, cost_curve, fairness_gaps, fairness_table,
                         make_pipeline, metrics, run, split, transform)

NONSENSITIVE = [f for f in ALL_FEATURES if f not in config.SENSITIVE]


def _row(name, r):
    m = r.metrics
    return {"variant": name, "roc_auc": m["roc_auc"], "recall": m["recall"], "precision": m["precision"],
            "expected_cost": m["expected_cost"], **{f"gap_{k}": v for k, v in r.gaps.items()}}


# =================================================================== 1. BIAS
def bias(df: pd.DataFrame, base: RunConfig) -> dict:
    rows, figs = [], []
    base = RunConfig(**{**base.to_dict(), "per_group_thresholds": False})  # variants 1-4 use one threshold for all
    r_full = run(base, df)
    rows.append(_row("1. all features (incl. age, gender, country)", r_full))
    rb = run(RunConfig(**{**base.to_dict(), "bias_x_group": "age_group=60+", "bias_x_share": 0.8}), df)
    rows.append(_row("2. BIASED training sample (80% of 60+ removed)", rb))
    r_drop = run(RunConfig(**{**base.to_dict(), "features": NONSENSITIVE}), df)
    rows.append(_row("3. sensitive attributes removed", r_drop))
    r_drop2 = run(RunConfig(**{**base.to_dict(), "features": [f for f in NONSENSITIVE if f != "digital_share_90d"]}), df)
    rows.append(_row("4. sensitive + proxy (digital_share) removed", r_drop2))
    r_pg = run(RunConfig(**{**base.to_dict(), "features": r_drop2.cfg.features, "per_group_thresholds": True,
                            "group_col": "age_group"}), df)
    rows.append(_row("5. as 4 + per-age-group thresholds", r_pg))
    # 6. fairlearn ThresholdOptimizer (true positive rate parity) on the trained pipeline
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        to = ThresholdOptimizer(estimator=r_drop2.pipe, constraints="true_positive_rate_parity",
                                objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
        to.fit(r_drop2.train[r_drop2.cfg.features], r_drop2.train["churn"], sensitive_features=r_drop2.train["age_group"])
        pred = to.predict(r_drop2.test[r_drop2.cfg.features], sensitive_features=r_drop2.test["age_group"],
                          random_state=config.SEED)
        m = metrics(r_drop2.test["churn"], r_drop2.p_test, 0.5, pred=pred)
        ft = fairness_table(r_drop2.test, r_drop2.test["churn"].values, pred, r_drop2.p_test)
        rows.append({"variant": "6. fairlearn ThresholdOptimizer (TPR parity, age)", "roc_auc": m["roc_auc"],
                     "recall": m["recall"], "precision": m["precision"], "expected_cost": m["expected_cost"],
                     **{f"gap_{k}": v for k, v in fairness_gaps(ft).items()}})
    except Exception as e:  # noqa: BLE001
        rows.append({"variant": f"6. ThresholdOptimizer failed: {e}"})
    table = pd.DataFrame(rows)

    f, axes = plots.fig(12, 3.6, 3, 1)
    for ax, (name, r) in zip(axes, [("all features", r_full), ("biased sample", rb), ("mitigated (4)", r_drop2)]):
        t = r.fairness[r.fairness.attribute == "age_group"]
        ax.bar(t["group"], t["recall"], color=plots.PALETTE[0], width=0.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"recall by age group – {name}")
    figs.append(plots.save(f, "f1_bias_recall_by_age"))
    measures = [
        "S-a eliminat vârsta, genul și țara din modelul de decizie, deoarece acestea sunt atribute sensibile "
        "și decizia de retenție nu trebuie să depindă de ele (GDPR, nediscriminare).",
        "S-a verificat proxy-ul digital_share_90d (corelat cu vârsta), deoarece eliminarea atributului sensibil "
        "nu ajută dacă o altă variabilă îl înlocuiește.",
        "S-a păstrat un singur prag pentru toți clienții (fără praguri pe grup de vârstă), deoarece după eliminarea "
        "proxy-ului diferența de recall era deja sub limita din matricea-contract, iar o decizie care depinde de "
        "vârstă ar contrazice regula de nediscriminare; recall-ul țintă se atinge prin alegerea pragului unic.",
    ]
    return {"table": table, "figs": figs, "measures": measures, "fairness_full": r_full.fairness,
            "fairness_mitigated": r_drop2.fairness}


# =================================================================== 2. NOISE
def noise_text(source="bank") -> pd.DataFrame:
    """Sentiment accuracy vs ground truth, raw vs denoised, per noise type."""
    from stage1 import db
    from ds.textproc import label_from_score
    fb = db.read_sql("SELECT msg_id, sentiment_raw, sentiment_score, is_empty FROM fact_feedback", source=source)
    truth = pd.read_csv((config.RAW if source == "bank" else config.RAW_BRANCH) / "sim_truth_messages.csv")
    m = fb.merge(truth, on="msg_id")
    m = m[m["is_empty"] == 0]
    m["raw_ok"] = m["sentiment_raw"].map(label_from_score) == m["true_sentiment"]
    m["den_ok"] = m["sentiment_score"].map(label_from_score) == m["true_sentiment"]
    t = m.groupby("noise_type").agg(n=("msg_id", "size"), accuracy_raw=("raw_ok", "mean"),
                                    accuracy_denoised=("den_ok", "mean"))
    t.loc["ALL"] = [len(m), m["raw_ok"].mean(), m["den_ok"].mean()]
    return t.sort_values("n", ascending=False)


def noise(df: pd.DataFrame, df_rawtext: pd.DataFrame, base: RunConfig, reps: int = 3) -> dict:
    text_t = noise_text()
    r_den = run(base, df)
    r_raw = run(base, df_rawtext)
    text_model = pd.DataFrame([_row("sentiment from DENOISED text", r_den), _row("sentiment from RAW text", r_raw)])
    rows = []
    tr, te = split(df, base.seed)
    for kind, levels in [("X: gaussian noise (sigma, std units)", [0, 0.25, 0.5, 1.0, 2.0]),
                         ("Y: labels flipped (share)", [0, 0.05, 0.10, 0.20, 0.30])]:
        for lv in levels:
            aucs, f1s = [], []
            for k in range(reps):
                c = RunConfig(**{**base.to_dict(), "seed": base.seed + k,
                                 "noise_x_sigma": lv if kind.startswith("X") else 0,
                                 "noise_y_flip": lv if kind.startswith("Y") else 0})
                r = run(c, train=tr, test=te, folds=3)
                aucs.append(r.metrics["roc_auc"]); f1s.append(r.metrics["f1"])
            rows.append({"noise": kind, "level": lv, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
                         "f1_mean": np.mean(f1s)})
    num_t = pd.DataFrame(rows)
    f, axes = plots.fig(11, 3.6, 2, 1)
    for ax, (k, g) in zip(axes, num_t.groupby("noise", sort=False)):
        ax.errorbar(g["level"], g["auc_mean"], yerr=g["auc_std"], marker="o", color=plots.PALETTE[0], label="ROC-AUC")
        ax.plot(g["level"], g["f1_mean"], marker="s", color=plots.PALETTE[1], label="F1")
        ax.set_title(k); ax.set_ylim(0.3, 1); ax.legend()
    figs = [plots.save(f, "f2_noise_x_vs_y")]
    measures = [
        "S-a efectuat denoising-ul textului (normalizare, diacritice, corectarea greșelilor, tratarea "
        "sarcasmului și a construcțiilor cu „dar”), deoarece feedback-ul brut inversa sau anula sentimentul.",
        "S-au tratat mesajele goale/corupte ca „fără conținut” (is_empty), deoarece un mesaj gol nu este un "
        "mesaj pozitiv; contactul în sine rămâne numărat.",
        "S-a folosit un model robust (boosting cu early stopping, regularizare L2) și netezirea alertelor cu "
        "EWMA în streaming, deoarece zgomotul numeric și zilnic produce alarme false.",
        "S-a verificat definiția etichetei (data ieșirii, fereastra de 90 de zile), deoarece zgomotul pe Y "
        "afectează modelul mult mai mult decât zgomotul pe X.",
    ]
    return {"table": num_t, "text_table": text_t, "text_model": text_model, "figs": figs, "measures": measures}


# =================================================================== 3. DECISION BOUNDARIES
def boundaries(df: pd.DataFrame, base: RunConfig, top2=("balance_change_3m", "nps_score")) -> dict:
    tr, te = split(df, base.seed)
    res = {m: run(RunConfig(**{**base.to_dict(), "model": m}), train=tr, test=te, folds=3)
           for m in ["logreg", "rf", "gb", "mlp"]}
    figs = []
    f, axes = plots.fig(11, 4, 2, 1)
    for i, (m, r) in enumerate(res.items()):
        fpr, tpr, _ = roc_curve(r.test["churn"], r.p_test)
        pr, rc, _ = precision_recall_curve(r.test["churn"], r.p_test)
        axes[0].plot(fpr, tpr, color=plots.PALETTE[i], label=f"{m} AUC={r.metrics['roc_auc']:.3f}")
        axes[1].plot(rc, pr, color=plots.PALETTE[i], label=f"{m} AP={r.metrics['pr_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], color=plots.GRAY, ls="--", lw=1)
    axes[0].set(title="ROC", xlabel="FPR", ylabel="TPR"); axes[0].legend(fontsize=8)
    axes[1].set(title="Precision-Recall", xlabel="recall", ylabel="precision"); axes[1].legend(fontsize=8)
    figs.append(plots.save(f, "f3_roc_pr"))

    r = res[base.model]
    y, p = r.test["churn"].values, r.p_test
    cc = cost_curve(r.train["churn"].values, r.p_oof, base.cost_fn, base.cost_fp)
    f, ax = plots.fig(7, 3.6)
    ax.plot(cc["threshold"], cc["cost"] / 1000, color=plots.PALETTE[0])
    ax.axvline(r.threshold, color=plots.PALETTE[1], ls="--", label=f"cost-optimal = {r.threshold:.2f}")
    ax.axvline(0.5, color=plots.GRAY, ls=":", label="0.5 default")
    ax.set(title="Expected cost vs threshold (out-of-fold, training)", xlabel="threshold", ylabel="cost (k EUR)")
    ax.legend()
    figs.append(plots.save(f, "f3_cost_vs_threshold"))

    cms = []
    for t in sorted({0.1, 0.2, round(r.threshold, 2), 0.5}):
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
        cms.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "recall": tp / (tp + fn), "precision": tp / max(1, tp + fp),
                    "cost": metrics(y, p, t, base.cost_fn, base.cost_fp)["expected_cost"]})
    deciles = pd.DataFrame({"p": p, "y": y})
    deciles["risk_decile"] = pd.qcut(deciles["p"].rank(method="first"), 10, labels=range(10, 0, -1))
    dec = deciles.groupby("risk_decile", observed=True).agg(clients=("y", "size"), leavers=("y", "sum"),
                                                            churn_rate=("y", "mean"), min_score=("p", "min"))
    dec = dec.sort_index(ascending=False)

    # 2D boundaries on the two strongest features
    feats2 = [f for f in top2 if f in df.columns]
    d2 = tr[feats2 + ["churn"]].dropna()
    lo = d2[feats2].quantile(0.01).values; hi = d2[feats2].quantile(0.99).values
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 150), np.linspace(lo[1], hi[1], 150))
    grid = pd.DataFrame({feats2[0]: xx.ravel(), feats2[1]: yy.ravel()})
    f, axes = plots.fig(14, 3.6, 4, 1)
    samp = d2.sample(min(900, len(d2)), random_state=1)
    for ax, m in zip(axes, ["logreg", "rf", "gb", "mlp"]):
        c = RunConfig(**{**base.to_dict(), "model": m, "features": feats2})
        pipe = make_pipeline(c).fit(d2[feats2], d2["churn"])
        zz = pipe.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=20, cmap=plots.DIV, alpha=0.55)
        ax.contour(xx, yy, zz, levels=[r.threshold], colors="k", linewidths=1.2)
        ax.scatter(samp[feats2[0]].clip(lo[0], hi[0]), samp[feats2[1]].clip(lo[1], hi[1]), s=6,
                   c=np.where(samp["churn"] == 1, plots.LEAVES, plots.STAYS), alpha=0.85,
                   edgecolors="white", linewidths=0.4)
        ax.set(title=m, xlabel=feats2[0], ylabel=feats2[1]); ax.grid(False)
    from matplotlib.lines import Line2D
    f.legend(handles=[Line2D([], [], marker="o", ls="", color=plots.STAYS, label="stays"),
                      Line2D([], [], marker="o", ls="", color=plots.LEAVES, label="leaves"),
                      Line2D([], [], color="k", label="boundary at the cost threshold")],
             loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    figs.append(plots.save(f, "f3_boundaries_2d"))

    # complexity curve (under/overfitting): GB depth and RF depth
    rows = []
    skf = StratifiedKFold(3, shuffle=True, random_state=base.seed)
    for depth in [1, 2, 3, 4, 6, 8, 12, 16]:
        for m in ["rf"]:
            c = RunConfig(**{**base.to_dict(), "model": m, "params": {"max_depth": depth, "min_samples_leaf": 1,
                                                                     "n_estimators": 150}})
            tr_s, va_s = [], []
            for a, b in skf.split(tr, tr["churn"]):
                pipe = make_pipeline(c).fit(tr.iloc[a][c.features], tr.iloc[a]["churn"])
                tr_s.append(roc_auc_score(tr.iloc[a]["churn"], pipe.predict_proba(tr.iloc[a][c.features])[:, 1]))
                va_s.append(roc_auc_score(tr.iloc[b]["churn"], pipe.predict_proba(tr.iloc[b][c.features])[:, 1]))
            rows.append({"model": m, "max_depth": depth, "train_auc": np.mean(tr_s), "valid_auc": np.mean(va_s)})
    comp = pd.DataFrame(rows)
    f, ax = plots.fig(7, 3.6)
    ax.plot(comp["max_depth"], comp["train_auc"], marker="o", color=plots.PALETTE[0], label="training")
    ax.plot(comp["max_depth"], comp["valid_auc"], marker="o", color=plots.PALETTE[1], label="validation")
    ax.set(title="Complexity curve (Random Forest depth): underfitting -> overfitting", xlabel="max_depth",
           ylabel="ROC-AUC"); ax.legend()
    figs.append(plots.save(f, "f3_complexity_curve"))
    comp_table = pd.DataFrame([{"model": m, **{k: v for k, v in rr.metrics.items()
                                               if k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall",
                                                        "f1", "lift_top10", "expected_cost", "threshold")},
                                "cv_auc": rr.cv_auc, "fit_s": rr.fit_seconds, "latency_ms": rr.latency_ms}
                               for m, rr in res.items()])
    measures = [
        "S-a ales pragul de decizie din matricea de cost (minimul costului așteptat pe predicții out-of-fold), "
        "deoarece un client pierdut costă mult mai mult decât o ofertă inutilă; pragul 0.5 ignoră această asimetrie.",
        "S-a segmentat scorul în decile de risc, deoarece echipa de retenție are capacitate limitată și "
        "contactează întâi decilele de sus.",
        "S-a ales adâncimea modelului din curba de complexitate (validare), deoarece peste acest punct modelul "
        "învață zgomotul (overfitting), iar sub el ignoră tipare reale (underfitting).",
    ]
    return {"table": comp_table, "confusion": pd.DataFrame(cms), "deciles": dec, "complexity": comp,
            "figs": figs, "measures": measures, "results": res}


# =================================================================== 4. SMOTE
def smote(df: pd.DataFrame, base: RunConfig) -> dict:
    from imblearn.over_sampling import SMOTE
    tr, te = split(df, base.seed)
    rows = []
    for bal in ["none", "class_weight", "ros", "smote", "smotenc"]:
        r = run(RunConfig(**{**base.to_dict(), "balancing": bal}), train=tr, test=te, folds=4)
        rows.append({"variant": bal, "cv_auc": r.cv_auc, **{k: r.metrics[k] for k in
                     ("roc_auc", "pr_auc", "recall", "precision", "f1", "expected_cost", "threshold")}})
    # WRONG: SMOTE before the split -> synthetic copies of test clients leak into training
    c = RunConfig(**{**base.to_dict(), "balancing": "none"})
    enc_pipe = make_pipeline(c)
    Z = enc_pipe.named_steps["encode"].fit_transform(df[c.features])
    Zs, ys = SMOTE(random_state=base.seed).fit_resample(Z, df["churn"])
    skf = StratifiedKFold(4, shuffle=True, random_state=base.seed)
    aucs = []
    for a, b in skf.split(Zs, ys):
        p2 = make_pipeline(c)
        p2.steps = p2.steps[1:]  # already encoded
        p2.fit(Zs[a], ys[a])
        aucs.append(roc_auc_score(ys[b], p2.predict_proba(Zs[b])[:, 1]))
    rows.append({"variant": "SMOTE BEFORE the split (WRONG)", "cv_auc": float(np.mean(aucs)),
                 "roc_auc": np.nan, "pr_auc": np.nan, "recall": np.nan, "precision": np.nan, "f1": np.nan,
                 "expected_cost": np.nan, "threshold": np.nan})
    table = pd.DataFrame(rows)

    # realism of synthetic clients
    Ztr = enc_pipe.named_steps["encode"].fit_transform(tr[c.features])
    Zr, yr = SMOTE(random_state=base.seed).fit_resample(Ztr, tr["churn"])
    syn = Zr[len(Ztr):]
    names = [f for f in c.features if f not in config.CATEGORICAL] + [f for f in c.features if f in config.CATEGORICAL]
    discrete = ["num_products", "has_card", "is_active_member", "no_feedback_6m", "nps_missing",
                "silent_decline", "complaints_180d", "tx_count_90d"]
    real_min = Ztr[tr["churn"].values == 1]
    sc = Ztr.std(axis=0) + 1e-9
    def nn(a, b):
        d = np.sqrt((((a[:, None, :] - b[None, :, :]) / sc) ** 2).sum(-1))
        return d.min(1)
    nn_syn = nn(syn[:400], real_min)
    nn_real = nn(real_min[:200], real_min[200:]) if len(real_min) > 400 else nn_syn
    real_t = pd.DataFrame([{"check": f"{f}: share of synthetic values that are not integers",
                            "value": float(np.mean(np.abs(syn[:, names.index(f)] - np.round(syn[:, names.index(f)])) > 1e-6))}
                           for f in discrete if f in names] +
                          [{"check": "median distance synthetic -> nearest real leaver (std units)", "value": float(np.median(nn_syn))},
                           {"check": "median distance real leaver -> nearest other real leaver", "value": float(np.median(nn_real))}])
    # SMOTE on a feature subset + SMOTE with noise
    sub = ["balance_change_3m", "nps_score", "nps_missing", "tx_trend", "days_since_last_msg", "complaints_180d",
           "avg_sentiment_180d", "amount_in_90d", "num_products", "is_active_member"]
    extra = []
    for name, cc in [("SMOTE, all features", RunConfig(**{**base.to_dict(), "balancing": "smote"})),
                     ("SMOTE, 10 selected features", RunConfig(**{**base.to_dict(), "balancing": "smote", "features": sub})),
                     ("SMOTE + label noise 10%", RunConfig(**{**base.to_dict(), "balancing": "smote", "noise_y_flip": 0.1})),
                     ("no balancing + label noise 10%", RunConfig(**{**base.to_dict(), "balancing": "none", "noise_y_flip": 0.1}))]:
        r = run(cc, train=tr, test=te, folds=3)
        extra.append({"variant": name, "roc_auc": r.metrics["roc_auc"], "recall": r.metrics["recall"],
                      "precision": r.metrics["precision"], "f1": r.metrics["f1"]})
    f, ax = plots.fig(8, 3.6)
    t = table.dropna(subset=["roc_auc"])
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["recall"], 0.38, color=plots.PALETTE[0], label="recall")
    ax.bar(x + 0.2, t["precision"], 0.38, color=plots.PALETTE[1], label="precision")
    ax.set_xticks(x, t["variant"]); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Balancing variants (test set)")
    figs = [plots.save(f, "f4_smote_variants")]
    ok = table.dropna(subset=["roc_auc"]).set_index("variant")
    best = ok["expected_cost"].idxmin()
    measures = [
        f"S-au comparat variantele de echilibrare, deoarece clienții care pleacă sunt minoritari "
        f"(~{df.churn.mean():.0%}); „{best}” are costul așteptat minim ({ok.loc[best, 'expected_cost']:,.0f} €), "
        f"iar fără echilibrare costul este {ok.loc['none', 'expected_cost']:,.0f} €. Diferența este mică, deoarece "
        "pragul ales din matricea de cost compensează deja dezechilibrul; modelul final rămâne fără echilibrare "
        "(mai simplu, probabilități nedistorsionate), cu SMOTE-NC ca alternativă documentată.",
        "S-a aplicat SMOTE numai în interiorul fold-urilor de antrenare (imblearn Pipeline), deoarece SMOTE înainte "
        "de împărțire copiază clienți din test în antrenare și umflă artificial scorul.",
        "S-a folosit SMOTE-NC pentru variabilele categoriale, deoarece SMOTE simplu interpolează coduri de țară/gen "
        "fără sens; s-a verificat realismul clienților sintetici (valori ne-întregi, distanța față de clienții reali).",
    ]
    return {"table": table, "realism": real_t, "subset_noise": pd.DataFrame(extra), "figs": figs,
            "measures": measures, "best": best}
