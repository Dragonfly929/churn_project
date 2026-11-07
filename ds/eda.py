"""Step 2 — EDA following the course material: unique identifier, then 5 steps
(missing values, description/types, shape, correlations, anomalies), KPIs,
hypotheses and the leakage trap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr

from ds import plots
from ds.catalog import ALL_FEATURES, FEATURES_DF
from ds.modeling import RunConfig, run


def step0_unique_id(df):
    return {"identifier": "client_id (one row per client at the cutoff)", "rows": len(df),
            "unique": int(df["client_id"].nunique()), "is_unique": bool(df["client_id"].is_unique)}


def step1_missing(df):
    m = df[ALL_FEATURES].isna().sum()
    out = pd.DataFrame({"missing": m, "share": m / len(df)}).sort_values("missing", ascending=False)
    why = {"nps_score": "NOT at random: silent clients do not answer the survey -> kept as nps_missing flag",
           "credit_score": "completely at random (CRM glitch) -> median imputation inside the pipeline",
           "age": "impossible birth dates set to NULL in cleaning -> median imputation"}
    out["why / decision"] = [why.get(f, "-") for f in out.index]
    return out[out["missing"] > 0]


def step2_types(df):
    t = FEATURES_DF[["feature", "group", "type"]].copy()
    d = df[ALL_FEATURES].describe(include="all").T
    t["mean"] = [d.loc[f, "mean"] if "mean" in d.columns and f in d.index else np.nan for f in t.feature]
    t["n_unique"] = [df[f].nunique() for f in t.feature]
    return t


def step3_shape(df):
    seg = {}
    for col in ["age_group", "country", "gender", "num_products", "silent"]:
        seg[col] = df.groupby(col)["churn"].agg(["mean", "count"]).rename(columns={"mean": "churn_rate"})
    low_var = [f for f in ALL_FEATURES if df[f].nunique(dropna=True) <= 1 or
               (df[f].value_counts(normalize=True).iloc[0] > 0.98)]
    return seg, low_var


def fig_distributions(df, feats=("tx_trend", "balance_change_3m", "days_since_last_msg", "amount_in_90d",
                                 "complaints_180d", "age")):
    f, axes = plots.fig(12, 6, 3, 2)
    for ax, c in zip(axes.ravel(), feats):
        lo, hi = df[c].quantile([0.01, 0.99])
        for y, col, lab in [(0, plots.STAYS, "stays"), (1, plots.LEAVES, "leaves")]:
            ax.hist(df.loc[df.churn == y, c].clip(lo, hi), bins=30, alpha=0.55, color=col, label=lab, density=True)
        ax.set_title(c)
    axes[0, 0].legend()
    return plots.save(f, "eda_distributions")


def fig_segments(df):
    f, axes = plots.fig(12, 3.4, 4, 1)
    for ax, c in zip(axes, ["age_group", "country", "num_products", "silent"]):
        s = df.groupby(c)["churn"].mean()
        ax.bar(s.index.astype(str), s.values, color=plots.PALETTE[0], width=0.6)
        ax.axhline(df.churn.mean(), color=plots.GRAY, ls="--", lw=1)
        ax.set_title(f"churn rate by {c}")
    return plots.save(f, "eda_segments")


def step4_correlations(df):
    num = [f for f in ALL_FEATURES if f not in ("gender", "country")]
    tgt = pd.Series({f: spearmanr(df[f], df["churn"], nan_policy="omit").statistic for f in num}, name="rho_churn")
    corr = df[num].corr(method="spearman")
    proxies = []
    for f in num:
        if f == "age":
            continue
        r = corr.loc[f, "age"]
        if abs(r) >= 0.3:
            proxies.append({"feature": f, "spearman_with_age": round(float(r), 3),
                            "note": "possible PROXY for the sensitive attribute age"})
    return tgt.sort_values(key=np.abs, ascending=False), corr, pd.DataFrame(proxies)


def fig_corr(corr):
    f, ax = plots.fig(9, 7.5)
    im = ax.imshow(corr.values, cmap=plots.DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    ax.grid(False)
    f.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Spearman correlations between features")
    return plots.save(f, "eda_correlations")


def step5_outliers(df):
    rows = []
    for f in ALL_FEATURES:
        if f in ("gender", "country") or df[f].nunique() <= 2:
            continue
        x = df[f].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_iqr = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
        n_z = int((np.abs((x - x.mean()) / (x.std() or 1)) > 4).sum())
        churn_out = df.loc[x.index[(x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)], "churn"].mean() if n_iqr else np.nan
        rows.append({"feature": f, "outliers_3IQR": n_iqr, "outliers_z4": n_z, "churn_rate_among_outliers": churn_out})
    out = pd.DataFrame(rows).sort_values("outliers_3IQR", ascending=False)
    out["decision"] = np.where(out["churn_rate_among_outliers"] > df.churn.mean() * 1.5,
                               "KEEP: outliers concentrate leavers = signal, not error",
                               "keep; robust models + scaling")
    return out


def kpis_and_hypotheses(df):
    leavers = df[df.churn == 1]
    ct = pd.crosstab(df["silent"], df["churn"])
    chi = chi2_contingency(ct)
    rate_s, rate_ns = df.loc[df.silent == 1, "churn"].mean(), df.loc[df.silent == 0, "churn"].mean()
    mw = mannwhitneyu(df.loc[df.churn == 1, "tx_trend"], df.loc[df.churn == 0, "tx_trend"])
    return {
        "B1 churn rate (90d)": df.churn.mean(),
        "B2 share of silent leavers": leavers["silent"].mean(),
        "share of silent clients overall": df["silent"].mean(),
        "H1 churn rate silent vs not": f"{rate_s:.3f} vs {rate_ns:.3f} (chi2 p={chi.pvalue:.2e}) -> "
                                       + ("H1 SUPPORTED" if rate_s > rate_ns and chi.pvalue < 0.05 else "H1 not supported"),
        "tx_trend leavers vs stayers (Mann-Whitney p)": f"{mw.pvalue:.2e}",
    }


def leakage_demo(df_ok, df_leaky, model="gb"):
    """Same model, same rows: features up to the cutoff vs features computed after the departures."""
    from ds.xai import shap_frame, global_importance
    out = {}
    for name, d in [("correct cutoff", df_ok), ("WRONG cutoff (leakage)", df_leaky)]:
        r = run(RunConfig(model=model, balancing="none"), d, folds=3)
        s, _ = shap_frame(r.pipe, r.test, r.cfg.features, max_rows=400)
        imp = global_importance(s)
        out[name] = {"roc_auc": r.metrics["roc_auc"], "recall": r.metrics["recall"],
                     "top_feature": imp.index[0], "top_feature_share": float(imp.iloc[0] / imp.sum()),
                     "top3": ", ".join(imp.index[:3])}
    return pd.DataFrame(out).T
