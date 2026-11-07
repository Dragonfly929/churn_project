"""Step 6 — the lab experiments applied to churn.

6.1 Experiments 1-2: noise and bias on X vs Y (repeated, mean +- std)
6.2 Representations: text embeddings + PCA, MLP hidden layer + PCA
6.3 Regularization L1 / L2 / ElasticNet: coefficients, C and alpha sweeps, F1, latency,
    coefficient heatmap, SHAP for the three, coefficients over time
6.4 Experiment A (toy data with known truth), Experiment B (learned synthetic copy),
    data sources compared (train on each x test on each)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from ds import plots
from ds.catalog import ALL_FEATURES
from ds.modeling import RunConfig, best_threshold, make_logreg, make_pipeline, run, split, transform, feature_names_out

NONSENS = [f for f in ALL_FEATURES if f not in config.SENSITIVE]


# =================================================================== 6.1 Experiments 1-2
def _prep(df, feats):
    X = df[feats].copy()
    X = X.fillna(X.median())
    return X.values.astype(float), df["churn"].values


def experiment_1_2(df: pd.DataFrame, reps: int = 100, levels=(0, 0.1, 0.2, 0.3, 0.4), seed=config.SEED):
    """Exp 1: random noise on X vs on Y.  Exp 2: systematic bias on X vs on Y.
    Logistic regression (fast) so 100 repetitions run in about a minute."""
    feats = [f for f in NONSENS if f != "nps_score"]
    X, y = _prep(df, feats)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    silent_idx = feats.index("no_feedback_6m")
    rows = []
    rng = np.random.default_rng(seed)
    for scen in ["noise_X", "noise_Y", "bias_X", "bias_Y"]:
        for lv in levels:
            aucs, f1s = [], []
            for _ in range(reps):
                Xa, ya = Xtr.copy(), ytr.copy()
                if scen == "noise_X":       # random Gaussian noise, sigma = lv * 5 std units
                    Xa = Xa + rng.normal(0, lv * 5, Xa.shape)
                elif scen == "noise_Y":     # random label flips
                    k = rng.random(len(ya)) < lv
                    ya[k] = 1 - ya[k]
                elif scen == "bias_X":      # systematic shift of the strongest features
                    Xa[:, :] = Xa + lv * 5 * np.sign(rng.normal(size=(1, Xa.shape[1])))
                elif scen == "bias_Y":      # systematic: silent leavers recorded as "stayed"
                    k = (ya == 1) & (Xtr[:, silent_idx] > 0) & (rng.random(len(ya)) < lv * 2.5)
                    ya[k] = 0
                if ya.min() == ya.max():
                    continue
                m = make_logreg(1.0, 0.0).fit(Xa, ya)
                p = m.predict_proba(Xte)[:, 1]
                aucs.append(roc_auc_score(yte, p))
                f1s.append(f1_score(yte, (p >= 0.25).astype(int)))
            rows.append({"scenario": scen, "level": lv, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
                         "f1_mean": np.mean(f1s), "f1_std": np.std(f1s), "reps": len(aucs)})
    t = pd.DataFrame(rows)
    f, axes = plots.fig(11, 3.6, 2, 1)
    for i, s in enumerate(["noise_X", "noise_Y", "bias_X", "bias_Y"]):
        g = t[t.scenario == s]
        ax = axes[0] if s.startswith("noise") else axes[1]
        ax.errorbar(g["level"], g["f1_mean"], yerr=g["f1_std"], marker="o", capsize=3,
                    color=plots.PALETTE[i], label=s)
    axes[0].set(title="Experiment 1: noise on X vs on Y (F1, mean ± std)", xlabel="level", ylabel="F1")
    axes[1].set(title="Experiment 2: bias on X vs on Y (F1, mean ± std)", xlabel="level")
    for a in axes:
        a.legend()
    plots.save(f, "e12_noise_bias_xy")
    # how the boundary moves (2 features, one draw per level)
    f2, axes2 = plots.fig(11, 3.6, 2, 1)
    i1, i2 = feats.index("balance_change_3m"), feats.index("days_since_last_msg")
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    for ax, scen in zip(axes2, ["noise_Y", "bias_Y"]):
        for j, lv in enumerate([0, 0.2, 0.4]):
            ya = ytr.copy()
            r2 = np.random.default_rng(j)
            if scen == "noise_Y":
                k = r2.random(len(ya)) < lv
                ya[k] = 1 - ya[k]
            else:
                k = (ya == 1) & (Xtr[:, silent_idx] > 0) & (r2.random(len(ya)) < lv * 2.5)
                ya[k] = 0
            m = make_logreg(1.0, 0.0).fit(Xtr[:, [i1, i2]], ya)
            zz = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
            cs = ax.contour(xx, yy, zz, levels=[0.25], colors=[plots.PALETTE[j]])
            ax.plot([], [], color=plots.PALETTE[j], label=f"level {lv}")
        ax.set(title=f"Boundary (p=0.25) moves under {scen}", xlabel="balance_change_3m (z)",
               ylabel="days_since_last_msg (z)")
        ax.legend()
    plots.save(f2, "e12_boundary_moves")
    return t


# =================================================================== 6.2 representations + PCA
def pca_text(source="bank"):
    from stage1 import db
    fb = db.read_sql("SELECT msg_id, text_clean FROM fact_feedback WHERE is_empty = 0", source=source)
    truth = pd.read_csv(config.RAW / "sim_truth_messages.csv")
    m = fb.merge(truth, on="msg_id").sample(min(3000, len(fb)), random_state=config.SEED)
    method = "TF-IDF + PCA (TruncatedSVD)"
    try:  # optional: multilingual sentence embeddings if installed (pip install sentence-transformers)
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").encode(m["text_clean"].tolist())
        pca = PCA(2, random_state=config.SEED)
        Z = pca.fit_transform(emb)
        var = pca.explained_variance_ratio_.sum()
        method = "sentence embeddings (MiniLM) + PCA"
    except Exception:  # noqa: BLE001
        T = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform(m["text_clean"])
        svd = TruncatedSVD(2, random_state=config.SEED)
        Z = svd.fit_transform(T)
        var = svd.explained_variance_ratio_.sum()
    f, ax = plots.fig(6.5, 4.5)
    for i, s in enumerate(["neg", "neu", "pos"]):
        k = m["true_sentiment"].values == s
        ax.scatter(Z[k, 0], Z[k, 1], s=6, alpha=0.5, color=plots.PALETTE[i], label=s)
    ax.set(title=f"Messages in 2D: {method}\nvariance kept = {var:.1%}", xlabel="PC1", ylabel="PC2")
    ax.legend()
    plots.save(f, "e62_text_pca")
    return {"method": method, "variance_kept": float(var)}


def pca_mlp_hidden(df: pd.DataFrame):
    c = RunConfig(model="mlp", balancing="none", features=NONSENS)
    tr, te = split(df, c.seed)
    pipe = make_pipeline(c).fit(tr[c.features], tr["churn"])
    Z = transform(pipe, te[c.features])
    mlp = pipe.named_steps["clf"]
    h = np.maximum(0, Z @ mlp.coefs_[0] + mlp.intercepts_[0])  # first hidden layer (ReLU)
    raw = PCA(2, random_state=1).fit(Z)
    hid = PCA(2, random_state=1).fit(h)
    from sklearn.metrics import silhouette_score
    out = {"raw_variance": float(raw.explained_variance_ratio_.sum()),
           "hidden_variance": float(hid.explained_variance_ratio_.sum()),
           "raw_silhouette": float(silhouette_score(raw.transform(Z), te["churn"])),
           "hidden_silhouette": float(silhouette_score(hid.transform(h), te["churn"]))}
    f, axes = plots.fig(11, 4, 2, 1)
    for ax, P, name in [(axes[0], raw.transform(Z), "raw (scaled) features"), (axes[1], hid.transform(h), "MLP hidden layer")]:
        for y, col, lab in [(0, plots.STAYS, "stays"), (1, plots.LEAVES, "leaves")]:
            k = te["churn"].values == y
            ax.scatter(P[k, 0], P[k, 1], s=6, alpha=0.5, color=col, label=lab)
        ax.set(title=f"PCA of {name}", xlabel="PC1", ylabel="PC2"); ax.legend()
    plots.save(f, "e62_mlp_hidden_pca")
    return out


# =================================================================== 6.3 Regularization
def regularization(df: pd.DataFrame, batches: dict[str, pd.DataFrame] | None = None):
    feats = NONSENS
    tr, te = split(df)
    base = RunConfig(model="logreg", balancing="class_weight", features=feats)
    out = {}
    coefs = {}
    shap_tops = {}
    for name, pen, l1r in [("L1 (Lasso)", "l1", 1.0), ("L2 (Ridge)", "l2", 0.0), ("ElasticNet a=0.5", "elasticnet", 0.5)]:
        c = RunConfig(**{**base.to_dict(), "params": {"C": 0.05, "penalty": pen, "l1_ratio": l1r}})
        pipe = make_pipeline(c).fit(tr[feats], tr["churn"])
        coefs[name] = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_names_out(pipe, feats))
        from ds.xai import shap_frame, top_k
        s, _ = shap_frame(pipe, te, feats, background=tr, max_rows=300)
        shap_tops[name] = top_k(s, 5)
    cdf = pd.DataFrame(coefs)
    out["coefficients"] = cdf
    out["zeros_L1"] = cdf.index[cdf["L1 (Lasso)"].abs() < 1e-8].tolist()
    out["shap_top5"] = pd.DataFrame(shap_tops)
    f, axes = plots.fig(13, 6, 3, 1)
    for ax, col in zip(axes, cdf.columns):
        v = cdf[col]
        colors = [plots.GRAY if abs(x) < 1e-8 else (plots.LEAVES if x > 0 else plots.STAYS) for x in v]
        ax.barh(v.index, v.values, color=colors)
        ax.set_title(f"{col} (C=0.05); grey = set to 0")
        ax.tick_params(axis="y", labelsize=7)
    plots.save(f, "e63_coefficients")
    # sweeps: C x alpha -> F1 and latency
    rows = []
    Ztr = make_pipeline(base).named_steps  # noqa: F841
    for l1r in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for C in np.logspace(-3, 1, 9):
            c = RunConfig(**{**base.to_dict(), "params": {"C": float(C), "penalty": "elasticnet", "l1_ratio": l1r}})
            t0 = time.time()
            pipe = make_pipeline(c).fit(tr[feats], tr["churn"])
            fit_ms = (time.time() - t0) * 1000
            t1 = time.time()
            p = pipe.predict_proba(te[feats])[:, 1]
            lat = (time.time() - t1) / len(te) * 1e6  # microseconds per client (batch)
            thr = best_threshold(te["churn"].values, p)
            coef = pipe.named_steps["clf"].coef_[0]
            rows.append({"l1_ratio": l1r, "C": C, "f1": f1_score(te["churn"], (p >= thr).astype(int)),
                         "auc": roc_auc_score(te["churn"], p), "fit_ms": fit_ms, "latency_us_per_client": lat,
                         "nonzero_coefs": int((np.abs(coef) > 1e-8).sum())})
    sw = pd.DataFrame(rows)
    out["sweep"] = sw
    f, axes = plots.fig(13, 3.6, 3, 1)
    for i, (l1r, g) in enumerate(sw.groupby("l1_ratio")):
        axes[0].plot(g["C"], g["f1"], marker="o", color=plots.PALETTE[i], label=f"α={l1r}")
        axes[1].plot(g["C"], g["fit_ms"], marker="o", color=plots.PALETTE[i], label=f"α={l1r}")
        axes[2].plot(g["C"], g["nonzero_coefs"], marker="o", color=plots.PALETTE[i], label=f"α={l1r}")
    for a, t in zip(axes, ["F1 vs C", "training latency (ms) vs C", "non-zero coefficients vs C"]):
        a.set_xscale("log"); a.set_title(t); a.set_xlabel("C (log)"); a.legend(fontsize=7)
    plots.save(f, "e63_sweeps")
    # heatmap of L1 coefficients against C
    hm = {}
    for C in np.logspace(-3, 1, 9):
        c = RunConfig(**{**base.to_dict(), "params": {"C": float(C), "penalty": "l1"}})
        pipe = make_pipeline(c).fit(tr[feats], tr["churn"])
        hm[f"{C:.3g}"] = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_names_out(pipe, feats))
    hm = pd.DataFrame(hm)
    f, ax = plots.fig(8, 7)
    lim = np.abs(hm.values).max()
    im = ax.imshow(hm.values, cmap=plots.DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=45); ax.set_yticks(range(len(hm)), hm.index, fontsize=7)
    ax.set_xlabel("C"); ax.set_title("L1 coefficients vs C (white = 0)"); ax.grid(False); f.colorbar(im, ax=ax, shrink=0.6)
    plots.save(f, "e63_heatmap_l1")
    out["heatmap"] = hm
    # coefficients over time: refit on successive monthly batches
    if batches:
        rows = []
        for cut, d in batches.items():
            for name, pen, l1r in [("L1", "l1", 1.0), ("L2", "l2", 0.0)]:
                c = RunConfig(**{**base.to_dict(), "params": {"C": 0.05, "penalty": pen, "l1_ratio": l1r}})
                pipe = make_pipeline(c).fit(d[feats], d["churn"])
                for fn, v in zip(feature_names_out(pipe, feats), pipe.named_steps["clf"].coef_[0]):
                    rows.append({"batch": cut, "model": name, "feature": fn, "coef": v})
        ot = pd.DataFrame(rows)
        out["over_time"] = ot
        top = ot[ot.model == "L2"].groupby("feature")["coef"].apply(lambda s: s.abs().mean()).nlargest(6).index
        f, axes = plots.fig(12, 3.8, 2, 1)
        for ax, mname in zip(axes, ["L1", "L2"]):
            g = ot[(ot.model == mname) & ot.feature.isin(top)]
            for i, (fn, gg) in enumerate(g.groupby("feature")):
                ax.plot(gg["batch"], gg["coef"], marker="o", color=plots.PALETTE[i % 8], label=fn)
            ax.set_title(f"{mname} coefficients over successive batches"); ax.tick_params(axis="x", rotation=45)
        axes[1].legend(fontsize=7)
        plots.save(f, "e63_coefs_over_time")
    return out


# =================================================================== 6.4 Experiment A
def experiment_a():
    X, y = make_classification(n_samples=3000, n_features=12, n_informative=6, n_redundant=2, n_repeated=0,
                               n_clusters_per_class=2, shuffle=False, random_state=config.SEED)
    names = [f"inf_{i}" for i in range(6)] + ["red_0", "red_1"] + [f"noise_{i}" for i in range(4)]
    X = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=config.SEED, stratify=y)
    rows, coefs = [], {}
    for C in np.logspace(-3, 1, 9):
        for name, l1r in [("L1", 1.0), ("ElasticNet", 0.5), ("L2", 0.0)]:
            t0 = time.time()
            m = make_logreg(float(C), l1r).fit(Xtr, ytr)
            ms = (time.time() - t0) * 1000
            p = m.predict_proba(Xte)[:, 1]
            co = m.coef_[0]
            if name == "L1":
                coefs[f"{C:.3g}"] = co
            rows.append({"model": name, "C": C, "f1": f1_score(yte, (p >= 0.5).astype(int)), "fit_ms": ms,
                         "zeros": ", ".join(n for n, v in zip(names, co) if abs(v) < 1e-8),
                         "noise_features_zeroed": sum(abs(v) < 1e-8 for n, v in zip(names, co) if n.startswith("noise")),
                         "informative_zeroed": sum(abs(v) < 1e-8 for n, v in zip(names, co) if n.startswith("inf"))})
    t = pd.DataFrame(rows)
    hm = pd.DataFrame(coefs, index=names)
    f, ax = plots.fig(7, 5)
    lim = np.abs(hm.values).max()
    ax.imshow(hm.values, cmap=plots.DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=45); ax.set_yticks(range(len(hm)), hm.index)
    ax.set(title="Experiment A: L1 coefficients vs C (known truth)", xlabel="C"); ax.grid(False)
    plots.save(f, "e64_expA_toy_l1")
    return t


# =================================================================== 6.4 Experiment B + sources
def gaussian_copula_synth(df: pd.DataFrame, n: int, seed=config.SEED) -> pd.DataFrame:
    """Learned synthetic copy. Uses SDV GaussianCopula if installed, else a small own implementation
    (empirical marginals + Gaussian copula of the normal scores)."""
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import GaussianCopulaSynthesizer
        md = Metadata.detect_from_dataframe(df)
        s = GaussianCopulaSynthesizer(md)
        s.fit(df)
        return s.sample(n), "SDV GaussianCopulaSynthesizer"
    except Exception:  # noqa: BLE001
        from scipy.stats import norm
        rng = np.random.default_rng(seed)
        cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        enc = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in cat}
        d = df.copy()
        for c in cat:
            d[c] = d[c].astype(str).map(enc[c])
        d = d.fillna(d.median())
        U = d.rank(method="average") / (len(d) + 1)
        Zs = norm.ppf(U.values)
        cov = np.corrcoef(Zs, rowvar=False)
        g = rng.multivariate_normal(np.zeros(d.shape[1]), cov, n)
        u = norm.cdf(g)
        out = pd.DataFrame({c: np.quantile(d[c].values, u[:, i], method="inverted_cdf") for i, c in enumerate(d.columns)})
        for c in cat:
            inv = {i: v for v, i in enc[c].items()}
            out[c] = out[c].round().astype(int).map(inv)
        return out, "own Gaussian copula (pip install sdv for the SDV version)"


def experiment_b(real: pd.DataFrame, label: str):
    from scipy.stats import ks_2samp
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = [c for c in real.columns if c != "client_id"]
    tr, te = train_test_split(real[cols], test_size=0.3, stratify=real["churn"], random_state=config.SEED)
    syn, method = gaussian_copula_synth(tr, len(tr))
    feats = [c for c in cols if c != "churn"]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(real[c])]
    fid = pd.DataFrame([{"feature": c, "ks_stat": ks_2samp(tr[c].dropna(), syn[c].dropna()).statistic} for c in num])
    corr_diff = float(np.abs(tr[num].corr() - syn[num].corr()).mean().mean())
    def fit_eval(a, b):
        Xa = pd.get_dummies(a[feats], dtype=float); Xb = pd.get_dummies(b[feats], dtype=float).reindex(columns=Xa.columns, fill_value=0)
        m = HistGradientBoostingClassifier(random_state=config.SEED).fit(Xa, a["churn"].astype(int))
        return roc_auc_score(b["churn"].astype(int), m.predict_proba(Xb)[:, 1])
    trtr, tstr = fit_eval(tr, te), fit_eval(syn, te)
    sc = StandardScaler().fit(tr[num].fillna(tr[num].median()))
    A = sc.transform(syn[num].fillna(tr[num].median()))[:1000]
    B = sc.transform(tr[num].fillna(tr[num].median()))
    dcr = np.array([np.sqrt(((B - a) ** 2).sum(1)).min() for a in A])
    Bi = B[:1000]
    rr = np.array([np.sort(np.sqrt(((B - a) ** 2).sum(1)))[1] for a in Bi])
    return {"source": label, "method": method, "mean_KS": float(fid["ks_stat"].mean()), "mean_abs_corr_diff": corr_diff,
            "AUC train-real/test-real (TRTR)": trtr, "AUC train-synthetic/test-real (TSTR)": tstr,
            "median DCR synthetic->real": float(np.median(dcr)), "median real->real NN distance": float(np.median(rr)),
            "share synthetic closer than 5th pct real NN": float((dcr < np.quantile(rr, 0.05)).mean())}, syn


def sources_matrix(sources: dict[str, pd.DataFrame]):
    """Train on each source, test on each (common features only)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    common = None
    for d in sources.values():
        cols = set(d.columns) - {"client_id", "churn", "geography", "country"}
        common = cols if common is None else common & cols
    common = sorted(common)
    splits = {k: train_test_split(d, test_size=0.3, stratify=d["churn"], random_state=config.SEED) for k, d in sources.items()}
    def X(d):
        return pd.get_dummies(d[common], dtype=float)
    res, l1keep, shap5 = {}, {}, {}
    from ds.xai import top_k
    import shap
    for a, (tra, _) in splits.items():
        Xa = X(tra)
        m = HistGradientBoostingClassifier(random_state=config.SEED).fit(Xa, tra["churn"].astype(int))
        res[a] = {}
        for b, (_, teb) in splits.items():
            Xb = X(teb).reindex(columns=Xa.columns, fill_value=0)
            res[a][b] = roc_auc_score(teb["churn"].astype(int), m.predict_proba(Xb)[:, 1])
        sv = shap.TreeExplainer(m).shap_values(Xa.sample(min(400, len(Xa)), random_state=1))
        shap5[a] = ", ".join(top_k(pd.DataFrame(sv, columns=Xa.columns), 5))
        sc = StandardScaler().fit(Xa.fillna(Xa.median()))
        lr = make_logreg(0.02, 1.0).fit(sc.transform(Xa.fillna(Xa.median())), tra["churn"].astype(int))
        l1keep[a] = ", ".join(c for c, v in zip(Xa.columns, lr.coef_[0]) if abs(v) > 1e-8)
    mat = pd.DataFrame(res).T
    mat.index.name = "trained on \\ tested on"
    return mat, pd.DataFrame({"L1 keeps (C=0.02)": l1keep, "SHAP top-5": shap5}), common
