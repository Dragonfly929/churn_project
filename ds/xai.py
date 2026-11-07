"""Step 4 — XAI: SHAP (global + local), LIME, stability (KPI F1),
reason codes, and the internal-vs-external comparison (KPI F2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from ds.catalog import REASON_OF, REASONS
from ds.modeling import RunConfig, cat_num, feature_names_out, make_pipeline, transform

warnings.filterwarnings("ignore", category=UserWarning)


def _base_feature(name: str) -> str:
    return name.split("=")[0]


def shap_frame(pipe, X: pd.DataFrame, features, background: pd.DataFrame | None = None,
               max_rows: int = 600, seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values aggregated back to the ORIGINAL features (one-hot columns summed).
    Returns (shap_df, X_used)."""
    import shap
    Xs = X.sample(min(max_rows, len(X)), random_state=seed) if len(X) > max_rows else X
    Z = transform(pipe, Xs[features])
    names = feature_names_out(pipe, features)
    clf = pipe.named_steps["clf"]
    kind = type(clf).__name__
    if kind in ("HistGradientBoostingClassifier", "RandomForestClassifier", "GradientBoostingClassifier"):
        ex = shap.TreeExplainer(clf)
        sv = ex.shap_values(Z, check_additivity=False)
    else:
        bg = background if background is not None else X
        Zb = transform(pipe, bg[features].sample(min(150, len(bg)), random_state=seed))
        if kind == "LogisticRegression":
            ex = shap.LinearExplainer(clf, shap.maskers.Independent(Zb, max_samples=len(Zb)))
            sv = ex.shap_values(Z)
        else:
            f = lambda a: clf.predict_proba(a)[:, 1]  # noqa: E731
            ex = shap.Explainer(f, Zb, algorithm="permutation", seed=seed)
            sv = ex(Z[:min(len(Z), 200)]).values
            Xs = Xs.iloc[:len(sv)]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    raw = pd.DataFrame(sv, columns=names, index=Xs.index)
    agg = raw.T.groupby(lambda c: _base_feature(c)).sum().T
    return agg[[f for f in features if f in agg.columns]], Xs


def global_importance(sdf: pd.DataFrame) -> pd.Series:
    return sdf.abs().mean().sort_values(ascending=False)


def top_k(sdf: pd.DataFrame, k: int = 5) -> list[str]:
    return global_importance(sdf).index[:k].tolist()


def reason_code(row: pd.Series) -> tuple[str, dict]:
    """Group positive SHAP contributions into business reasons; the largest wins."""
    sums = {r: 0.0 for r in REASONS}
    for f, v in row.items():
        if v > 0 and f in REASON_OF:
            sums[REASON_OF[f]] += v
    nonprofile = {r: v for r, v in sums.items() if r != "profile"}
    best = max(nonprofile, key=nonprofile.get) if max(nonprofile.values()) > 0 else "profile"
    return best, sums


def stability(cfg: RunConfig, df: pd.DataFrame, n: int = 5, k: int = 5) -> tuple[float, list]:
    """KPI F1: retrain on bootstrap samples with different seeds; mean pairwise Jaccard of top-k."""
    tops = []
    for i in range(n):
        c = RunConfig(**{**cfg.to_dict(), "seed": cfg.seed + i + 1})
        boot = df.sample(frac=0.8, random_state=cfg.seed + i)
        pipe = make_pipeline(c).fit(boot[c.features], boot["churn"])
        sdf, _ = shap_frame(pipe, df[c.features], c.features, max_rows=400, seed=i)
        tops.append(top_k(sdf, k))
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for i, a in enumerate(tops) for b in tops[i + 1:]]
    return float(np.mean(jac)), tops


def lime_explainer(pipe, train: pd.DataFrame, features, seed=config.SEED):
    """Build LIME once on the original feature space (categoricals ordinal-coded for LIME).
    Returns (explainer, encode, predict): encode turns rows into LIME's numbers, predict scores them."""
    from lime.lime_tabular import LimeTabularExplainer
    cats, _ = cat_num(features)
    maps = {c: sorted(train[c].dropna().astype(str).unique()) for c in cats}
    Xtr = train[features].copy()
    for c in cats:
        Xtr[c] = Xtr[c].astype(str).map({v: i for i, v in enumerate(maps[c])})
    med = Xtr.astype(float).median()

    def encode(df):
        a = df[features].copy()
        for c in cats:
            a[c] = a[c].astype(str).map({v: i for i, v in enumerate(maps[c])}).fillna(0)
        return a.astype(float).fillna(med)

    def predict(arr):
        d = pd.DataFrame(arr, columns=features)
        for c in cats:
            d[c] = d[c].round().clip(0, len(maps[c]) - 1).astype(int).map(lambda i, c=c: maps[c][i])
        return pipe.predict_proba(d)

    ex = LimeTabularExplainer(encode(train).values, feature_names=features, class_names=["stays", "leaves"],
                              categorical_features=[features.index(c) for c in cats],
                              categorical_names={features.index(c): maps[c] for c in cats},
                              discretize_continuous=True, random_state=seed, mode="classification")
    return ex, encode, predict


def lime_explain(pipe, train: pd.DataFrame, row: pd.Series, features, num_features: int = 8, seed=config.SEED):
    """LIME for one client: the conditions LIME found and their weights towards 'leaves'."""
    ex, encode, predict = lime_explainer(pipe, train, features, seed)
    x = encode(row.to_frame().T).values[0]
    exp = ex.explain_instance(x, predict, num_features=num_features, labels=(1,))
    out = pd.DataFrame(exp.as_list(label=1), columns=["condition", "weight"])
    out["feature"] = [next((f for f in sorted(features, key=len, reverse=True) if f in c), c) for c in out["condition"]]
    return out


def _normalize(v: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(v))
    return v / m if m > 0 else v


def interpret_difference(diff: float) -> str:
    """Same four levels as the teacher's SHAP vs LIME app."""
    if diff < 0.2:
        return "strong agreement: similar influence"
    if diff < 0.5:
        return "moderate difference: probably LIME's local limits"
    if diff < 0.8:
        return "large difference: probably feature interactions"
    return "very large difference: possible model instability"


def shap_lime_map(pipe, train: pd.DataFrame, test: pd.DataFrame, features, n: int = 150,
                  num_samples: int = 1500, seed: int = config.SEED) -> dict:
    """Client map for the SHAP vs LIME check (idea of the teacher's app_dec_xai).

    For n test clients (leavers and stayers, stratified): SHAP and LIME contributions for every
    feature, each normalised by its own largest absolute value; their Pearson correlation per
    client; the absolute difference per feature; and a 2D PCA projection of the clients in the
    model's own (encoded, scaled) feature space. PCA is only for drawing: the model uses all features.
    """
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA

    k = min(n, len(test))
    parts = [g.sample(max(1, round(k * len(g) / len(test))), random_state=seed) for _, g in test.groupby("churn")]
    sample = pd.concat(parts).sort_index()
    sdf, used = shap_frame(pipe, sample, features, background=train, max_rows=len(sample), seed=seed)
    sample = sample.loc[used.index]
    shap_m = sdf.reindex(columns=features).fillna(0.0).values

    ex, encode, predict = lime_explainer(pipe, train, features, seed)
    Xl = encode(sample).values
    lime_m = np.zeros_like(shap_m)
    for i, x in enumerate(Xl):
        exp = ex.explain_instance(x, predict, num_features=len(features), labels=(1,), num_samples=num_samples)
        for j, w in exp.local_exp[1]:
            lime_m[i, j] = w

    shap_n = np.vstack([_normalize(r) for r in shap_m])
    lime_n = np.vstack([_normalize(r) for r in lime_m])
    corr = np.array([pearsonr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
                     for a, b in zip(shap_n, lime_n)])
    pca = PCA(2, random_state=seed)
    Z = pca.fit_transform(transform(pipe, sample[features]))
    p = pipe.predict_proba(sample[features])[:, 1]
    points = pd.DataFrame({"client_id": sample["client_id"].values, "PC1": Z[:, 0], "PC2": Z[:, 1],
                           "risk": p, "left": sample["churn"].values, "corr": corr}, index=sample.index)
    return {"points": points, "values": sample[features],
            "shap": pd.DataFrame(shap_n, index=sample.index, columns=features),
            "lime": pd.DataFrame(lime_n, index=sample.index, columns=features),
            "explained_variance": float(pca.explained_variance_ratio_.sum())}


def shap_lime_agreement(shap_row: pd.Series, lime_df: pd.DataFrame) -> dict:
    lw = lime_df.groupby("feature")["weight"].sum()
    common = [f for f in lw.index if f in shap_row.index]
    if len(common) < 3:
        return {"rank_corr": np.nan, "sign_agreement": np.nan, "n": len(common)}
    rho = spearmanr(shap_row[common].abs(), lw[common].abs()).statistic
    sign = float(np.mean(np.sign(shap_row[common]) == np.sign(lw[common])))
    return {"rank_corr": float(rho), "sign_agreement": sign, "n": len(common)}


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (data drift)."""
    e, a = expected.dropna().astype(float), actual.dropna().astype(float)
    if e.nunique() <= 1 or a.empty:
        return 0.0
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        cats = sorted(set(e.unique()) | set(a.unique()))
        pe = np.array([(e == c).mean() for c in cats]) + 1e-4
        pa = np.array([(a == c).mean() for c in cats]) + 1e-4
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        pe = np.histogram(e, qs)[0] / len(e) + 1e-4
        pa = np.histogram(a, qs)[0] / len(a) + 1e-4
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_table(ref: pd.DataFrame, new: pd.DataFrame, features) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in config.CATEGORICAL:
            cats = sorted(set(ref[f].dropna()) | set(new[f].dropna()))
            pe = np.array([(ref[f] == c).mean() for c in cats]) + 1e-4
            pa = np.array([(new[f] == c).mean() for c in cats]) + 1e-4
            v = float(np.sum((pa - pe) * np.log(pa / pe)))
        else:
            v = psi(ref[f], new[f])
        rows.append({"feature": f, "psi": v, "drift": "HIGH" if v >= 0.2 else ("moderate" if v >= 0.1 else "low")})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def compare_importance(imp_int: pd.Series, imp_ext: pd.Series) -> dict:
    common = [f for f in imp_int.index if f in imp_ext.index]
    rho = spearmanr(imp_int[common], imp_ext[common]).statistic
    flips = [f for f in common[:8] if (imp_ext.rank(ascending=False)[f] - imp_int.rank(ascending=False)[f]) > 5]
    return {"rank_corr": float(rho), "top5_int": list(imp_int.index[:5]), "top5_ext": list(imp_ext.index[:5]),
            "jaccard_top5": len(set(imp_int.index[:5]) & set(imp_ext.index[:5])) / len(set(imp_int.index[:5]) | set(imp_ext.index[:5])),
            "moved_down_a_lot": flips}


def pick_contrasting_clients(test: pd.DataFrame, p: np.ndarray) -> dict:
    """A silent leaver and a vocal leaver with high predicted risk (step 4, SHAP local)."""
    t = test.assign(p=p)
    lv = t[t["churn"] == 1]
    if "no_feedback_6m" not in t.columns:  # sources without behaviour features (github)
        return {"highest-risk leaver": lv.sort_values("p", ascending=False).index[0]} if len(lv) else {}
    silent = lv[(lv["no_feedback_6m"] == 1) & (lv["nps_missing"] == 1)].sort_values("p", ascending=False)
    vocal = lv[(lv["complaints_180d"] > 0)].sort_values("p", ascending=False)
    out = {}
    if len(silent):
        out["silent leaver"] = silent.index[0]
    if len(vocal):
        out["vocal leaver"] = vocal.index[0]
    return out
