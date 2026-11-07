"""Step 3 — models, cost-based threshold, metrics, contract matrix, perturbations.

Pipeline layout (imblearn, so every sampler runs ONLY inside the training folds):
    impute+ordinal-encode -> [sampler: ROS / SMOTE / SMOTE-NC] -> scale+one-hot -> classifier
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

import config
from ds.catalog import ALL_FEATURES

MODELS = {"logreg": "Logistic regression (baseline)", "rf": "Random Forest",
          "gb": "Gradient Boosting (black box)", "mlp": "Neural network MLP (DL)"}
BALANCING = ["none", "class_weight", "ros", "smote", "smotenc"]
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0, "penalty": "l2"},
    "rf": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5},
    "gb": {"learning_rate": 0.05, "max_depth": 4, "max_iter": 300},
    "mlp": {"hidden": "64,32", "alpha": 1e-3},
}


@dataclass
class RunConfig:
    model: str = "gb"
    params: dict = field(default_factory=dict)
    balancing: str = "class_weight"
    features: list = field(default_factory=lambda: list(ALL_FEATURES))
    threshold_mode: str = "cost"      # cost | cost+recall | 0.5 | manual
    min_recall: float = config.CONTRACT["M1_recall_min"] + 0.04   # cost+recall: out-of-fold target with a safety margin (OOF recall is slightly optimistic)
    manual_threshold: float = 0.5
    cost_fn: float = config.COST_FN
    cost_fp: float = config.COST_FP
    per_group_thresholds: bool = False
    group_col: str = "age_group"
    noise_x_sigma: float = 0.0        # Gaussian noise on numeric X (in std units), training only
    noise_y_flip: float = 0.0         # share of training labels flipped
    bias_y_silent: float = 0.0        # share of SILENT leavers relabelled "stayed" (training)
    bias_x_group: str = ""            # e.g. "age_group=60+" : remove part of this group from training
    bias_x_share: float = 0.0
    seed: int = config.SEED

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------- helpers
def age_group(age: pd.Series) -> pd.Series:
    """Age bands; a missing age (e.g. an impossible age blanked by cleaning) becomes "unknown"."""
    g = pd.cut(age, [0, 29, 44, 59, 200], labels=["18-29", "30-44", "45-59", "60+"])
    return g.cat.add_categories("unknown").fillna("unknown").astype(str)


def group_labels(s: pd.Series) -> np.ndarray:
    """Group values as plain strings, missing -> "missing" (pandas 3 keeps NaN inside str columns)."""
    return np.array(["missing" if pd.isna(v) else str(v) for v in s], dtype=object)


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = age_group(df["age"])
    df["silent"] = ((df["no_feedback_6m"] == 1) & (df["nps_missing"] == 1)).astype(int)
    df["age_country"] = df["age_group"] + "|" + df["country"].astype(str)
    return df


def load_dataset(name: str) -> pd.DataFrame:
    return add_groups(pd.read_csv(config.DATASETS / f"{name}.csv"))


def split(df: pd.DataFrame, seed: int = config.SEED, test_size: float = 0.2):
    return train_test_split(df, test_size=test_size, stratify=df["churn"], random_state=seed)


def cat_num(features):
    cats = [f for f in features if f in config.CATEGORICAL]
    nums = [f for f in features if f not in config.CATEGORICAL]
    return cats, nums


# ----------------------------------------------------------------- pipeline
def make_pipeline(cfg: RunConfig, n_pos: int | None = None) -> Pipeline:
    cats, nums = cat_num(cfg.features)
    p = {**DEFAULT_PARAMS[cfg.model], **(cfg.params or {})}
    enc = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), nums),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cats),
    ], verbose_feature_names_out=False)
    n_num = len(nums)
    cat_idx = list(range(n_num, n_num + len(cats)))
    post = ColumnTransformer([
        ("num", StandardScaler(), list(range(n_num))),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
    ])
    cw = "balanced" if cfg.balancing == "class_weight" else None
    if cfg.model == "logreg":
        pen = p.get("penalty", "l2")
        l1r = {"l2": 0.0, "l1": 1.0}.get(pen, float(p.get("l1_ratio", 0.5)))
        clf = make_logreg(float(p["C"]), l1r, cw, cfg.seed)
    elif cfg.model == "rf":
        clf = RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
                                     min_samples_leaf=int(p["min_samples_leaf"]), class_weight=cw,
                                     n_jobs=-1, random_state=cfg.seed)
    elif cfg.model == "gb":
        clf = HistGradientBoostingClassifier(learning_rate=float(p["learning_rate"]), max_depth=int(p["max_depth"]),
                                             max_iter=int(p["max_iter"]), class_weight=cw, early_stopping=True,
                                             validation_fraction=0.15, n_iter_no_change=20, random_state=cfg.seed)
    elif cfg.model == "mlp":
        hid = tuple(int(h) for h in str(p["hidden"]).split(",") if h.strip())
        clf = MLPClassifier(hidden_layer_sizes=hid, alpha=float(p["alpha"]), early_stopping=True,
                            max_iter=500, random_state=cfg.seed)
    else:
        raise ValueError(cfg.model)

    sampler = None
    bal = cfg.balancing
    if cfg.model == "mlp" and bal == "class_weight":
        bal = "ros"  # MLP has no class_weight -> random oversampling (stated in the report)
    if bal == "ros":
        sampler = RandomOverSampler(random_state=cfg.seed)
    elif bal == "smote":
        sampler = SMOTE(random_state=cfg.seed, k_neighbors=5)
    elif bal == "smotenc":
        sampler = SMOTENC(categorical_features=cat_idx, random_state=cfg.seed) if cat_idx else SMOTE(random_state=cfg.seed)
    steps = [("encode", enc)]
    if sampler is not None:
        steps.append(("sampler", sampler))
        # plain SMOTE interpolates the ordinal codes of country/gender (e.g. 0.37) -> round them back,
        # otherwise every invented value becomes a new one-hot category
        steps.append(("round_cats", FunctionTransformer(_round_cols, kw_args={"idx": cat_idx})))
    steps += [("scale", post), ("clf", clf)]
    return Pipeline(steps)


def _round_cols(Z, idx):
    Z = np.array(Z, dtype=float, copy=True)
    if idx:
        Z[:, idx] = np.round(Z[:, idx])
    return Z


def make_logreg(C: float, l1_ratio: float, class_weight=None, seed=config.SEED) -> LogisticRegression:
    """L2 (l1_ratio=0), L1 (=1) or ElasticNet (in between), across sklearn versions.
    sklearn >= 1.8 uses l1_ratio alone; older versions need penalty=..."""
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    solver = "lbfgs" if l1_ratio == 0 else "saga"
    kw = dict(C=C, max_iter=5000, class_weight=class_weight, random_state=seed, solver=solver)
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, **kw)
    if l1_ratio == 0:
        return LogisticRegression(penalty="l2", **kw)
    if l1_ratio == 1:
        return LogisticRegression(penalty="l1", **kw)
    return LogisticRegression(penalty="elasticnet", l1_ratio=l1_ratio, **kw)


def feature_names_out(pipe: Pipeline, features) -> list[str]:
    cats, nums = cat_num(features)
    names = list(nums)
    ohe = pipe.named_steps["scale"].named_transformers_["cat"]
    if cats:
        for c, cv in zip(cats, ohe.categories_):
            enc = pipe.named_steps["encode"].named_transformers_["cat"].named_steps["ord"]
            labels = enc.categories_[cats.index(c)]
            for v in cv:
                lab = labels[int(v)] if 0 <= int(v) < len(labels) else "unknown"
                names.append(f"{c}={lab}")
    return names


def transform(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Z = pipe.named_steps["encode"].transform(X)
    return pipe.named_steps["scale"].transform(Z)


# ----------------------------------------------------------------- perturbations (training only)
def perturb_training(train: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    tr = train.copy()
    if cfg.bias_x_group and cfg.bias_x_share > 0:
        col, val = cfg.bias_x_group.split("=")
        idx = tr.index[(tr[col].astype(str) == val)]
        drop = rng.choice(idx, int(len(idx) * cfg.bias_x_share), replace=False)
        tr = tr.drop(index=drop)
    if cfg.bias_y_silent > 0:
        idx = tr.index[(tr["churn"] == 1) & (tr["silent"] == 1)]
        flip = rng.choice(idx, int(len(idx) * cfg.bias_y_silent), replace=False)
        tr.loc[flip, "churn"] = 0
    if cfg.noise_y_flip > 0:
        flip = rng.choice(tr.index, int(len(tr) * cfg.noise_y_flip), replace=False)
        tr.loc[flip, "churn"] = 1 - tr.loc[flip, "churn"]
    if cfg.noise_x_sigma > 0:
        _, nums = cat_num(cfg.features)
        for c in nums:
            sd = tr[c].std()
            if sd and np.isfinite(sd):
                tr[c] = tr[c] + rng.normal(0, cfg.noise_x_sigma * sd, len(tr))
    return tr


# ----------------------------------------------------------------- costs and metrics
def expected_cost(y, pred, cost_fn=config.COST_FN, cost_fp=config.COST_FP, save=config.OFFER_SAVE_RATE):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    return float(fn * cost_fn + fp * cost_fp + tp * (cost_fp + (1 - save) * cost_fn))


def cost_curve(y, p, cost_fn=config.COST_FN, cost_fp=config.COST_FP) -> pd.DataFrame:
    ths = np.round(np.arange(0.01, 0.991, 0.01), 2)
    return pd.DataFrame({"threshold": ths,
                         "cost": [expected_cost(y, (p >= t).astype(int), cost_fn, cost_fp) for t in ths],
                         "recall": [recall_score(y, (p >= t).astype(int), zero_division=0) for t in ths],
                         "precision": [precision_score(y, (p >= t).astype(int), zero_division=0) for t in ths]})


def best_threshold(y, p, cost_fn=config.COST_FN, cost_fp=config.COST_FP) -> float:
    cc = cost_curve(y, p, cost_fn, cost_fp)
    return float(cc.loc[cc["cost"].idxmin(), "threshold"])


def lift_at(y, p, top=0.10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(len(p) * top))
    return float(y[np.argsort(-p)[:k]].mean() / max(1e-9, y.mean()))


def metrics(y, p, thr, cost_fn=config.COST_FN, cost_fp=config.COST_FP, pred=None) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int) if pred is None else np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "lift_top10": lift_at(y, p), "threshold": thr,
        "expected_cost": expected_cost(y, pred, cost_fn, cost_fp),
        "cost_no_model": float(y.sum() * cost_fn),
        "cost_offer_all": expected_cost(y, np.ones_like(y), cost_fn, cost_fp),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def fairness_table(df: pd.DataFrame, y, pred, p, groups=("age_group", "gender", "country")) -> pd.DataFrame:
    rows = []
    y, pred = np.asarray(y), np.asarray(pred)
    for g in groups:
        vals = group_labels(df[g])
        for v in sorted(set(vals)):
            m = vals == v
            yy, pp = y[m], pred[m]
            rows.append({"attribute": g, "group": v, "n": int(m.sum()), "churn_rate": yy.mean(),
                         "recall": recall_score(yy, pp, zero_division=0) if yy.sum() else np.nan,
                         "selection_rate": pp.mean(),
                         "fpr": ((pp == 1) & (yy == 0)).sum() / max(1, (yy == 0).sum()),
                         "auc": roc_auc_score(yy, np.asarray(p)[m]) if 0 < yy.sum() < len(yy) else np.nan})
    return pd.DataFrame(rows)


def fairness_gaps(ft: pd.DataFrame) -> dict:
    return {a: float(g["recall"].max() - g["recall"].min()) for a, g in ft.groupby("attribute")}


def group_thresholds(df_tr, y, p, col, target_recall) -> dict:
    """Per-group thresholds that equalise recall (bias mitigation)."""
    out = {}
    vals = group_labels(df_tr[col])
    for v in sorted(set(vals)):
        m = (vals == v) & (np.asarray(y) == 1)
        if m.sum() < 5:
            continue
        out[v] = float(np.quantile(np.asarray(p)[m], 1 - target_recall))
    return out


def apply_thresholds(df, p, thr, gthr: dict | None, col):
    if not gthr:
        return (np.asarray(p) >= thr).astype(int)
    t = df[col].astype(str).map(gthr).fillna(thr).values
    return (np.asarray(p) >= t).astype(int)


# ----------------------------------------------------------------- the full train/evaluate step
@dataclass
class RunResult:
    cfg: RunConfig
    pipe: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame
    p_oof: np.ndarray
    p_test: np.ndarray
    threshold: float
    group_thr: dict
    pred_test: np.ndarray
    metrics: dict
    fairness: pd.DataFrame
    gaps: dict
    cv_auc: float
    fit_seconds: float
    latency_ms: float


def oof_predict(cfg: RunConfig, df: pd.DataFrame, folds: int = 5) -> np.ndarray:
    skf = StratifiedKFold(folds, shuffle=True, random_state=cfg.seed)
    oof = np.zeros(len(df))
    X, y = df[cfg.features], df["churn"].values
    for tr, va in skf.split(X, y):
        pipe = make_pipeline(cfg).fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


def run(cfg: RunConfig, df: pd.DataFrame | None = None, train=None, test=None, folds: int = 4) -> RunResult:
    if train is None:
        train, test = split(df, cfg.seed)
    tr = perturb_training(train, cfg)
    p_oof = oof_predict(cfg, tr, folds)
    t0 = time.time()
    pipe = make_pipeline(cfg).fit(tr[cfg.features], tr["churn"].values)
    fit_s = time.time() - t0
    p_test = pipe.predict_proba(test[cfg.features])[:, 1]
    t1 = time.time()
    for i in range(min(50, len(test))):
        pipe.predict_proba(test[cfg.features].iloc[[i]])
    lat = (time.time() - t1) / min(50, len(test)) * 1000
    if cfg.threshold_mode == "cost":
        thr = best_threshold(tr["churn"].values, p_oof, cfg.cost_fn, cfg.cost_fp)
    elif cfg.threshold_mode == "cost+recall":  # cheapest threshold that still meets contract row M1 (out-of-fold)
        cc = cost_curve(tr["churn"].values, p_oof, cfg.cost_fn, cfg.cost_fp)
        ok = cc[cc["recall"] >= cfg.min_recall]
        thr = float(ok.loc[ok["cost"].idxmin(), "threshold"]) if len(ok) else 0.01
    elif cfg.threshold_mode == "manual":
        thr = cfg.manual_threshold
    else:
        thr = 0.5
    gthr = {}
    if cfg.per_group_thresholds:
        overall_recall = recall_score(tr["churn"], (p_oof >= thr).astype(int))
        gthr = group_thresholds(tr, tr["churn"].values, p_oof, cfg.group_col, overall_recall)
    pred = apply_thresholds(test, p_test, thr, gthr, cfg.group_col)
    met = metrics(test["churn"].values, p_test, thr, cfg.cost_fn, cfg.cost_fp, pred=pred)
    met["latency_ms_single"] = lat
    # fairness is measured on ALL out-of-sample predictions (out-of-fold for training rows + test rows):
    # test groups alone are too small (25-50 leavers per age group) for a stable recall estimate
    all_df = pd.concat([train.loc[tr.index], test])
    p_all = np.r_[p_oof, p_test]
    pred_all = apply_thresholds(all_df, p_all, thr, gthr, cfg.group_col)
    ft = fairness_table(all_df, all_df["churn"].values, pred_all, p_all)
    return RunResult(cfg, pipe, tr, test, p_oof, p_test, thr, gthr, pred, met, ft, fairness_gaps(ft),
                     roc_auc_score(tr["churn"], p_oof), fit_s, lat)


# ----------------------------------------------------------------- contract matrix
def contract_matrix(res: RunResult, extra: dict | None = None, th: dict | None = None) -> pd.DataFrame:
    th = {**config.CONTRACT, **(th or {})}
    extra = extra or {}
    m = res.metrics
    gap = max(res.gaps.values()) if res.gaps else np.nan
    rows = [
        ("M1 recall of leavers", f">= {th['M1_recall_min']:.2f}", m["recall"], m["recall"] >= th["M1_recall_min"]),
        ("M2a ROC-AUC", f">= {th['M2_auc_min']:.2f}", m["roc_auc"], m["roc_auc"] >= th["M2_auc_min"]),
        ("M2b expected cost vs no model (EUR)", f"< {m['cost_no_model']:.0f}", m["expected_cost"],
         m["expected_cost"] < m["cost_no_model"]),
        ("Fairness: max recall gap (age/gender/country)", f"<= {th['fairness_gap_max']:.2f}", gap,
         bool(gap <= th["fairness_gap_max"])),
        ("Leakage alarm: AUC suspiciously high", f"< {th['leakage_auc_max']:.2f}", m["roc_auc"],
         m["roc_auc"] < th["leakage_auc_max"]),
    ]
    if "jaccard" in extra:
        rows.append(("F1 XAI stability (Jaccard top-5)", f">= {th['F1_jaccard_min']:.2f}", extra["jaccard"],
                     extra["jaccard"] >= th["F1_jaccard_min"]))
    if "rank_corr" in extra:
        rows.append(("F2a external SHAP rank correlation", f">= {th['F2_rankcorr_min']:.2f}", extra["rank_corr"],
                     extra["rank_corr"] >= th["F2_rankcorr_min"]))
    if "ext_auc" in extra:
        rows.append(("F2b external ROC-AUC", f">= {th['M2_auc_min']:.2f}", extra["ext_auc"],
                     extra["ext_auc"] >= th["M2_auc_min"]))
    if "max_psi" in extra:
        rows.append(("F2c max PSI (drift) external", f"< {th['F2_psi_max']:.2f}", extra["max_psi"],
                     extra["max_psi"] < th["F2_psi_max"]))
    if "data_contract" in extra:
        rows.append(("Data contract (input validity)", "all rules pass", float(extra["data_contract"]),
                     bool(extra["data_contract"])))
    if "latency_p95" in extra:
        rows.append(("Streaming latency p95 (ms)", f"< {th['latency_p95_ms_max']:.0f}", extra["latency_p95"],
                     extra["latency_p95"] < th["latency_p95_ms_max"]))
    out = pd.DataFrame(rows, columns=["requirement", "threshold", "measured", "pass"])
    out["verdict"] = np.where(out["pass"], "PASS", "FAIL")
    return out


def cost_matrix_table(cost_fn=config.COST_FN, cost_fp=config.COST_FP, save=config.OFFER_SAVE_RATE):
    return pd.DataFrame({"Predicted: stays": ["0 €", f"{cost_fn:.0f} € (lost client)"],
                         "Predicted: leaves": [f"{cost_fp:.0f} € (useless offer)",
                                               f"{cost_fp:.0f} € + {(1 - save):.0%}·{cost_fn:.0f} € = "
                                               f"{cost_fp + (1 - save) * cost_fn:.0f} €"]},
                        index=["Actually stays", "Actually leaves"])