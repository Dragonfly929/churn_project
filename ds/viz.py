"""Interactive charts (Plotly) for the two Streamlit apps — one visual system.

Palette: built from the project's colour inspiration (teal #0C7B93 / pink #FF2E63 / lavender #7469B6 /
plum #8A2A78 / peach #F2A67B ...), adjusted to the steps that pass the colour checks
(lightness band, chroma floor, colour-blind separation between neighbours).
Fixed order: colour follows the entity — "stays" is always teal, "leaves" always pink.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

CAT = ["#1A8FB0", "#FF2E63", "#7469B6", "#2E9E6E", "#B04A98", "#E07B39", "#3E6FD0", "#B88A10"]
STAYS, LEAVES = CAT[0], CAT[1]
PLUM, NAVY, INK, MUTED, GRID = "#8A2A78", "#142850", "#252A34", "#6B6F7B", "#ECE7F0"
SEQ = ["#E6F7F9", "#C6EBF0", "#9FDCE5", "#6FC6D6", "#3FAEC4", "#1A8FB0", "#16759A", "#1B5C82", "#1E4468", "#142850"]
DIV = [[0, "#1A8FB0"], [0.5, "#F2F0F4"], [1, "#FF2E63"]]   # two poles + neutral midpoint
CHURN_COLORS = {"stays": STAYS, "leaves": LEAVES}

pio.templates["churn"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=15, color=INK),
    title=dict(font=dict(size=19, color=NAVY, weight=700), x=0.0, xanchor="left", y=0.98, yanchor="top"),
    colorway=CAT, paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor="#D6CDE0", linecolor="#C9BFD6", ticks="outside", tickcolor="#C9BFD6",
               tickfont=dict(size=14), title=dict(font=dict(size=15))),
    yaxis=dict(automargin=True, gridcolor=GRID, zerolinecolor="#D6CDE0", linecolor="#C9BFD6", tickfont=dict(size=14),
               title=dict(font=dict(size=15))),
    hoverlabel=dict(font=dict(size=14), bgcolor="white", bordercolor=PLUM),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=14), title=None),
    margin=dict(l=10, r=30, t=95, b=10), bargap=0.25,
))
pio.templates.default = "churn"


def _lab(y):
    return np.where(np.asarray(y) == 1, "leaves", "stays")


# ------------------------------------------------------------------ dataset / EDA
def hist_by_churn(df, col, bins=40, clip=True):
    d = df[[col, "churn"]].dropna().copy()
    if clip and pd.api.types.is_numeric_dtype(d[col]):
        lo, hi = d[col].quantile([0.01, 0.99])
        d[col] = d[col].clip(lo, hi)
    d["client"] = _lab(d["churn"])
    f = go.Figure()
    for k in ["stays", "leaves"]:
        f.add_histogram(x=d.loc[d.client == k, col], name=k, marker_color=CHURN_COLORS[k], opacity=0.75,
                        nbinsx=bins, histnorm="percent",
                        hovertemplate=f"{k}<br>{col}: %{{x}}<br>%{{y:.1f}}% of {k}<extra></extra>")
    f.update_layout(barmode="overlay", title=f"{col}: leavers vs stayers (share of each group)",
                    xaxis_title=col, yaxis_title="% of group", height=380)
    return f


def box_by_churn(df, col):
    d = df[[col, "churn"]].dropna().copy()
    d["client"] = _lab(d["churn"])
    f = go.Figure()
    for k in ["stays", "leaves"]:
        f.add_box(y=d.loc[d.client == k, col], name=k, marker_color=CHURN_COLORS[k], boxmean=True)
    f.update_layout(title=f"{col}: spread and outliers", showlegend=False, height=380)
    return f


def rate_by_segment(df, col, overall=None):
    s = df.groupby(col)["churn"].agg(["mean", "count"]).reset_index()
    s[col] = s[col].astype(str)
    f = go.Figure(go.Bar(x=s[col], y=s["mean"], marker_color=PLUM, text=[f"{v:.0%}" for v in s["mean"]],
                         textposition="outside", customdata=s["count"],
                         hovertemplate=f"{col}=%{{x}}<br>churn rate %{{y:.1%}}<br>clients %{{customdata}}<extra></extra>"))
    if overall is not None:
        f.add_hline(y=overall, line_dash="dash", line_color=MUTED, annotation_text=f"overall {overall:.0%}",
                    annotation_position="top left")
    f.update_layout(title=f"churn rate by {col}", yaxis_tickformat=".0%", height=340,
                    yaxis_range=[0, max(0.05, s["mean"].max() * 1.25)])
    return f


def scatter_by_churn(df, x, y, n=2500):
    d = df[list(dict.fromkeys([x, y, "churn", "client_id"]))].dropna()
    d = d.sample(min(n, len(d)), random_state=1)
    d["client"] = _lab(d["churn"])
    f = go.Figure()
    for k in ["stays", "leaves"]:
        g = d[d.client == k]
        f.add_scatter(x=g[x], y=g[y], mode="markers", name=k, customdata=g["client_id"],
                      marker=dict(color=CHURN_COLORS[k], size=8, opacity=0.65, line=dict(width=1, color="white")),
                      hovertemplate=f"client %{{customdata}}<br>{x}: %{{x}}<br>{y}: %{{y}}<extra>{k}</extra>")
    f.update_layout(title=f"{x} vs {y}", xaxis_title=x, yaxis_title=y, height=440)
    return f


def corr_heatmap(corr):
    f = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale=DIV, zmin=-1, zmax=1,
                             hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>", colorbar=dict(title="ρ")))
    f.update_layout(title="Spearman correlations (hover a cell)", height=640, xaxis_tickangle=-60,
                    xaxis=dict(tickfont=dict(size=11)), yaxis=dict(tickfont=dict(size=11), autorange="reversed"))
    return f


def hbar(series, title, color=PLUM, fmt=".3f", height=None, signed=False):
    s = series.sort_values()
    colors = [(LEAVES if v > 0 else STAYS) for v in s.values] if signed else color
    f = go.Figure(go.Bar(x=s.values, y=s.index.astype(str), orientation="h", marker_color=colors,
                         text=[format(v, fmt) for v in s.values], textposition="outside",
                         hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    lo, hi = min(0.0, float(s.min())), max(0.0, float(s.max()))
    pad = (hi - lo) * 0.18 or 1
    f.update_traces(cliponaxis=False)
    f.update_layout(title=title, height=height or max(320, 28 * len(s) + 110), xaxis_title=None,
                    xaxis_range=[lo - (pad if lo < 0 else 0), hi + pad], showlegend=False)
    return f


def table_counts(counts: dict):
    s = pd.Series(counts)
    f = go.Figure(go.Bar(x=s.index, y=s.values, marker_color=[PLUM if k.startswith("dim") else CAT[0] for k in s.index],
                         text=[f"{v:,}" for v in s.values], textposition="outside",
                         hovertemplate="%{x}: %{y:,} rows<extra></extra>"))
    f.update_layout(title="Rows per DWH table (plum = dimensions, teal = facts)", yaxis_type="log",
                    yaxis_title="rows (log scale)", height=380)
    return f


# ------------------------------------------------------------------ model evaluation
def roc_pr(y, p, thr=None, extra: dict | None = None):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    curves = {"model": (y, p)} if extra is None else extra
    f = make_subplots(1, 2, subplot_titles=("ROC curve", "Precision–Recall curve"), horizontal_spacing=0.1)
    for i, (name, (yy, pp)) in enumerate(curves.items()):
        fpr, tpr, th = roc_curve(yy, pp)
        pr, rc, th2 = precision_recall_curve(yy, pp)
        c = CAT[i % len(CAT)] if extra else PLUM
        f.add_scatter(x=fpr, y=tpr, name=f"{name} AUC {roc_auc_score(yy, pp):.3f}", line=dict(color=c, width=2.5),
                      customdata=th, hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<br>threshold %{customdata:.2f}<extra></extra>",
                      row=1, col=1)
        f.add_scatter(x=rc, y=pr, name=f"{name} AP {average_precision_score(yy, pp):.3f}", line=dict(color=c, width=2.5),
                      showlegend=bool(extra), hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<extra></extra>", row=1, col=2)
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED), showlegend=False, row=1, col=1)
    if thr is not None and extra is None:
        pred = (np.asarray(p) >= thr)
        yv = np.asarray(y)
        tpr_t = (pred & (yv == 1)).sum() / max(1, (yv == 1).sum())
        fpr_t = (pred & (yv == 0)).sum() / max(1, (yv == 0).sum())
        prec_t = (pred & (yv == 1)).sum() / max(1, pred.sum())
        f.add_scatter(x=[fpr_t], y=[tpr_t], mode="markers", marker=dict(size=14, color=LEAVES, line=dict(width=2, color="white")),
                      name=f"threshold {thr:.2f}", row=1, col=1)
        f.add_scatter(x=[tpr_t], y=[prec_t], mode="markers", marker=dict(size=14, color=LEAVES, line=dict(width=2, color="white")),
                      showlegend=False, row=1, col=2)
    f.update_xaxes(title_text="false positive rate", row=1, col=1)
    f.update_yaxes(title_text="true positive rate (recall)", row=1, col=1)
    f.update_xaxes(title_text="recall", row=1, col=2)
    f.update_yaxes(title_text="precision", row=1, col=2)
    f.update_layout(height=470, legend=dict(yanchor="top", y=-0.18, xanchor="center", x=0.5), margin=dict(b=40))
    f.update_annotations(font=dict(size=17, color=NAVY))
    return f


def confusion(tn, fp, fn, tp, title="Confusion matrix"):
    z = [[tn, fp], [fn, tp]]
    labels = [[f"TN<br><b>{tn}</b><br>stays, left alone", f"FP<br><b>{fp}</b><br>useless offer"],
              [f"FN<br><b>{fn}</b><br>lost client", f"TP<br><b>{tp}</b><br>caught in time"]]
    f = go.Figure(go.Heatmap(z=z, x=["predicted: stays", "predicted: leaves"], y=["actually stays", "actually leaves"],
                             colorscale=[[0, "#F5EFF8"], [1, "#8A2A78"]], showscale=False, text=labels,
                             texttemplate="%{text}", textfont=dict(size=16), hoverinfo="skip"))
    f.update_layout(title=title, height=360, yaxis_autorange="reversed")
    return f


def cost_curve(cc, thr_opt, thr_now=None):
    f = make_subplots(specs=[[{"secondary_y": False}]])
    f.add_scatter(x=cc["threshold"], y=cc["cost"] / 1000, name="expected cost (k€)", line=dict(color=PLUM, width=3),
                  hovertemplate="threshold %{x:.2f}<br>cost %{y:.1f} k€<extra></extra>")
    f.add_vline(x=thr_opt, line_dash="dash", line_color=LEAVES, annotation_text=f"cost-optimal {thr_opt:.2f}",
                annotation_position="top right")
    f.add_vline(x=0.5, line_dash="dot", line_color=MUTED, annotation_text="0.5 default", annotation_position="bottom right")
    if thr_now is not None and abs(thr_now - thr_opt) > 0.02:
        f.add_vline(x=thr_now, line_color=CAT[0], annotation_text=f"yours {thr_now:.2f}", annotation_position="top left")
    f.update_layout(title="Expected cost vs threshold", xaxis_title="threshold", yaxis_title="k€", height=380, showlegend=False)
    return f


def threshold_tradeoff(cc):
    f = go.Figure()
    f.add_scatter(x=cc["threshold"], y=cc["recall"], name="recall", line=dict(color=LEAVES, width=3))
    f.add_scatter(x=cc["threshold"], y=cc["precision"], name="precision", line=dict(color=CAT[0], width=3))
    f.update_layout(title="Recall and precision vs threshold", xaxis_title="threshold", yaxis_tickformat=".0%", height=380,
                    hovermode="x unified")
    return f


def deciles(d):
    f = go.Figure(go.Bar(x=d.index.astype(str), y=d["churn_rate"], marker_color=SEQ[3:][::-1][:len(d)][::-1] if False else PLUM,
                         text=[f"{v:.0%}" for v in d["churn_rate"]], textposition="outside", customdata=d[["leavers", "clients"]],
                         hovertemplate="decile %{x}<br>churn %{y:.0%}<br>%{customdata[0]} leavers of %{customdata[1]}<extra></extra>"))
    f.update_layout(title="Churn rate by risk decile (10 = highest risk: call these first)", xaxis_title="risk decile",
                    yaxis_tickformat=".0%", height=360)
    return f


def fairness_bars(ft):
    f = make_subplots(1, ft["attribute"].nunique(), subplot_titles=list(ft["attribute"].unique()), shared_yaxes=True)
    for i, (a, g) in enumerate(ft.groupby("attribute", sort=False)):
        f.add_bar(x=g["group"].astype(str), y=g["recall"], marker_color=CAT[i + 2], name=a,
                  text=[f"{v:.0%}" if v == v else "" for v in g["recall"]], textposition="outside",
                  customdata=g[["n", "churn_rate"]],
                  hovertemplate="%{x}<br>recall %{y:.1%}<br>clients %{customdata[0]}<br>churn rate %{customdata[1]:.1%}<extra></extra>",
                  row=1, col=i + 1)
    f.update_yaxes(range=[0, 1.1], tickformat=".0%")
    f.update_layout(title="Recall of leavers by group (equal bars = fair)", showlegend=False, height=380)
    return f


# ------------------------------------------------------------------ XAI
def shap_beeswarm(sdf, X, top=12):
    imp = sdf.abs().mean().sort_values(ascending=False).index[:top][::-1]
    f = go.Figure()
    rng = np.random.default_rng(0)
    for i, feat in enumerate(imp):
        v = pd.to_numeric(X[feat], errors="coerce") if feat in X else pd.Series(np.nan, index=sdf.index)
        vn = ((v - v.min()) / ((v.max() - v.min()) or 1)).fillna(0.5)  # missing value = neutral grey
        f.add_scatter(x=sdf[feat], y=i + rng.uniform(-0.28, 0.28, len(sdf)), mode="markers", showlegend=False,
                      marker=dict(size=6, color=vn, colorscale=[[0, STAYS], [0.5, "#D9D3E3"], [1, LEAVES]], opacity=0.75,
                                  showscale=i == 0, colorbar=dict(title="feature<br>value", tickvals=[0, 1], ticktext=["low", "high"])),
                      customdata=np.c_[v.values], hovertemplate=f"{feat}<br>SHAP %{{x:.3f}}<br>value %{{customdata[0]}}<extra></extra>")
    f.add_vline(x=0, line_color=MUTED)
    f.update_layout(title="SHAP beeswarm: each dot is a client (right = pushes towards leaving)",
                    yaxis=dict(tickvals=list(range(len(imp))), ticktext=list(imp)), xaxis_title="SHAP value", height=520)
    return f


def shap_local(row, top=10, base=None, p=None):
    s = row.reindex(row.abs().sort_values(ascending=False).index[:top])[::-1]
    f = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[LEAVES if v > 0 else STAYS for v in s.values],
                         text=[f"{v:+.3f}" for v in s.values], textposition="outside",
                         hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    f.add_vline(x=0, line_color=MUTED)
    f.update_layout(title="Why this client? (pink pushes towards leaving, teal towards staying)", height=420, xaxis_title="SHAP")
    return f


def importance_compare(a: pd.Series, b: pd.Series, la="internal", lb="external"):
    d = pd.DataFrame({la: a, lb: b}).fillna(0).sort_values(la)
    f = go.Figure()
    f.add_bar(y=d.index, x=d[la], name=la, orientation="h", marker_color=CAT[0])
    f.add_bar(y=d.index, x=d[lb], name=lb, orientation="h", marker_color=CAT[2])
    f.update_layout(barmode="group", title="Mean |SHAP|: our data vs external data", height=max(420, 28 * len(d)))
    return f


def psi_bars(ps):
    d = ps.sort_values("psi")
    col = [LEAVES if v >= 0.2 else (CAT[5] if v >= 0.1 else CAT[3]) for v in d["psi"]]
    f = go.Figure(go.Bar(x=d["psi"], y=d["feature"], orientation="h", marker_color=col, text=d["drift"],
                         textposition="outside", hovertemplate="%{y}<br>PSI %{x:.3f}<extra></extra>"))
    f.add_vline(x=0.2, line_dash="dash", line_color=LEAVES, annotation_text="0.2 = high drift")
    f.add_vline(x=0.1, line_dash="dot", line_color=CAT[5])
    f.update_layout(title="Data drift per feature (PSI)", height=max(420, 26 * len(d)))
    return f


# ------------------------------------------------------------------ generic from result tables
def lines(df, x, ys: list, title, err: dict | None = None, pct=False, logx=False):
    f = go.Figure()
    for i, y in enumerate(ys):
        e = dict(type="data", array=df[err[y]], visible=True) if err and y in err else None
        f.add_scatter(x=df[x], y=df[y], name=y, mode="lines+markers", line=dict(width=3, color=CAT[i % 8]),
                      marker=dict(size=9), error_y=e)
    f.update_layout(title=title, xaxis_title=x, height=400, hovermode="x unified")
    if pct:
        f.update_yaxes(tickformat=".0%")
    if logx:
        f.update_xaxes(type="log")
    return f


def grouped_bars(df, x, ys: list, title, pct=False):
    f = go.Figure()
    for i, y in enumerate(ys):
        f.add_bar(x=df[x].astype(str), y=df[y], name=y, marker_color=CAT[i % 8],
                  text=[(f"{v:.0%}" if pct else f"{v:,.2f}") if v == v else "" for v in df[y]], textposition="outside")
    f.update_traces(cliponaxis=False)
    f.update_layout(barmode="group", title=title, height=440)
    if pct:
        f.update_yaxes(tickformat=".0%")
    return f


def iterations_chart(entries):
    rows = [{"iteration": f"#{e['id']} {e['name']}", "recall": e["metrics"]["recall"], "precision": e["metrics"]["precision"],
             "ROC-AUC": e["metrics"]["roc_auc"], "max fairness gap": max(e["fairness_gaps"].values()) if e["fairness_gaps"] else np.nan,
             "verdict": e["verdict"]} for e in entries]
    d = pd.DataFrame(rows)
    d["label"] = [f"{it}<br><b><span style='color:{CAT[3] if v == 'PASS' else LEAVES}'>{v}</span></b>"
                  for it, v in zip(d["iteration"], d["verdict"])]
    f = grouped_bars(d, "label", ["recall", "precision", "ROC-AUC", "max fairness gap"], "Iterations compared", pct=True)
    f.update_yaxes(range=[0, 1.08], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    f.update_xaxes(tickfont=dict(size=15))
    return f


def lead_time_hist(details: dict):
    f = go.Figure()
    for i, (lab, m) in enumerate(details.items()):
        v = m["lead_days"].dropna()
        f.add_histogram(x=v, name=f"{lab} (median {v.median():.0f} d)", marker_color=CAT[[0, 1, 2][i % 3]], opacity=0.7,
                        xbins=dict(start=0, end=95, size=5))
    f.update_layout(barmode="overlay", title="Lead time: days between the alert and the departure", xaxis_title="days",
                    yaxis_title="leavers caught", height=400)
    return f


# ------------------------------------------------------------------ XAI client map (SHAP vs LIME)
OK_COLOR, FLAG_COLOR = CAT[3], "#C2272D"   # green / red as in the teacher's app; red kept apart from the "leaves" pink


def xai_map(pts: pd.DataFrame, threshold: float):
    """PCA map of clients: colour = SHAP/LIME agreement, symbol = the model's decision."""
    f = go.Figure()
    for status, color in [("OK", OK_COLOR), ("FLAGGED", FLAG_COLOR)]:
        for decision, symbol in [("no alert (stays)", "circle"), ("alert (leaves)", "diamond")]:
            g = pts[(pts["status"] == status) & (pts["model says"] == decision)]
            if g.empty:
                continue
            f.add_scatter(
                x=g["PC1"], y=g["PC2"], mode="markers", name=f"{status} · {decision}",
                customdata=np.array([[int(i), str(c), float(r), float(k), "left" if l == 1 else "stayed"]
                                     for i, c, r, k, l in zip(g.index, g["client_id"], g["risk"], g["corr"], g["left"])],
                                    dtype=object),
                marker=dict(color=color, symbol=symbol, size=12 if status == "FLAGGED" else 10, opacity=0.85,
                            line=dict(width=1.2, color=INK if status == "FLAGGED" else "white")),
                hovertemplate=("client %{customdata[1]}<br>risk %{customdata[2]:.0%}<br>SHAP–LIME correlation "
                               "%{customdata[3]:.2f}<br>actually %{customdata[4]}<extra>" + status + "</extra>"))
    f.update_layout(title=f"Clients on 2 PCA axes · diamond = alert (risk ≥ {threshold:.2f}) · red = SHAP and LIME disagree",
                    xaxis_title="PC1", yaxis_title="PC2", height=560,
                    legend=dict(y=1.0, yanchor="bottom"), margin=dict(t=120))
    return f


def selected_rows(event, pts: pd.DataFrame) -> list:
    """Row labels of the points selected on xai_map (click, box or lasso), in a stable order."""
    try:
        points = event.selection.points if event else []
    except AttributeError:
        points = (event or {}).get("selection", {}).get("points", [])
    out = []
    for p in points:
        cd = p.get("customdata")
        if cd is not None and len(cd):
            lab = int(float(cd[0]))
        else:
            hit = pts[np.isclose(pts["PC1"], p.get("x", np.nan)) & np.isclose(pts["PC2"], p.get("y", np.nan))]
            lab = hit.index[0] if len(hit) else None
        if lab is not None and lab in pts.index and lab not in out:
            out.append(lab)
    return out


def contrib_bars(s: pd.Series, title: str, top: int = 12):
    """Normalised contributions of one client (SHAP or LIME), largest |value| on top."""
    s = s.reindex(s.abs().sort_values(ascending=False).index[:top])[::-1]
    f = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=[LEAVES if v > 0 else STAYS for v in s.values],
                         text=[f"{v:+.2f}" for v in s.values], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    f.add_vline(x=0, line_color=MUTED)
    f.update_layout(title=title, height=460, xaxis=dict(range=[-1.3, 1.3], title="normalised contribution"), showlegend=False)
    return f
