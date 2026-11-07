"""Shared look for both Streamlit apps: CSS on top of .streamlit/config.toml, KPI cards, logo, icon credits."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

CSS = """
<style>
/* bigger, clearer text where it matters */
h1 {font-size: 2.35rem !important; color: #142850; letter-spacing: -0.01em;}
h2 {font-size: 1.75rem !important; color: #290044;}
h3 {font-size: 1.35rem !important; color: #500071;}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-size: 1.02rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding: 0.18rem 0.3rem; border-radius: 0.4rem;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(255, 211, 139, 0.14);}
button[role="tab"] p {font-size: 1.08rem !important; font-weight: 650;}
[data-testid="stMetricLabel"] p {font-size: 1.0rem !important; font-weight: 600; color: #500071;}
[data-testid="stMetric"] {background: #FBF8FD; border: 1px solid #E1D5EA; border-left: 6px solid #8A2A78;
                          border-radius: 0.7rem; padding: 0.7rem 1rem;}
[data-testid="stExpander"] summary p {font-weight: 650;}
/* hero banner and cards */
.hero {background: linear-gradient(120deg, #290044 0%, #500071 55%, #8A2A78 100%); color: #FFF7E6;
       padding: 1.3rem 1.6rem; border-radius: 1rem; margin-bottom: 1rem;}
.hero h1 {color: #FFD38B !important; margin: 0 0 .3rem 0;}
.hero p {font-size: 1.15rem; margin: 0; color: #F3EBC8;}
.kcard {background: #FFFFFF; border: 1px solid #E1D5EA; border-radius: 0.9rem; padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 8px rgba(41,0,68,0.06); min-height: 9.2rem; margin-bottom: .6rem;}
.kcard .lab {font-size: 0.95rem; color: #6B6F7B; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
.kcard .val {font-size: 2.2rem; font-weight: 800; color: #142850; line-height: 1.15;}
.kcard .sub {font-size: 0.95rem; color: #500071;}
.pill {display: inline-block; padding: .15rem .7rem; border-radius: 999px; font-weight: 750; font-size: 1rem;}
.pass {background: #DDF3EA; color: #1E6E4C;} .fail {background: #FFE1E8; color: #B0123F;}
.src {border-left: 6px solid #1A8FB0; background: #F4FAFC; padding: .7rem 1rem; border-radius: .6rem; margin-bottom: .6rem;}
.src.ext {border-left-color: #7469B6; background: #F6F4FB;} .src.opt {border-left-color: #B88A10; background: #FBF8EE;}
.src b {font-size: 1.08rem;}
/* the research cycle (home page) */
.cyc {display: grid; grid-template-columns: repeat(6, 1fr); column-gap: 1.6rem; row-gap: .25rem; margin: .4rem 0 1rem 0;}
.cyc .n {position: relative; border: 2px solid #8A2A78; background: #F5EFF8; border-radius: .7rem; padding: .55rem .4rem;
         text-align: center; font-weight: 650; color: #252A34; font-size: 1.0rem; line-height: 1.25;}
.cyc .n small {display: block; font-weight: 500; color: #500071; font-size: .88rem;}
.cyc .n.data {border-color: #1A8FB0; background: #E6F7F9;} .cyc .n.ext {border-color: #7469B6; background: #F1EEFA;}
.cyc .n.verdict {border-color: #FF2E63; background: #FFE1E8; border-width: 3px;}
.cyc .n.r::after {content: "→"; position: absolute; right: -1.35rem; top: 50%; transform: translateY(-50%); color: #7469B6; font-size: 1.3rem;}
.cyc .n.l::before {content: "←"; position: absolute; left: -1.35rem; top: 50%; transform: translateY(-50%); color: #7469B6; font-size: 1.3rem;}
.cyc .v {text-align: center; font-size: .92rem; font-weight: 650; min-height: 1.6rem;}
.cyc .v.fail {color: #C81E4C;} .cyc .v.loop {color: #16759A;} .cyc .v.up {color: #7469B6; font-size: 1.25rem;}
</style>
"""


def setup(logo: str = "churn.png"):
    st.markdown(CSS, unsafe_allow_html=True)
    p = ASSETS / logo
    if p.exists():
        st.logo(str(p), size="large")


def hero(title: str, subtitle: str, icon: str | None = None):
    c1, c2 = st.columns([1, 9]) if icon and (ASSETS / icon).exists() else (None, None)
    html = f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>'
    if c1 is not None:
        c1.image(str(ASSETS / icon), width=96)
        c2.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def cards(items: list[tuple[str, str, str]]):
    """items = [(label, value, sub-text)] -> a row of KPI cards."""
    cols = st.columns(len(items))
    for c, (lab, val, sub) in zip(cols, items):
        c.markdown(f'<div class="kcard"><div class="lab">{lab}</div><div class="val">{val}</div>'
                   f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def cycle_diagram():
    """The teacher's research board as two rows: 1-6 left to right, 7-12 right to left, back to 1 on FAIL."""
    top = [("12 Verdict", "", "verdict"), ("11 RAG", "", ""), ("10 Streaming", "RTAP", "data"),
           ("9 Raw external", "data", "ext"), ("8 XAI", "SHAP + LIME", ""), ("7 Decision", "boundaries", "")]
    bottom = [("1 Problem", "+ KPIs", ""), ("2 Data store", "DWH + query", "data"), ("3 Metrics", "KPIs", ""),
              ("4 Contract", "matrix", ""), ("5 Features", "", ""), ("6 ML / DL", "black box", "")]
    cell = lambda t, sub, cls, arrow: f'<div class="n {cls} {arrow}">{t}<small>{sub or "&nbsp;"}</small></div>'  # noqa: E731
    html = '<div class="cyc">'
    html += "".join(cell(t, sub, c, "l" if i > 0 else "") for i, (t, sub, c) in enumerate(top))
    html += ('<div class="v fail">↓ FAIL: back to 1</div><div class="v"></div><div class="v"></div><div class="v"></div>'
             '<div class="v loop">⇣ inner loop</div><div class="v up">↑</div>')
    html += "".join(cell(t, sub, c, "r" if i < 5 else "") for i, (t, sub, c) in enumerate(bottom))
    st.markdown(html + "</div>", unsafe_allow_html=True)


def verdict_pill(ok: bool) -> str:
    return f'<span class="pill {"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span>'


def credits():
    """Short clickable icon credit in the sidebar (Flaticon's free licence asks for attribution)."""
    if any((ASSETS / f).exists() for f in ("churn.png", "emergency.png")):
        st.sidebar.caption("Icons: [iconsitee](https://www.flaticon.com/free-icon/churn-rate_18419481) · "
                           "[ViconsDesign](https://www.flaticon.com/free-icons/emergency) — Flaticon")


def chart(fig, key=None, select: bool = False):
    """Draw a Plotly figure with the project template. select=True lets the user click or
    box/lasso-select points and returns the selection (the app reruns on every selection)."""
    from ds import viz  # noqa: F401  (registers the "churn" template)
    fig.update_layout(template="churn", plot_bgcolor="white", paper_bgcolor="white")  # explicit: another library may reset plotly's default template
    legend = fig.layout.showlegend is not False and sum(1 for t in fig.data if t.showlegend is not False and t.name) > 1
    if not legend and fig.layout.margin.t is None:
        fig.update_layout(margin=dict(t=60))  # no legend row above the plot: no empty band under the title
    opts = dict(width="stretch", key=key, theme=None, config={"displaylogo": False})  # theme=None: keep our template
    if select:
        return st.plotly_chart(fig, on_select="rerun", selection_mode=("points", "box", "lasso"), **opts)
    st.plotly_chart(fig, **opts)
