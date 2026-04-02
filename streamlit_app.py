import io
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------- App config ----------------
TITLE = "Weekly Risk On / Risk Off Meter"
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.caption("Weekly regime dashboard for positioning, gross, and hedge posture. Data: Yahoo Finance + FRED.")

# ---------------- Style helpers ----------------
GRID = "#e8e8e8"
TEXT = "#222222"


def card_box(inner_html: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px; background:#fafafa; color:{TEXT}; font-size:14px; line-height:1.35;">
          {inner_html}
        </div>
        """.strip(),
        unsafe_allow_html=True,
    )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    lookback_years = st.selectbox("History window", [3, 5, 10], index=1)
    smooth_weeks = st.slider("Smoothing (weeks)", 1, 6, 2)
    st.markdown("---")
    st.markdown(
        """
        **Design**
        - Core tape score from equity, credit, and cross-asset ratios
        - Macro / liquidity sleeve from free FRED data
        - Stress penalty based on your existing stress-composite logic family
        - Weekly sampling to reduce noise
        """
    )

# ---------------- Constants ----------------
START = (datetime.today() - timedelta(days=365 * lookback_years + 120)).strftime("%Y-%m-%d")

SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
YF_TICKERS = [
    "SPY", "RSP", "XLY", "XLP", "IWM", "QQQ",
    "HYG", "LQD", "IEF", "SPHB", "SPLV",
    "SMH", "FXY", "DBC", "GLD",
    "^VIX", "^VIX3M", "^TNX", "^GSPC",
    "HG=F"
] + SECTORS

FRED_SERIES = {
    "WALCL": "WALCL",
    "RRP": "RRPONTSYD",
    "TGA": "WTREGEN",
    "HY_OAS": "BAMLH0A0HYM2",
    "T10Y3M": "T10Y3M",
    "CP3M": "DCPF3M",
    "TBILL3M": "DTB3",
}

FACTOR_TICKERS = ["MTUM", "QUAL", "VLUE", "USMV", "SIZE"]


# ---------------- Data loaders ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_yahoo(tickers, start):
    try:
        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    out = {}

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        for t in tickers:
            if t in lvl0:
                df_t = raw[t]
                if "Close" in df_t.columns:
                    s = pd.to_numeric(df_t["Close"], errors="coerce").dropna()
                    if not s.empty:
                        out[t] = s
    else:
        if "Close" in raw.columns and len(tickers) == 1:
            out[tickers[0]] = pd.to_numeric(raw["Close"], errors="coerce").dropna()

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(out).sort_index().dropna(how="all")


@st.cache_data(show_spinner=False, ttl=3600)
def load_fred(series_map, start_date):
    idx = pd.date_range(start=pd.to_datetime(start_date), end=pd.Timestamp.today(), freq="B")
    out = pd.DataFrame(index=idx)

    for label, fred_id in series_map.items():
        out[label] = np.nan
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"

        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=20) as resp:
                raw = resp.read()

            df = pd.read_csv(io.BytesIO(raw))
            df.columns = ["Date", label]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df[label] = pd.to_numeric(df[label], errors="coerce")

            s = df.set_index("Date")[label].sort_index()
            out[label] = s.reindex(idx)

        except (HTTPError, URLError, TimeoutError, ValueError):
            pass

    return out


def weekly_last(df):
    out = df.resample("W-FRI").last()
    if isinstance(df, pd.Series):
        return out.dropna()
    return out


# ---------------- Helpers ----------------
def get_series(df: pd.DataFrame, col: str, index: pd.Index) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").reindex(index)
    return pd.Series(np.nan, index=index, name=col)


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.align(b, join="outer")
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def zscore_last_window(s: pd.Series, window=52) -> pd.Series:
    mu = s.rolling(window, min_periods=max(8, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(8, window // 4)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def score_from_z(s: pd.Series, window=52, invert=False) -> pd.Series:
    z = zscore_last_window(s, window=window).clip(-2.5, 2.5)
    score = (50 + 20 * z).clip(0, 100)
    if invert:
        score = 100 - score
    return score.clip(0, 100)


def trend_score(s: pd.Series, fast=10, slow=40) -> pd.Series:
    fast_ma = s.rolling(fast, min_periods=1).mean()
    slow_ma = s.rolling(slow, min_periods=1).mean()

    score = pd.Series(50.0, index=s.index)
    score += np.where(s > fast_ma, 25, -25)
    score += np.where(fast_ma > slow_ma, 25, -25)
    return score.clip(0, 100)


def roc_score(s: pd.Series, periods=4, window=52, invert=False) -> pd.Series:
    roc = s.pct_change(periods)
    return score_from_z(roc, window=window, invert=invert)


def combo_score(s: pd.Series, invert=False) -> pd.Series:
    tr = trend_score(s)
    rc = roc_score(s, invert=invert)
    out = 0.6 * tr + 0.4 * rc
    if invert:
        out = 100 - out
    return out.clip(0, 100)


def realized_vol(px: pd.Series, window=21, ann=252) -> pd.Series:
    r = px.pct_change()
    return r.rolling(window).std() * np.sqrt(ann)


def drawdown(px: pd.Series) -> pd.Series:
    peak = px.cummax()
    return (px / peak) - 1.0


def label_regime(x: float) -> str:
    if x >= 70:
        return "Risk On"
    if x >= 55:
        return "Mild Risk On"
    if x >= 45:
        return "Neutral"
    if x >= 30:
        return "Mild Risk Off"
    return "Risk Off"


# ---------------- Load data ----------------
px_d = load_yahoo(YF_TICKERS, START)
fred_d = load_fred(FRED_SERIES, START)

if px_d.empty:
    st.error("Failed to download Yahoo Finance data.")
    st.stop()

px_w = weekly_last(px_d)
master_index = px_w.index

fred_w = weekly_last(fred_d).reindex(master_index).ffill()

if smooth_weeks > 1:
    px_w = px_w.rolling(smooth_weeks, min_periods=1).mean()
    fred_w = fred_w.rolling(smooth_weeks, min_periods=1).mean()

# ---------------- Series extraction ----------------
spy = get_series(px_w, "SPY", master_index)
rsp = get_series(px_w, "RSP", master_index)
xly = get_series(px_w, "XLY", master_index)
xlp = get_series(px_w, "XLP", master_index)
iwm = get_series(px_w, "IWM", master_index)
hyg = get_series(px_w, "HYG", master_index)
lqd = get_series(px_w, "LQD", master_index)
ief = get_series(px_w, "IEF", master_index)
sphb = get_series(px_w, "SPHB", master_index)
splv = get_series(px_w, "SPLV", master_index)
smh = get_series(px_w, "SMH", master_index)
fxy = get_series(px_w, "FXY", master_index)
dbc = get_series(px_w, "DBC", master_index)
gld = get_series(px_w, "GLD", master_index)
vix = get_series(px_w, "^VIX", master_index).rename("VIX")
vix3m = get_series(px_w, "^VIX3M", master_index)
gspc = get_series(px_w, "^GSPC", master_index)
hg = get_series(px_w, "HG=F", master_index)

# ---------------- Derived series ----------------
rsp_spy = safe_ratio(rsp, spy).rename("RSP/SPY")
xly_xlp = safe_ratio(xly, xlp).rename("XLY/XLP")
iwm_spy = safe_ratio(iwm, spy).rename("IWM/SPY")
sphb_splv = safe_ratio(sphb, splv).rename("SPHB/SPLV")

hyg_lqd = safe_ratio(hyg, lqd).rename("HYG/LQD")
hyg_ief = safe_ratio(hyg, ief).rename("HYG/IEF")
vix_vix3m = safe_ratio(vix, vix3m).rename("VIX/VIX3M")

smh_fxy = safe_ratio(smh, fxy).rename("SMH/FXY")
dbc_gld = safe_ratio(dbc, gld).rename("DBC/GLD")
copper_gold = safe_ratio(hg, gld).rename("Copper/GoldProxy")

fred_w["NET_LIQUIDITY"] = (
    get_series(fred_w, "WALCL", master_index)
    - get_series(fred_w, "RRP", master_index)
    - get_series(fred_w, "TGA", master_index)
)
netliq = get_series(fred_w, "NET_LIQUIDITY", master_index).rename("Net Liquidity")
curve = get_series(fred_w, "T10Y3M", master_index).rename("10Y-3M")
funding = (
    get_series(fred_w, "CP3M", master_index)
    - get_series(fred_w, "TBILL3M", master_index)
).rename("CP-TBill")

# Sector breadth
sector_px = pd.DataFrame(index=master_index)
for s in SECTORS:
    sector_px[s] = get_series(px_w, s, master_index)

sector_fast = sector_px.rolling(10, min_periods=1).mean()
sector_slow = sector_px.rolling(40, min_periods=1).mean()

pct_above_fast = ((sector_px > sector_fast).sum(axis=1) / len(SECTORS) * 100.0).rename("% Above 10W")
pct_fast_above_slow = ((sector_fast > sector_slow).sum(axis=1) / len(SECTORS) * 100.0).rename("% 10W > 40W")
sector_breadth = (0.5 * pct_above_fast + 0.5 * pct_fast_above_slow).rename("Sector Breadth")

# Factor sleeve
factor_px_d = load_yahoo(FACTOR_TICKERS, START)
factor_regime = pd.Series(50.0, index=master_index, name="Factor Regime")

if not factor_px_d.empty:
    factor_w = weekly_last(factor_px_d).reindex(master_index).ffill()
    mom_rows = []

    for f in factor_w.columns:
        s = pd.to_numeric(factor_w[f], errors="coerce")
        short = s.pct_change(4)
        trend = pd.Series(np.where(s > s.rolling(10, min_periods=1).mean(), 1, -1), index=s.index)
        inflection = np.sign(short.diff())

        frame = pd.DataFrame(
            {
                "Short": short,
                "Trend": trend,
                "Inflection": inflection,
            }
        )

        score = (
            0.4 * frame["Short"].fillna(0)
            + 0.3 * frame["Inflection"].fillna(0)
            + 0.3 * frame["Trend"].fillna(0)
        )
        mom_rows.append(score.rename(f))

    if mom_rows:
        factor_scores = pd.concat(mom_rows, axis=1)
        factor_raw = factor_scores.mean(axis=1)
        factor_regime = (50 + 40 * zscore_last_window(factor_raw, window=52).clip(-1.25, 1.25)).clip(0, 100)
        factor_regime = factor_regime.reindex(master_index).ffill().fillna(50)

# ---------------- Bucket scores ----------------
signals = pd.DataFrame(index=master_index)

signals["RSP/SPY"] = combo_score(rsp_spy).reindex(master_index).fillna(50)
signals["XLY/XLP"] = combo_score(xly_xlp).reindex(master_index).fillna(50)
signals["IWM/SPY"] = combo_score(iwm_spy).reindex(master_index).fillna(50)
signals["SPHB/SPLV"] = combo_score(sphb_splv).reindex(master_index).fillna(50)
signals["Sector Breadth"] = sector_breadth.reindex(master_index).fillna(50).clip(0, 100)

signals["HYG/LQD"] = combo_score(hyg_lqd).reindex(master_index).fillna(50)
signals["HYG/IEF"] = combo_score(hyg_ief).reindex(master_index).fillna(50)
signals["VIX"] = score_from_z(vix, invert=True).reindex(master_index).fillna(50)
signals["VIX/VIX3M"] = score_from_z(vix_vix3m, invert=True).reindex(master_index).fillna(50)

signals["SMH/FXY"] = combo_score(smh_fxy).reindex(master_index).fillna(50)
signals["DBC/GLD"] = combo_score(dbc_gld).reindex(master_index).fillna(50)
signals["Copper/Gold"] = combo_score(copper_gold).reindex(master_index).fillna(50)

signals["Net Liquidity"] = roc_score(netliq, periods=4, window=52).reindex(master_index).fillna(50)
signals["10Y-3M"] = combo_score(curve).reindex(master_index).fillna(50)
signals["Funding"] = score_from_z(funding, invert=True).reindex(master_index).fillna(50)
signals["Factor Regime"] = factor_regime.reindex(master_index).fillna(50).clip(0, 100)

eq_cols = ["RSP/SPY", "XLY/XLP", "IWM/SPY", "SPHB/SPLV", "Sector Breadth", "Factor Regime"]
credit_cols = ["HYG/LQD", "HYG/IEF", "VIX", "VIX/VIX3M"]
xasset_cols = ["SMH/FXY", "DBC/GLD", "Copper/Gold"]
macro_cols = ["Net Liquidity", "10Y-3M", "Funding"]

equity_bucket = signals[eq_cols].mean(axis=1).rename("Equity Leadership")
credit_bucket = signals[credit_cols].mean(axis=1).rename("Credit / Vol")
xasset_bucket = signals[xasset_cols].mean(axis=1).rename("Cross Asset")
macro_bucket = signals[macro_cols].mean(axis=1).rename("Liquidity / Macro")

base_score = (
    0.35 * equity_bucket
    + 0.30 * credit_bucket
    + 0.20 * xasset_bucket
    + 0.15 * macro_bucket
).rename("Base Score")

# ---------------- Stress penalty sleeve ----------------
spy_rsp_stress = safe_ratio(spy, rsp).rename("SPY/RSP")

spx_daily = pd.to_numeric(px_d["^GSPC"], errors="coerce").dropna() if "^GSPC" in px_d.columns else pd.Series(dtype=float)
rv21_daily = realized_vol(spx_daily, 21) if not spx_daily.empty else pd.Series(dtype=float)
rv21_w = weekly_last(rv21_daily).reindex(master_index).ffill() if not rv21_daily.empty else pd.Series(np.nan, index=master_index)

dd_w = (-drawdown(gspc) * 100.0).rename("Drawdown")

stress_components = pd.DataFrame(index=master_index)
stress_components["VIX"] = score_from_z(vix, invert=False).reindex(master_index).fillna(50)
stress_components["HY_OAS"] = score_from_z(get_series(fred_w, "HY_OAS", master_index), invert=False).reindex(master_index).fillna(50)
stress_components["HYG_LQD"] = score_from_z(hyg_lqd, invert=True).reindex(master_index).fillna(50)
stress_components["Curve"] = score_from_z(curve, invert=True).reindex(master_index).fillna(50)
stress_components["Funding"] = score_from_z(funding, invert=False).reindex(master_index).fillna(50)
stress_components["Drawdown"] = score_from_z(dd_w, invert=False).reindex(master_index).fillna(50)
stress_components["RV21"] = score_from_z(rv21_w, invert=False).reindex(master_index).fillna(50)
stress_components["Breadth"] = score_from_z(spy_rsp_stress, invert=False).reindex(master_index).fillna(50)

stress_weights = {
    "VIX": 0.20,
    "HY_OAS": 0.15,
    "HYG_LQD": 0.15,
    "Curve": 0.10,
    "Funding": 0.10,
    "Drawdown": 0.15,
    "RV21": 0.10,
    "Breadth": 0.05,
}

stress_score = sum(stress_weights[k] * stress_components[k] for k in stress_weights).rename("Stress Score")
stress_penalty = ((stress_score - 50).clip(lower=0) * 0.45).rename("Stress Penalty")
risk_score = (base_score - stress_penalty).clip(0, 100).rename("Risk Score")

# ---------------- Diagnostics ----------------
signal_now = signals.iloc[-1].sort_values(ascending=False).to_frame("Score")
signal_now["1W Δ"] = signals.diff(1).iloc[-1].reindex(signal_now.index)
signal_now["4W Δ"] = signals.diff(4).iloc[-1].reindex(signal_now.index)

breadth = ((signals > 50).sum(axis=1) / signals.shape[1] * 100.0).rename("Signal Breadth")
confidence = (100.0 - signals.std(axis=1).clip(0, 50) * 2.0).clip(0, 100).rename("Confidence")

curr = float(risk_score.iloc[-1])
prev_1w = float(risk_score.iloc[-2]) if len(risk_score) > 1 else curr
delta_1w = curr - prev_1w
delta_4w = float(risk_score.iloc[-1] - risk_score.iloc[-5]) if len(risk_score) > 4 else np.nan
regime = label_regime(curr)

# ---------------- Summary box ----------------
summary = (
    f"<b>{regime}</b><br>"
    f"Headline score: <b>{curr:.1f}</b> | "
    f"1W change: <b>{delta_1w:+.1f}</b> | "
    f"Signal breadth: <b>{breadth.iloc[-1]:.0f}%</b> | "
    f"Confidence: <b>{confidence.iloc[-1]:.0f}</b><br><br>"
    f"Bucket read: Equity {equity_bucket.iloc[-1]:.0f}, Credit/Vol {credit_bucket.iloc[-1]:.0f}, "
    f"Cross-Asset {xasset_bucket.iloc[-1]:.0f}, Liquidity/Macro {macro_bucket.iloc[-1]:.0f}, "
    f"Stress {stress_score.iloc[-1]:.0f}."
)
card_box(summary)

# ---------------- Layout ----------------
top_left, top_right = st.columns([1.15, 1.0])

with top_left:
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=curr,
            delta={"reference": prev_1w},
            title={"text": regime},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.30},
                "steps": [
                    {"range": [0, 30], "color": "#7f1d1d"},
                    {"range": [30, 45], "color": "#b45309"},
                    {"range": [45, 55], "color": "#737373"},
                    {"range": [55, 70], "color": "#16a34a"},
                    {"range": [70, 100], "color": "#166534"},
                ],
            },
        )
    )
    fig_g.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with top_right:
    metric_cols = st.columns(4)
    metric_cols[0].metric("1W Change", f"{delta_1w:+.1f}")
    metric_cols[1].metric("4W Change", "NA" if pd.isna(delta_4w) else f"{delta_4w:+.1f}")
    metric_cols[2].metric("Breadth", f"{breadth.iloc[-1]:.0f}%")
    metric_cols[3].metric("Confidence", f"{confidence.iloc[-1]:.0f}")

    fig_bars = go.Figure()
    for label, value, color in [
        ("Equity Leadership", equity_bucket.iloc[-1], "#A8DADC"),
        ("Credit / Vol", credit_bucket.iloc[-1], "#F4A261"),
        ("Cross Asset", xasset_bucket.iloc[-1], "#90BE6D"),
        ("Liquidity / Macro", macro_bucket.iloc[-1], "#CDB4DB"),
        ("Stress Score", stress_score.iloc[-1], "#E07A73"),
    ]:
        fig_bars.add_trace(
            go.Bar(
                x=[value],
                y=[label],
                orientation="h",
                marker_color=color,
                text=[f"{value:.0f}"],
                textposition="outside",
                showlegend=False,
            )
        )

    fig_bars.update_layout(
        height=320,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(range=[0, 100], title="Score"),
        yaxis=dict(title=""),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

mid_left, mid_right = st.columns([1.1, 0.9])

with mid_left:
    history = risk_score.tail(52)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Risk Score"))
    fig_hist.add_hrect(y0=70, y1=100, fillcolor="green", opacity=0.06, line_width=0)
    fig_hist.add_hrect(y0=55, y1=70, fillcolor="lightgreen", opacity=0.06, line_width=0)
    fig_hist.add_hrect(y0=45, y1=55, fillcolor="gray", opacity=0.06, line_width=0)
    fig_hist.add_hrect(y0=30, y1=45, fillcolor="orange", opacity=0.06, line_width=0)
    fig_hist.add_hrect(y0=0, y1=30, fillcolor="red", opacity=0.06, line_width=0)
    fig_hist.update_layout(
        title="52-Week Regime History",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(range=[0, 100], title="Score"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with mid_right:
    heat_df = signal_now.copy().sort_values("Score", ascending=False)
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=[heat_df["Score"].values],
            x=heat_df.index.tolist(),
            y=["Signals"],
            zmin=0,
            zmax=100,
            colorscale="RdYlGn",
            text=[[f"{v:.0f}" for v in heat_df["Score"].values]],
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig_heat.update_layout(
        title="Signal Heatmap",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("Signal Snapshot")
display_df = signal_now.copy()
display_df["Score"] = display_df["Score"].map(lambda x: f"{x:.1f}")
display_df["1W Δ"] = display_df["1W Δ"].map(lambda x: f"{x:+.1f}")
display_df["4W Δ"] = display_df["4W Δ"].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "NA")
st.dataframe(display_df, use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
