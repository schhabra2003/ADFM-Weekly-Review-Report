# pages/18_Weekly_Risk_On_Off_Meter.py
# ADFM - Weekly Risk On / Risk Off Meter
# Built to sit on top of existing suite logic:
# - Ratio charts for tape
# - Stress composite as penalty
# - Cross-asset confirmation
# - Sector / factor participation

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
    "WALCL": "WALCL",           # Fed balance sheet
    "RRP": "RRPONTSYD",         # Reverse repo
    "TGA": "WTREGEN",           # Treasury General Account
    "HY_OAS": "BAMLH0A0HYM2",   # HY OAS
    "T10Y3M": "T10Y3M",         # 10Y minus 3M
    "CP3M": "DCPF3M",           # 3M AA commercial paper
    "TBILL3M": "DTB3",          # 3M T-bill
}

# ---------------- Data loaders ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_yahoo(tickers, start):
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        for t in tickers:
            if t in lvl0:
                df_t = raw[t]
                if "Close" in df_t.columns:
                    s = df_t["Close"].dropna()
                    if not s.empty:
                        out[t] = s
    else:
        if "Close" in raw.columns and len(tickers) == 1:
            out[tickers[0]] = raw["Close"].dropna()

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(out).sort_index().dropna(how="all")

@st.cache_data(show_spinner=False, ttl=3600)
def load_fred(series_map):
    out = {}
    for label, fred_id in series_map.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        df = pd.read_csv(url)
        df.columns = ["Date", label]
        df["Date"] = pd.to_datetime(df["Date"])
        df[label] = pd.to_numeric(df[label], errors="coerce")
        out[label] = df.set_index("Date")[label]
    return pd.DataFrame(out).sort_index()

def weekly_last(df):
    if isinstance(df, pd.Series):
        return df.resample("W-FRI").last().dropna()
    return df.resample("W-FRI").last().dropna(how="all")

# ---------------- Signal helpers ----------------
def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.align(b, join="inner")
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def zscore_last_window(s: pd.Series, window=52) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z

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
fred_d = load_fred(FRED_SERIES)

if px_d.empty:
    st.error("Failed to download Yahoo Finance data.")
    st.stop()

px_w = weekly_last(px_d)
fred_w = weekly_last(fred_d)

# Align everything to common start
common_start = max(px_w.index.min(), fred_w.index.min())
px_w = px_w.loc[common_start:].copy()
fred_w = fred_w.loc[common_start:].copy()

# Smooth selected inputs a bit for weekly reporting
if smooth_weeks > 1:
    px_w = px_w.rolling(smooth_weeks, min_periods=1).mean()
    fred_w = fred_w.rolling(smooth_weeks, min_periods=1).mean()

# ---------------- Derived series ----------------
# Equity / ratio sleeve
rsp_spy = safe_ratio(px_w["RSP"], px_w["SPY"]).rename("RSP/SPY")
xly_xlp = safe_ratio(px_w["XLY"], px_w["XLP"]).rename("XLY/XLP")
iwm_spy = safe_ratio(px_w["IWM"], px_w["SPY"]).rename("IWM/SPY")
sphb_splv = safe_ratio(px_w["SPHB"], px_w["SPLV"]).rename("SPHB/SPLV")

# Credit / vol sleeve
hyg_lqd = safe_ratio(px_w["HYG"], px_w["LQD"]).rename("HYG/LQD")
hyg_ief = safe_ratio(px_w["HYG"], px_w["IEF"]).rename("HYG/IEF")
vix = px_w["^VIX"].dropna().rename("VIX")
vix_vix3m = safe_ratio(px_w["^VIX"], px_w["^VIX3M"]).rename("VIX/VIX3M")

# Cross-asset sleeve
smh_fxy = safe_ratio(px_w["SMH"], px_w["FXY"]).rename("SMH/FXY")
dbc_gld = safe_ratio(px_w["DBC"], px_w["GLD"]).rename("DBC/GLD")
copper_gold = safe_ratio(px_w["HG=F"], px_w["GLD"]).rename("Copper/GoldProxy")

# Macro / liquidity sleeve
fred_w["NET_LIQUIDITY"] = fred_w["WALCL"] - fred_w["RRP"] - fred_w["TGA"]
netliq = fred_w["NET_LIQUIDITY"].rename("Net Liquidity")
curve = fred_w["T10Y3M"].rename("10Y-3M")
funding = (fred_w["CP3M"] - fred_w["TBILL3M"]).rename("CP-TBill")

# Sector breadth
sector_px = px_w[SECTORS].dropna(how="all")
sector_fast = sector_px.rolling(10, min_periods=1).mean()
sector_slow = sector_px.rolling(40, min_periods=1).mean()

pct_above_fast = ((sector_px > sector_fast).sum(axis=1) / len(SECTORS) * 100.0).rename("% Above 10W")
pct_fast_above_slow = ((sector_fast > sector_slow).sum(axis=1) / len(SECTORS) * 100.0).rename("% 10W > 40W")
sector_breadth = (0.5 * pct_above_fast + 0.5 * pct_fast_above_slow).rename("Sector Breadth")

# Factor sleeve proxies
# Keep simple and free-data friendly
FACTOR_TICKERS = ["MTUM", "QUAL", "VLUE", "USMV", "SIZE"]
factor_px_d = load_yahoo(FACTOR_TICKERS, START)
factor_w = weekly_last(factor_px_d) if not factor_px_d.empty else pd.DataFrame()

factor_regime = pd.Series(index=px_w.index, dtype=float)
if not factor_w.empty:
    factor_w = factor_w.reindex(px_w.index).ffill()
    mom_rows = []
    for f in factor_w.columns:
        s = factor_w[f].dropna()
        short = s.pct_change(4)
        long = s.pct_change(13)
        trend = np.where(s > s.rolling(10, min_periods=1).mean(), 1, -1)
        inflection = np.sign(short.diff())
        frame = pd.DataFrame(
            {
                "Short": short,
                "Long": long,
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
    factor_scores = pd.concat(mom_rows, axis=1)
    factor_raw = factor_scores.mean(axis=1)
    factor_regime = (50 + 40 * zscore_last_window(factor_raw, window=52).clip(-1.25, 1.25)).clip(0, 100)
    factor_regime = factor_regime.reindex(px_w.index).ffill()
else:
    factor_regime = pd.Series(50.0, index=px_w.index)

# ---------------- Bucket scores ----------------
signals = {}

signals["RSP/SPY"] = combo_score(rsp_spy)
signals["XLY/XLP"] = combo_score(xly_xlp)
signals["IWM/SPY"] = combo_score(iwm_spy)
signals["SPHB/SPLV"] = combo_score(sphb_splv)
signals["Sector Breadth"] = sector_breadth.clip(0, 100)

signals["HYG/LQD"] = combo_score(hyg_lqd)
signals["HYG/IEF"] = combo_score(hyg_ief)
signals["VIX"] = score_from_z(vix, invert=True)
signals["VIX/VIX3M"] = score_from_z(vix_vix3m, invert=True)

signals["SMH/FXY"] = combo_score(smh_fxy)
signals["DBC/GLD"] = combo_score(dbc_gld)
signals["Copper/Gold"] = combo_score(copper_gold)

signals["Net Liquidity"] = roc_score(netliq, periods=4, window=52)
signals["10Y-3M"] = combo_score(curve)
signals["Funding"] = score_from_z(funding, invert=True)
signals["Factor Regime"] = factor_regime.clip(0, 100)

eq_cols = ["RSP/SPY", "XLY/XLP", "IWM/SPY", "SPHB/SPLV", "Sector Breadth", "Factor Regime"]
credit_cols = ["HYG/LQD", "HYG/IEF", "VIX", "VIX/VIX3M"]
xasset_cols = ["SMH/FXY", "DBC/GLD", "Copper/Gold"]
macro_cols = ["Net Liquidity", "10Y-3M", "Funding"]

sig_df = pd.concat(signals, axis=1).dropna(how="all")
sig_df = sig_df.reindex(px_w.index).ffill()

equity_bucket = sig_df[eq_cols].mean(axis=1).rename("Equity Leadership")
credit_bucket = sig_df[credit_cols].mean(axis=1).rename("Credit / Vol")
xasset_bucket = sig_df[xasset_cols].mean(axis=1).rename("Cross Asset")
macro_bucket = sig_df[macro_cols].mean(axis=1).rename("Liquidity / Macro")

base_score = (
    0.35 * equity_bucket
    + 0.30 * credit_bucket
    + 0.20 * xasset_bucket
    + 0.15 * macro_bucket
).rename("Base Score")

# ---------------- Stress penalty sleeve ----------------
# Mirrors your existing stress-composite logic family
spx_w = px_w["^GSPC"].dropna()
spy_rsp_stress = safe_ratio(px_w["SPY"], px_w["RSP"]).rename("SPY/RSP")

spx_daily = px_d["^GSPC"].dropna() if "^GSPC" in px_d.columns else pd.Series(dtype=float)
rv21_daily = realized_vol(spx_daily, 21)
rv21_w = weekly_last(rv21_daily).reindex(px_w.index).ffill()

dd_w = (-drawdown(spx_w) * 100.0).rename("Drawdown")

stress_components = pd.DataFrame(index=px_w.index)
stress_components["VIX"] = score_from_z(vix, invert=False)
stress_components["HY_OAS"] = score_from_z(fred_w["HY_OAS"], invert=False)
stress_components["HYG_LQD"] = score_from_z(hyg_lqd, invert=True)
stress_components["Curve"] = score_from_z(curve, invert=True)
stress_components["Funding"] = score_from_z(funding, invert=False)
stress_components["Drawdown"] = score_from_z(dd_w, invert=False)
stress_components["RV21"] = score_from_z(rv21_w, invert=False)
stress_components["Breadth"] = score_from_z(spy_rsp_stress, invert=False)

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
bucket_df = pd.concat([equity_bucket, credit_bucket, xasset_bucket, macro_bucket, stress_score], axis=1)
bucket_now = bucket_df.iloc[-1].copy()

signal_now = sig_df.iloc[-1].sort_values(ascending=False).to_frame("Score")
signal_now["1W Δ"] = sig_df.diff(1).iloc[-1].reindex(signal_now.index)
signal_now["4W Δ"] = sig_df.diff(4).iloc[-1].reindex(signal_now.index)

breadth = ((sig_df > 50).sum(axis=1) / sig_df.shape[1] * 100.0).rename("Signal Breadth")
confidence = (100.0 - sig_df.std(axis=1).clip(0, 50) * 2.0).clip(0, 100).rename("Confidence")

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
