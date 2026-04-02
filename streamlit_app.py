import io
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==============================
# App config
# ==============================
TITLE = "Weekly Risk On / Risk Off Meter"
SUBTITLE = "Allocator-grade weekly regime dashboard across tape, macro, fragility, and cross-asset confirmation."

st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.caption(SUBTITLE)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Settings")
    lookback_years = st.selectbox("History window", [3, 5, 10], index=1)
    smooth_weeks = st.slider("Smoothing (weeks)", 1, 6, 2)
    min_confirm_weeks = st.slider("State change confirmation (weeks)", 1, 3, 2)
    regime_window = st.selectbox("Cross-asset correlation window", [21, 63], index=1)
    st.divider()
    st.markdown(
        """
        **Model design**
        - Fast Tape = equity leadership + credit/vol
        - Macro Backdrop = liquidity, curve, funding, real-rate/FCI context
        - Fragility = stress, drawdown, vol, spread widening, breadth damage
        - Confirmation = cross-asset ratios + rolling-correlation alignment
        - Final state uses a persistence-aware state machine, not a flat average
        """
    )

START = (datetime.today() - timedelta(days=int(365.25 * lookback_years + 180))).strftime("%Y-%m-%d")

# ==============================
# Universe
# ==============================
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
CYCLICALS = ["XLY", "XLI", "XLF", "XLB", "XLK", "XLE"]
DEFENSIVES = ["XLP", "XLV", "XLU", "XLRE"]

YF_TICKERS = [
    "SPY", "RSP", "IWM", "QQQ", "TLT",
    "XLY", "XLP", "SPHB", "SPLV",
    "XLI", "XLU", "XLF", "XLV", "XLE", "XLB", "XLRE", "XLK", "XLC",
    "HYG", "LQD", "IEF", "EEM",
    "DBC", "GLD", "SMH", "FXY",
    "AUDJPY=X", "USDJPY=X", "EURUSD=X",
    "HG=F", "CL=F",
    "^VIX", "^VIX3M", "^GSPC", "^TNX", "^RUT"
] + SECTORS
YF_TICKERS = list(dict.fromkeys(YF_TICKERS))

FACTOR_TICKERS = ["MTUM", "QUAL", "VLUE", "USMV", "SIZE"]

FRED_SERIES = {
    "WALCL": "WALCL",
    "RRP": "RRPONTSYD",
    "TGA": "WTREGEN",
    "HY_OAS": "BAMLH0A0HYM2",
    "T10Y3M": "T10Y3M",
    "T10Y2Y": "T10Y2Y",
    "CP3M": "DCPF3M",
    "TBILL3M": "DTB3",
    "REAL10Y": "DFII10",
    "BREAKEVEN10Y": "T10YIE",
    "NFCI": "NFCI",
}

# ==============================
# Data loaders
# ==============================
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
            if t in lvl0 and "Close" in raw[t].columns:
                s = pd.to_numeric(raw[t]["Close"], errors="coerce").dropna()
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


# ==============================
# Helpers
# ==============================
def weekly_last(df):
    out = df.resample("W-FRI").last()
    if isinstance(df, pd.Series):
        return out.dropna()
    return out


def get_series(df: pd.DataFrame, col: str, index: pd.Index) -> pd.Series:
    if isinstance(df, pd.Series):
        s = pd.to_numeric(df, errors="coerce")
        return s.reindex(index)
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        return s.reindex(index)
    return pd.Series(np.nan, index=index, name=col)


def safe_ratio(a: pd.Series, b: pd.Series, name: str = "") -> pd.Series:
    a, b = a.align(b, join="outer")
    out = a / b.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    if name:
        out = out.rename(name)
    return out


def rolling_pct_rank(series: pd.Series, window=104) -> pd.Series:
    def _pct(x):
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1])
    return series.rolling(window, min_periods=max(12, window // 4)).apply(_pct, raw=False)


def zscore(series: pd.Series, window=52) -> pd.Series:
    mu = series.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = series.rolling(window, min_periods=max(10, window // 4)).std(ddof=0)
    return (series - mu) / sd.replace(0, np.nan)


def level_score(series: pd.Series, high_good=True, window=104) -> pd.Series:
    score = (rolling_pct_rank(series, window=window) * 100.0).clip(0, 100)
    return score if high_good else (100 - score)


def roc_score(series: pd.Series, periods=4, high_good=True, window=52) -> pd.Series:
    roc = series.pct_change(periods)
    score = (50 + 18 * zscore(roc, window=window).clip(-2.5, 2.5)).clip(0, 100)
    return score if high_good else (100 - score)


def trend_score(series: pd.Series, fast=10, slow=40) -> pd.Series:
    fast_ma = series.rolling(fast, min_periods=1).mean()
    slow_ma = series.rolling(slow, min_periods=1).mean()
    out = pd.Series(50.0, index=series.index)
    out += np.where(series > fast_ma, 25, -25)
    out += np.where(fast_ma > slow_ma, 25, -25)
    return out.clip(0, 100)


def composite_signal(series: pd.Series, high_good=True, fast=10, slow=40) -> pd.Series:
    t = trend_score(series, fast=fast, slow=slow)
    r = roc_score(series, periods=4, high_good=high_good, window=52)
    l = level_score(series, high_good=high_good, window=104)
    out = 0.40 * t + 0.35 * r + 0.25 * l
    return out.clip(0, 100)


def realized_vol(prices: pd.Series, window=21, annualization=252) -> pd.Series:
    r = prices.pct_change()
    return r.rolling(window).std() * np.sqrt(annualization)


def drawdown(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1.0


def rolling_corr_score(a: pd.Series, b: pd.Series, window=63, high_good=True) -> pd.Series:
    a, b = a.align(b, join="inner")
    corr = a.pct_change().rolling(window).corr(b.pct_change())
    sc = level_score(corr, high_good=high_good, window=104)
    return sc.reindex(a.index)


def classify_state(fast_tape, macro, fragility, confirm):
    # 0 Risk Off, 1 Mild Risk Off, 2 Neutral, 3 Mild Risk On, 4 Risk On
    if (fragility >= 72 and fast_tape <= 48) or (fragility >= 78):
        return 0
    if fast_tape >= 66 and fragility <= 44 and (macro >= 52 or confirm >= 56):
        return 4
    if fast_tape >= 57 and fragility <= 54 and (macro >= 48 or confirm >= 50):
        return 3
    if fast_tape <= 44 and fragility >= 60:
        return 1
    return 2


def persist_states(raw_state: pd.Series, confirm_weeks=2) -> pd.Series:
    raw_state = raw_state.astype(float).copy()
    persisted = raw_state.copy()
    if raw_state.empty:
        return persisted

    persisted.iloc[0] = raw_state.iloc[0]
    streak_state = raw_state.iloc[0]
    streak_len = 1

    for i in range(1, len(raw_state)):
        curr = raw_state.iloc[i]
        prev_persist = persisted.iloc[i - 1]

        if curr == streak_state:
            streak_len += 1
        else:
            streak_state = curr
            streak_len = 1

        if curr in (0, 4):
            persisted.iloc[i] = curr
        elif curr == prev_persist:
            persisted.iloc[i] = curr
        elif streak_len >= confirm_weeks:
            persisted.iloc[i] = curr
        else:
            persisted.iloc[i] = prev_persist

    return persisted


def state_label(x):
    mapping = {
        0: "Risk Off",
        1: "Mild Risk Off",
        2: "Neutral / Transition",
        3: "Mild Risk On",
        4: "Risk On",
    }
    return mapping.get(int(x), "Neutral / Transition")


def posture_text(score, state_num, confidence):
    if state_num >= 4 and confidence >= 65:
        return "Increase gross selectively"
    if state_num >= 3:
        return "Lean risk-on, but keep hedges tactical"
    if state_num <= 0 and confidence >= 60:
        return "Reduce gross and raise hedges"
    if state_num <= 1:
        return "Defensive bias, keep gross contained"
    return "Balanced posture, wait for confirmation"


def band_text(score):
    if score >= 70:
        return "High"
    if score >= 55:
        return "Positive"
    if score >= 45:
        return "Balanced"
    if score >= 30:
        return "Cautious"
    return "High Risk"


def fmt_delta(x):
    return "NA" if pd.isna(x) else f"{x:+.1f}"


# ==============================
# Load data
# ==============================
with st.spinner("Loading market and macro data..."):
    px_d = load_yahoo(YF_TICKERS, START)
    fred_d = load_fred(FRED_SERIES, START)
    factor_d = load_yahoo(FACTOR_TICKERS, START)

if px_d.empty:
    st.error("Failed to download Yahoo Finance data.")
    st.stop()

px_w = weekly_last(px_d)
master_index = px_w.index
fred_w = weekly_last(fred_d).reindex(master_index).ffill()
factor_w = weekly_last(factor_d).reindex(master_index).ffill() if not factor_d.empty else pd.DataFrame(index=master_index)

if smooth_weeks > 1:
    px_w = px_w.rolling(smooth_weeks, min_periods=1).mean()
    fred_w = fred_w.rolling(smooth_weeks, min_periods=1).mean()
    if not factor_w.empty:
        factor_w = factor_w.rolling(smooth_weeks, min_periods=1).mean()

# ==============================
# Extract primary series
# ==============================
spy = get_series(px_w, "SPY", master_index)
rsp = get_series(px_w, "RSP", master_index)
iwm = get_series(px_w, "IWM", master_index)
qqq = get_series(px_w, "QQQ", master_index)
tlt = get_series(px_w, "TLT", master_index)
xly = get_series(px_w, "XLY", master_index)
xlp = get_series(px_w, "XLP", master_index)
sphb = get_series(px_w, "SPHB", master_index)
splv = get_series(px_w, "SPLV", master_index)
xli = get_series(px_w, "XLI", master_index)
xlu = get_series(px_w, "XLU", master_index)
xlF = get_series(px_w, "XLF", master_index)
xlv = get_series(px_w, "XLV", master_index)
xle = get_series(px_w, "XLE", master_index)
xlb = get_series(px_w, "XLB", master_index)
xlre = get_series(px_w, "XLRE", master_index)
xlk = get_series(px_w, "XLK", master_index)
hyg = get_series(px_w, "HYG", master_index)
lqd = get_series(px_w, "LQD", master_index)
ief = get_series(px_w, "IEF", master_index)
eem = get_series(px_w, "EEM", master_index)
dbc = get_series(px_w, "DBC", master_index)
gld = get_series(px_w, "GLD", master_index)
smh = get_series(px_w, "SMH", master_index)
fxy = get_series(px_w, "FXY", master_index)
audjpy = get_series(px_w, "AUDJPY=X", master_index)
hg = get_series(px_w, "HG=F", master_index)
cl = get_series(px_w, "CL=F", master_index)
vix = get_series(px_w, "^VIX", master_index)
vix3m = get_series(px_w, "^VIX3M", master_index)
gspc = get_series(px_w, "^GSPC", master_index)
tnx = get_series(px_w, "^TNX", master_index)

# Derived macro
fred_w["NET_LIQUIDITY"] = get_series(fred_w, "WALCL", master_index) - get_series(fred_w, "RRP", master_index) - get_series(fred_w, "TGA", master_index)
netliq = get_series(fred_w, "NET_LIQUIDITY", master_index)
curve_10y3m = get_series(fred_w, "T10Y3M", master_index)
curve_10y2y = get_series(fred_w, "T10Y2Y", master_index)
funding = get_series(fred_w, "CP3M", master_index) - get_series(fred_w, "TBILL3M", master_index)
real10y = get_series(fred_w, "REAL10Y", master_index)
breakeven10y = get_series(fred_w, "BREAKEVEN10Y", master_index)
hy_oas = get_series(fred_w, "HY_OAS", master_index)
nfci = get_series(fred_w, "NFCI", master_index)

# ==============================
# Sleeve construction
# ==============================
# ---- Equity / tape ----
sector_px = pd.DataFrame(index=master_index)
for s in SECTORS:
    sector_px[s] = get_series(px_w, s, master_index)

sector_fast = sector_px.rolling(10, min_periods=1).mean()
sector_slow = sector_px.rolling(40, min_periods=1).mean()
sector_breadth = ((sector_px > sector_fast).sum(axis=1) / len(SECTORS) * 100.0)
sector_trend_breadth = ((sector_fast > sector_slow).sum(axis=1) / len(SECTORS) * 100.0)
sector_breadth_score = (0.5 * sector_breadth + 0.5 * sector_trend_breadth).clip(0, 100)

cyc_basket = sector_px[CYCLICALS].mean(axis=1)
def_basket = sector_px[DEFENSIVES].mean(axis=1)
cyc_def = safe_ratio(cyc_basket, def_basket, "Cyclical/Defensive EW")

# ---- Factor regime and crowding ----
factor_regime = pd.Series(50.0, index=master_index)
factor_breadth = pd.Series(50.0, index=master_index)
factor_crowding = pd.Series(50.0, index=master_index)

if not factor_w.empty:
    factor_short_pct = []
    trend_flags = []
    turning_up_flags = []
    turning_down_flags = []
    corr_rows = []

    for f in factor_w.columns:
        s = pd.to_numeric(factor_w[f], errors="coerce")
        short = s.pct_change(4) * 100.0
        long = s.pct_change(13) * 100.0
        fast = s.rolling(10, min_periods=1).mean()
        slow = s.rolling(40, min_periods=1).mean()
        trend_up = (fast > slow).astype(float)
        trend_down = (fast < slow).astype(float)
        infl = short - long
        turning_up = (infl > infl.rolling(4, min_periods=1).mean()).astype(float)
        turning_down = (infl < infl.rolling(4, min_periods=1).mean()).astype(float)

        factor_short_pct.append(short.rename(f))
        trend_flags.append(trend_up.rename(f))
        turning_up_flags.append(turning_up.rename(f))
        turning_down_flags.append(turning_down.rename(f))
        corr_rows.append(s.pct_change().rename(f))

    short_df = pd.concat(factor_short_pct, axis=1)
    trend_up_df = pd.concat(trend_flags, axis=1)
    turning_up_df = pd.concat(turning_up_flags, axis=1)
    turning_down_df = pd.concat(turning_down_flags, axis=1)
    factor_ret = pd.concat(corr_rows, axis=1)

    raw_score = (
        0.4 * short_df.mean(axis=1)
        + 0.3 * (turning_up_df.mean(axis=1) - turning_down_df.mean(axis=1))
        + 0.3 * (2 * trend_up_df.mean(axis=1) - 1)
    )
    factor_regime = (50 + 50 * (raw_score / 5.0)).clip(0, 100).reindex(master_index).fillna(50)
    factor_breadth = (trend_up_df.mean(axis=1) * 100.0).reindex(master_index).fillna(50)

    abs_corr_avg = []
    for i in range(len(master_index)):
        window_df = factor_ret.loc[: master_index[i]].tail(26).dropna(how="all")
        if window_df.shape[1] < 2 or len(window_df) < 8:
            abs_corr_avg.append(np.nan)
            continue
        corr = window_df.corr().abs()
        vals = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack()
        abs_corr_avg.append(vals.mean() if not vals.empty else np.nan)
    factor_crowding = (pd.Series(abs_corr_avg, index=master_index) * 100.0).clip(0, 100).fillna(50)

# ---- Signals ----
signals = pd.DataFrame(index=master_index)
signal_group = {}

# Tape: equity leadership
signals["RSP/SPY"] = composite_signal(safe_ratio(rsp, spy, "RSP/SPY"), high_good=True).reindex(master_index).fillna(50)
signal_group["RSP/SPY"] = "Fast Tape"
signals["IWM/SPY"] = composite_signal(safe_ratio(iwm, spy, "IWM/SPY"), high_good=True).reindex(master_index).fillna(50)
signal_group["IWM/SPY"] = "Fast Tape"
signals["XLY/XLP"] = composite_signal(safe_ratio(xly, xlp, "XLY/XLP"), high_good=True).reindex(master_index).fillna(50)
signal_group["XLY/XLP"] = "Fast Tape"
signals["SPHB/SPLV"] = composite_signal(safe_ratio(sphb, splv, "SPHB/SPLV"), high_good=True).reindex(master_index).fillna(50)
signal_group["SPHB/SPLV"] = "Fast Tape"
signals["Cyc/Def EW"] = composite_signal(cyc_def, high_good=True).reindex(master_index).fillna(50)
signal_group["Cyc/Def EW"] = "Fast Tape"
signals["XLI/XLU"] = composite_signal(safe_ratio(xli, xlu, "XLI/XLU"), high_good=True).reindex(master_index).fillna(50)
signal_group["XLI/XLU"] = "Fast Tape"
signals["XLF/XLU"] = composite_signal(safe_ratio(xlF, xlu, "XLF/XLU"), high_good=True).reindex(master_index).fillna(50)
signal_group["XLF/XLU"] = "Fast Tape"
signals["Sector Breadth"] = sector_breadth_score.reindex(master_index).fillna(50)
signal_group["Sector Breadth"] = "Fast Tape"
signals["Factor Regime"] = factor_regime.reindex(master_index).fillna(50)
signal_group["Factor Regime"] = "Fast Tape"
signals["Factor Breadth"] = factor_breadth.reindex(master_index).fillna(50)
signal_group["Factor Breadth"] = "Fast Tape"

# Credit / vol
signals["HYG/LQD"] = composite_signal(safe_ratio(hyg, lqd, "HYG/LQD"), high_good=True).reindex(master_index).fillna(50)
signal_group["HYG/LQD"] = "Credit / Vol"
signals["HYG/IEF"] = composite_signal(safe_ratio(hyg, ief, "HYG/IEF"), high_good=True).reindex(master_index).fillna(50)
signal_group["HYG/IEF"] = "Credit / Vol"
signals["VIX"] = composite_signal(vix, high_good=False).reindex(master_index).fillna(50)
signal_group["VIX"] = "Credit / Vol"
signals["VIX/VIX3M"] = composite_signal(safe_ratio(vix, vix3m, "VIX/VIX3M"), high_good=False).reindex(master_index).fillna(50)
signal_group["VIX/VIX3M"] = "Credit / Vol"
signals["HY OAS"] = composite_signal(hy_oas, high_good=False).reindex(master_index).fillna(50)
signal_group["HY OAS"] = "Credit / Vol"

# Macro backdrop
signals["Net Liquidity"] = (0.6 * roc_score(netliq, periods=4, high_good=True) + 0.4 * level_score(netliq, high_good=True)).reindex(master_index).fillna(50)
signal_group["Net Liquidity"] = "Macro Backdrop"
signals["10Y-3M"] = composite_signal(curve_10y3m, high_good=True).reindex(master_index).fillna(50)
signal_group["10Y-3M"] = "Macro Backdrop"
signals["10Y-2Y"] = composite_signal(curve_10y2y, high_good=True).reindex(master_index).fillna(50)
signal_group["10Y-2Y"] = "Macro Backdrop"
signals["Funding"] = composite_signal(funding, high_good=False).reindex(master_index).fillna(50)
signal_group["Funding"] = "Macro Backdrop"
signals["Real 10Y"] = (0.65 * roc_score(real10y, periods=4, high_good=False) + 0.35 * level_score(real10y, high_good=False)).reindex(master_index).fillna(50)
signal_group["Real 10Y"] = "Macro Backdrop"
signals["NFCI"] = composite_signal(nfci, high_good=False).reindex(master_index).fillna(50)
signal_group["NFCI"] = "Macro Backdrop"
signals["Breakeven 10Y"] = composite_signal(breakeven10y, high_good=True).reindex(master_index).fillna(50)
signal_group["Breakeven 10Y"] = "Macro Backdrop"

# Cross-asset confirmation
signals["DBC/GLD"] = composite_signal(safe_ratio(dbc, gld, "DBC/GLD"), high_good=True).reindex(master_index).fillna(50)
signal_group["DBC/GLD"] = "Confirmation"
signals["Copper/Gold"] = composite_signal(safe_ratio(hg, gld, "Copper/Gold"), high_good=True).reindex(master_index).fillna(50)
signal_group["Copper/Gold"] = "Confirmation"
signals["AUDJPY"] = composite_signal(audjpy, high_good=True).reindex(master_index).fillna(50)
signal_group["AUDJPY"] = "Confirmation"
signals["SMH/FXY"] = composite_signal(safe_ratio(smh, fxy, "SMH/FXY"), high_good=True).reindex(master_index).fillna(50)
signal_group["SMH/FXY"] = "Confirmation"
signals["SPY/TLT"] = composite_signal(safe_ratio(spy, tlt, "SPY/TLT"), high_good=True).reindex(master_index).fillna(50)
signal_group["SPY/TLT"] = "Confirmation"

# Rolling-correlation alignment from compass logic
corr_spx_tnx = rolling_corr_score(get_series(px_d, "^GSPC", px_d.index), get_series(px_d, "^TNX", px_d.index), window=regime_window, high_good=False)
corr_hyg_oil = rolling_corr_score(get_series(px_d, "HYG", px_d.index), get_series(px_d, "CL=F", px_d.index), window=regime_window, high_good=True)
corr_eem_hyg = rolling_corr_score(get_series(px_d, "EEM", px_d.index), get_series(px_d, "HYG", px_d.index), window=regime_window, high_good=True)

signals["Corr SPX-10Y"] = weekly_last(corr_spx_tnx).reindex(master_index).ffill().fillna(50)
signal_group["Corr SPX-10Y"] = "Confirmation"
signals["Corr HY-Oil"] = weekly_last(corr_hyg_oil).reindex(master_index).ffill().fillna(50)
signal_group["Corr HY-Oil"] = "Confirmation"
signals["Corr EM-HY"] = weekly_last(corr_eem_hyg).reindex(master_index).ffill().fillna(50)
signal_group["Corr EM-HY"] = "Confirmation"

# Fragility sleeve - stress composite family
spx_daily = get_series(px_d, "^GSPC", px_d.index)
rv21 = realized_vol(spx_daily, 21)
rv21_w = weekly_last(rv21).reindex(master_index).ffill()
drawdown_w = (-drawdown(gspc) * 100.0).reindex(master_index)
spy_rsp_stress = safe_ratio(spy, rsp, "SPY/RSP")
spread_accel = hy_oas.diff(4)

fragility_components = pd.DataFrame(index=master_index)
fragility_components["VIX"] = composite_signal(vix, high_good=True).reindex(master_index).fillna(50)
fragility_components["HY OAS"] = composite_signal(hy_oas, high_good=True).reindex(master_index).fillna(50)
fragility_components["HYG/LQD"] = composite_signal(safe_ratio(hyg, lqd, "HYG/LQD"), high_good=False).reindex(master_index).fillna(50)
fragility_components["Curve"] = composite_signal(curve_10y3m, high_good=False).reindex(master_index).fillna(50)
fragility_components["Funding"] = composite_signal(funding, high_good=True).reindex(master_index).fillna(50)
fragility_components["Drawdown"] = composite_signal(drawdown_w, high_good=True).reindex(master_index).fillna(50)
fragility_components["RV21"] = composite_signal(rv21_w, high_good=True).reindex(master_index).fillna(50)
fragility_components["Breadth Damage"] = composite_signal(spy_rsp_stress, high_good=True).reindex(master_index).fillna(50)
fragility_components["Spread Accel"] = composite_signal(spread_accel, high_good=True).reindex(master_index).fillna(50)

fragility_weights = {
    "VIX": 0.18,
    "HY OAS": 0.15,
    "HYG/LQD": 0.14,
    "Curve": 0.10,
    "Funding": 0.08,
    "Drawdown": 0.12,
    "RV21": 0.10,
    "Breadth Damage": 0.07,
    "Spread Accel": 0.06,
}
fragility = sum(fragility_weights[k] * fragility_components[k] for k in fragility_weights).rename("Fragility")

# ==============================
# Sleeves and headline
# ==============================
fast_tape = signals[[c for c in signals.columns if signal_group[c] == "Fast Tape"]].mean(axis=1).rename("Fast Tape")
credit_vol = signals[[c for c in signals.columns if signal_group[c] == "Credit / Vol"]].mean(axis=1).rename("Credit / Vol")
macro_backdrop = signals[[c for c in signals.columns if signal_group[c] == "Macro Backdrop"]].mean(axis=1).rename("Macro Backdrop")
confirmation = signals[[c for c in signals.columns if signal_group[c] == "Confirmation"]].mean(axis=1).rename("Confirmation")

headline_score = (
    0.40 * (0.60 * fast_tape + 0.40 * credit_vol)
    + 0.25 * macro_backdrop
    + 0.20 * confirmation
    + 0.15 * (100 - fragility)
).clip(0, 100).rename("Headline Score")

raw_state = pd.Series(
    [classify_state((0.60 * fast_tape.iloc[i] + 0.40 * credit_vol.iloc[i]), macro_backdrop.iloc[i], fragility.iloc[i], confirmation.iloc[i]) for i in range(len(master_index))],
    index=master_index,
    name="Raw State",
)
persistent_state = persist_states(raw_state, confirm_weeks=min_confirm_weeks).rename("Persistent State")

# Breadth, disagreement, coverage, confidence
signal_breadth = ((signals > 50).sum(axis=1) / signals.shape[1] * 100.0).rename("Signal Breadth")

sleeve_df = pd.concat([fast_tape, credit_vol, macro_backdrop, confirmation, fragility], axis=1)
# For agreement, invert fragility so higher = better
agreement_df = pd.concat([fast_tape, credit_vol, macro_backdrop, confirmation, (100 - fragility).rename("1-Fragility")], axis=1)
disagreement = agreement_df.std(axis=1).rename("Disagreement")

coverage_flags = pd.DataFrame(index=master_index)
for c in signals.columns:
    coverage_flags[c] = ~signals[c].eq(50)
data_coverage = coverage_flags.mean(axis=1) * 100.0

confidence = (
    0.35 * (100 - disagreement.clip(0, 50) * 2)
    + 0.30 * signal_breadth
    + 0.20 * data_coverage
    + 0.15 * (100 - factor_crowding.clip(0, 100))
).clip(0, 100).rename("Confidence")

# Posture outputs
cyc_signal = signals["Cyc/Def EW"]
duration_signal = (0.45 * signals["Real 10Y"] + 0.30 * signals["10Y-3M"] + 0.25 * (100 - fragility)).clip(0, 100)

curr_idx = master_index[-1]
state_now = int(persistent_state.iloc[-1])
state_prev = int(persistent_state.iloc[-2]) if len(persistent_state) > 1 else state_now
score_now = float(headline_score.iloc[-1])
score_prev = float(headline_score.iloc[-2]) if len(headline_score) > 1 else score_now
score_4w = float(headline_score.iloc[-5]) if len(headline_score) > 4 else np.nan
conf_now = float(confidence.iloc[-1])

# Decision text
if score_now >= 62 and fragility.iloc[-1] <= 50:
    equity_beta = "Add equity beta"
elif score_now <= 42 or fragility.iloc[-1] >= 65:
    equity_beta = "Cut equity beta"
else:
    equity_beta = "Hold balanced equity beta"

if credit_vol.iloc[-1] >= 60 and fragility.iloc[-1] <= 55:
    credit_beta_stance = "Own credit beta"
elif credit_vol.iloc[-1] <= 45 or fragility.iloc[-1] >= 65:
    credit_beta_stance = "De-risk credit beta"
else:
    credit_beta_stance = "Neutral credit exposure"

if duration_signal.iloc[-1] >= 58:
    duration_stance = "Favor duration"
elif duration_signal.iloc[-1] <= 42:
    duration_stance = "Underweight duration"
else:
    duration_stance = "Neutral duration"

if cyc_signal.iloc[-1] >= 58:
    cyc_vs_def = "Prefer cyclicals"
elif cyc_signal.iloc[-1] <= 42:
    cyc_vs_def = "Prefer defensives"
else:
    cyc_vs_def = "Balanced cyclicals vs defensives"

if fragility.iloc[-1] >= 68:
    hedge_urgency = "High hedge urgency"
elif fragility.iloc[-1] >= 58:
    hedge_urgency = "Moderate hedge urgency"
else:
    hedge_urgency = "Low hedge urgency"

# ==============================
# Presentation helpers
# ==============================
def stat_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px 16px; background:rgba(255,255,255,0.03);">
            <div style="font-size:12px; color:#9ca3af; margin-bottom:4px;">{title}</div>
            <div style="font-size:28px; font-weight:700; line-height:1.1;">{value}</div>
            <div style="font-size:12px; color:#9ca3af; margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_box():
    posture = posture_text(score_now, state_now, conf_now)
    st.markdown(
        f"""
        <div style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 20px; background:rgba(255,255,255,0.04); margin-bottom:8px;">
            <div style="font-size:13px; color:#9ca3af; margin-bottom:6px;">Allocator posture</div>
            <div style="font-size:26px; font-weight:700; margin-bottom:8px;">{state_label(state_now)}</div>
            <div style="font-size:15px; color:#d1d5db; margin-bottom:10px;">{posture}</div>
            <div style="font-size:13px; color:#9ca3af;">Headline score <b>{score_now:.1f}</b> | 1W change <b>{score_now - score_prev:+.1f}</b> | 4W change <b>{'NA' if pd.isna(score_4w) else f'{score_now - score_4w:+.1f}'}</b> | Breadth <b>{signal_breadth.iloc[-1]:.0f}%</b> | Confidence <b>{conf_now:.0f}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


summary_box()

# Top stat cards
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    stat_card("Headline", f"{score_now:.0f}", band_text(score_now))
with c2:
    stat_card("Confidence", f"{conf_now:.0f}", "Model confidence")
with c3:
    stat_card("1W Change", f"{score_now - score_prev:+.1f}", "Weekly momentum")
with c4:
    stat_card("Breadth", f"{signal_breadth.iloc[-1]:.0f}%", "Signals > 50")
with c5:
    stat_card("Fragility", f"{fragility.iloc[-1]:.0f}", band_text(100 - fragility.iloc[-1]))

# Sleeve tiles
st.markdown("### Sleeve dashboard")
s1, s2, s3, s4 = st.columns(4)
for col, title, series in [
    (s1, "Fast Tape", (0.60 * fast_tape + 0.40 * credit_vol)),
    (s2, "Macro Backdrop", macro_backdrop),
    (s3, "Cross-Asset Confirmation", confirmation),
    (s4, "Fragility", fragility),
]:
    with col:
        curr = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) > 1 else curr
        delta = curr - prev
        spark = series.tail(20)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spark.index, y=spark.values, mode="lines", line=dict(width=2), showlegend=False))
        fig.update_layout(
            template="plotly_dark",
            height=180,
            margin=dict(l=10, r=10, t=38, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(range=[0, 100], showgrid=False, zeroline=False, showticklabels=False),
            title=dict(text=f"{title}<br><sup>{curr:.1f} | {delta:+.1f} 1W</sup>", x=0.03, xanchor="left"),
        )
        st.plotly_chart(fig, use_container_width=True)

# Decision grid
st.markdown("### Allocator outputs")
d1, d2, d3, d4, d5 = st.columns(5)
with d1:
    stat_card("Equity beta", equity_beta)
with d2:
    stat_card("Credit beta", credit_beta_stance)
with d3:
    stat_card("Duration", duration_stance)
with d4:
    stat_card("Cyclicals vs defensives", cyc_vs_def)
with d5:
    stat_card("Hedge urgency", hedge_urgency)

# Regime history + sleeve comparison
left, right = st.columns([1.35, 1.0])

with left:
    hist = headline_score.tail(104)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Headline", line=dict(width=2.5, color="#60a5fa")))
    fig_hist.add_hrect(y0=70, y1=100, fillcolor="#166534", opacity=0.12, line_width=0)
    fig_hist.add_hrect(y0=55, y1=70, fillcolor="#16a34a", opacity=0.10, line_width=0)
    fig_hist.add_hrect(y0=45, y1=55, fillcolor="#737373", opacity=0.10, line_width=0)
    fig_hist.add_hrect(y0=30, y1=45, fillcolor="#b45309", opacity=0.10, line_width=0)
    fig_hist.add_hrect(y0=0, y1=30, fillcolor="#7f1d1d", opacity=0.12, line_width=0)
    fig_hist.update_layout(
        template="plotly_dark",
        title="52-Week Regime History",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        yaxis=dict(range=[0, 100], title="Score"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    bucket_now = pd.Series({
        "Fast Tape": float((0.60 * fast_tape + 0.40 * credit_vol).iloc[-1]),
        "Macro Backdrop": float(macro_backdrop.iloc[-1]),
        "Confirmation": float(confirmation.iloc[-1]),
        "Fragility": float(fragility.iloc[-1]),
    }).sort_values(ascending=True)
    colors = ["#93c5fd", "#c4b5fd", "#86efac", "#fca5a5"]
    fig_bars = go.Figure()
    for i, (label, value) in enumerate(bucket_now.items()):
        fig_bars.add_trace(
            go.Bar(
                x=[value],
                y=[label],
                orientation="h",
                text=[f"{value:.0f}"],
                textposition="outside",
                marker_color=colors[i],
                showlegend=False,
            )
        )
    fig_bars.update_layout(
        template="plotly_dark",
        title="Sleeve Scores",
        height=360,
        margin=dict(l=10, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(range=[0, 100], title="Score"),
        yaxis=dict(title=""),
        barmode="group",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

# Grouped heatmap
st.markdown("### Grouped signal heatmap")
heat_cols = list(signals.columns)
heat_rows = ["Fast Tape", "Credit / Vol", "Macro Backdrop", "Confirmation"]
heat = pd.DataFrame(np.nan, index=heat_rows, columns=heat_cols)
for col in heat_cols:
    heat.loc[signal_group[col], col] = signals[col].iloc[-1]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=heat.values,
        x=heat.columns.tolist(),
        y=heat.index.tolist(),
        colorscale="RdYlGn",
        zmin=0,
        zmax=100,
        text=np.where(np.isnan(heat.values), "", np.round(heat.values).astype("Int64").astype(str)),
        texttemplate="%{text}",
        hovertemplate="Signal: %{x}<br>Sleeve: %{y}<br>Score: %{z:.1f}<extra></extra>",
        colorbar=dict(title="Score"),
    )
)
fig_heat.update_layout(
    template="plotly_dark",
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
)
st.plotly_chart(fig_heat, use_container_width=True)

# Disagreement / confidence diagnostics
mid1, mid2 = st.columns([1.05, 0.95])
with mid1:
    diag = pd.DataFrame({
        "Disagreement": disagreement.tail(52),
        "Confidence": confidence.tail(52),
        "Factor Crowding": factor_crowding.tail(52),
    })
    fig_diag = make_subplots(specs=[[{"secondary_y": True}]])
    fig_diag.add_trace(go.Scatter(x=diag.index, y=diag["Confidence"], name="Confidence", line=dict(color="#60a5fa", width=2)), secondary_y=False)
    fig_diag.add_trace(go.Scatter(x=diag.index, y=diag["Disagreement"], name="Disagreement", line=dict(color="#f59e0b", width=2)), secondary_y=True)
    fig_diag.add_trace(go.Scatter(x=diag.index, y=diag["Factor Crowding"], name="Factor Crowding", line=dict(color="#f87171", width=2, dash="dot")), secondary_y=True)
    fig_diag.update_layout(
        template="plotly_dark",
        title="Confidence vs disagreement",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_diag.update_yaxes(title_text="Confidence", range=[0, 100], secondary_y=False)
    fig_diag.update_yaxes(title_text="Disagreement / Crowding", range=[0, 100], secondary_y=True)
    st.plotly_chart(fig_diag, use_container_width=True)

with mid2:
    raw_disp = pd.Series({
        "Coverage": float(data_coverage.iloc[-1]),
        "Breadth": float(signal_breadth.iloc[-1]),
        "Confidence": float(confidence.iloc[-1]),
        "Crowding (inv)": float(100 - factor_crowding.iloc[-1]),
    }).sort_values(ascending=True)
    fig_diag_bar = go.Figure()
    for label, value in raw_disp.items():
        fig_diag_bar.add_trace(go.Bar(x=[value], y=[label], orientation="h", text=[f"{value:.0f}"], textposition="outside", showlegend=False))
    fig_diag_bar.update_layout(
        template="plotly_dark",
        title="Model quality diagnostics",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(range=[0, 100], title="Score"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_diag_bar, use_container_width=True)

# Detail table behind expander
with st.expander("Signal snapshot and model internals"):
    table = pd.DataFrame(index=signals.columns)
    table["Sleeve"] = [signal_group[c] for c in signals.columns]
    table["Score"] = signals.iloc[-1].round(1)
    table["1W Δ"] = signals.diff(1).iloc[-1].round(1)
    table["4W Δ"] = signals.diff(4).iloc[-1].round(1)
    table = table.sort_values(["Sleeve", "Score"], ascending=[True, False])
    st.dataframe(table, use_container_width=True)

    internals = pd.DataFrame({
        "Fast Tape": fast_tape,
        "Credit / Vol": credit_vol,
        "Macro Backdrop": macro_backdrop,
        "Confirmation": confirmation,
        "Fragility": fragility,
        "Headline": headline_score,
        "Raw State": raw_state,
        "Persistent State": persistent_state,
        "Confidence": confidence,
        "Breadth": signal_breadth,
        "Coverage": data_coverage,
        "Factor Crowding": factor_crowding,
    }).tail(12)
    st.dataframe(internals.round(1), use_container_width=True)

st.caption("© 2026 AD Fund Management LP")
