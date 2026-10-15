"""Phase 2-Pro feature engineering — 61-feature stack from raw OHLCV + index + industry.

Features:

Multi-horizon returns (7):
  ret_1d, ret_5d, ret_10d, ret_20d, ret_60d, ret_120d, ret_5d_minus_20d

Skip-window momentum (3):
  mom_20_skip5, mom_60_skip5, mom_60_over_20

Volatility (5):
  vol_5d, vol_20d, vol_60d, downside_vol_20d, parkinson_20d

Microstructure (4):
  oc_ratio, hl_range, overnight_ret, intraday_ret

Trend (5):
  close_over_ma5, close_over_ma20, close_over_ma60, close_over_ma120, ma5_over_ma20

Technicals (3):
  rsi_14, macd_hist, boll_pctb_20

Volume / amount (5):
  volume_z_20d, amount_log, amount_ma_20d, amount_ratio_5_20, days_since_volume_spike

Turnover (3):
  turnover_ma_20d, turnover_std_20d, turnover_ratio_5_20

Liquidity (Amihud) (3):
  amihud_5d, amihud_20d, amihud_ratio_5_20

Beta / idio vol (2):
  beta_60d, idio_vol_60d

Industry-neutral residuals (3):
  ret_20d_indneut, ret_60d_indneut, mom_60_skip5_indneut

Cross-sectional ranks (18):
  daily pct rank of every base feature (subset)

Total: 61 features
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

FORWARD_HORIZON = 5
TARGET_COL = "target_5d"

# --- Base features (~40 raw) ----------------------------------------------

BASE_FEATURES = [
    # Multi-horizon returns
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "ret_5d_minus_20d",
    # Skip-window momentum
    "mom_20_skip5", "mom_60_skip5", "mom_60_over_20",
    # Volatility
    "vol_5d", "vol_20d", "vol_60d", "downside_vol_20d", "parkinson_20d",
    # Microstructure
    "oc_ratio", "hl_range", "overnight_ret", "intraday_ret",
    # Trend
    "close_over_ma5", "close_over_ma20", "close_over_ma60",
    "close_over_ma120", "ma5_over_ma20",
    # Technicals
    "rsi_14", "macd_hist", "boll_pctb_20",
    # Volume/amount
    "volume_z_20d", "amount_log", "amount_ma_20d",
    "amount_ratio_5_20", "days_since_volume_spike",
    # Turnover
    "turnover_ma_20d", "turnover_std_20d", "turnover_ratio_5_20",
    # Liquidity (Amihud)
    "amihud_5d", "amihud_20d", "amihud_ratio_5_20",
    # Beta / idio vol
    "beta_60d", "idio_vol_60d",
    # Industry-neutral
    "ret_20d_indneut", "ret_60d_indneut", "mom_60_skip5_indneut",
]

# Cross-sectional rank versions (subset that benefits from ranking)
RANKED_FEATURES = [
    "ret_5d", "ret_20d", "ret_60d", "ret_120d",
    "vol_20d", "vol_60d", "parkinson_20d",
    "close_over_ma20", "close_over_ma60", "close_over_ma120",
    "amount_ratio_5_20", "amihud_20d",
    "beta_60d", "idio_vol_60d",
    "ret_20d_indneut", "ret_60d_indneut",
    "boll_pctb_20", "rsi_14",
]

ALL_FEATURES = BASE_FEATURES + [f"{f}_rank" for f in RANKED_FEATURES]


def _per_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-stock features (depends only on a single stock's history)."""
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    op = df["open"].astype(float) if "open" in df.columns else close
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    amt = df["amount"].astype(float) if "amount" in df.columns else pd.Series(np.nan, index=df.index)
    turn = df["turnover"].astype(float) if "turnover" in df.columns else pd.Series(np.nan, index=df.index)

    # Returns
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    df["ret_60d"] = close.pct_change(60)
    df["ret_120d"] = close.pct_change(120)
    df["ret_5d_minus_20d"] = df["ret_5d"] - df["ret_20d"]

    # Skip-window momentum: 20d return over [-20, -5], 60d over [-60, -5]
    df["mom_20_skip5"] = close.shift(5).pct_change(15)   # close(t-5)/close(t-20) - 1
    df["mom_60_skip5"] = close.shift(5).pct_change(55)   # close(t-5)/close(t-60) - 1
    df["mom_60_over_20"] = df["ret_60d"] - df["ret_20d"]

    # Volatility
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_60d"] = df["ret_1d"].rolling(60).std()
    neg = df["ret_1d"].where(df["ret_1d"] < 0, 0.0)
    df["downside_vol_20d"] = neg.rolling(20).std()
    # Parkinson volatility: sqrt(1/(4 ln 2) * mean(ln(H/L)^2))
    log_hl = np.log((high / low.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan))
    df["parkinson_20d"] = np.sqrt((log_hl ** 2).rolling(20).mean() / (4 * np.log(2)))

    # Microstructure
    df["oc_ratio"] = (close - op) / op.replace(0, np.nan)
    df["hl_range"] = (high - low) / close.replace(0, np.nan)
    df["overnight_ret"] = op / close.shift(1).replace(0, np.nan) - 1.0
    df["intraday_ret"] = close / op.replace(0, np.nan) - 1.0

    # Trend (close over moving average)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    df["close_over_ma5"] = close / ma5.replace(0, np.nan) - 1.0
    df["close_over_ma20"] = close / ma20.replace(0, np.nan) - 1.0
    df["close_over_ma60"] = close / ma60.replace(0, np.nan) - 1.0
    df["close_over_ma120"] = close / ma120.replace(0, np.nan) - 1.0
    df["ma5_over_ma20"] = ma5 / ma20.replace(0, np.nan) - 1.0

    # Technicals — RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD histogram (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - sig) / close.replace(0, np.nan)

    # Bollinger %B (20, 2)
    bm = close.rolling(20).mean()
    bs = close.rolling(20).std()
    upper = bm + 2 * bs
    lower = bm - 2 * bs
    rng = (upper - lower).replace(0, np.nan)
    df["boll_pctb_20"] = (close - lower) / rng

    # Volume / amount
    vmean = vol.rolling(20).mean()
    vstd = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vmean) / vstd
    df["amount_log"] = np.log(amt.replace(0, np.nan))
    df["amount_ma_20d"] = amt.rolling(20).mean()
    df["amount_ratio_5_20"] = (
        amt.rolling(5).mean() / amt.rolling(20).mean().replace(0, np.nan) - 1.0
    )
    # Days since most recent volume spike (>2x 20d average)
    spike = (vol > 2.0 * vmean).astype(int)
    days_since = []
    last_spike = -10**6
    for i, s in enumerate(spike.values):
        if s == 1:
            last_spike = i
        days_since.append(i - last_spike if last_spike >= 0 else np.nan)
    df["days_since_volume_spike"] = days_since
    df["days_since_volume_spike"] = df["days_since_volume_spike"].clip(upper=120)

    # Turnover
    df["turnover_ma_20d"] = turn.rolling(20).mean()
    df["turnover_std_20d"] = turn.rolling(20).std()
    df["turnover_ratio_5_20"] = (
        turn.rolling(5).mean() / turn.rolling(20).mean().replace(0, np.nan) - 1.0
    )

    # Amihud illiquidity
    if amt.notna().any():
        ill = (df["ret_1d"].abs() / amt.replace(0, np.nan)) * 1e9
        df["amihud_5d"] = ill.rolling(5).mean()
        df["amihud_20d"] = ill.rolling(20).mean()
        df["amihud_ratio_5_20"] = (
            df["amihud_5d"] / df["amihud_20d"].replace(0, np.nan) - 1.0
        )
    else:
        df["amihud_5d"] = np.nan
        df["amihud_20d"] = np.nan
        df["amihud_ratio_5_20"] = np.nan

    # Forward target
    df[TARGET_COL] = close.shift(-FORWARD_HORIZON) / close - 1.0

    return df


def _add_beta_and_idio(panel: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Rolling 60-day beta and idiosyncratic vol vs CSI500."""
    idx = index_df.copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx["idx_ret"] = idx["close"].pct_change()

    panel = panel.merge(idx[["date", "idx_ret"]], on="date", how="left")

    def _stock_beta(g):
        g = g.sort_values("date").copy()
        r = g["ret_1d"]
        m = g["idx_ret"]
        # rolling 60-day beta and residual vol
        cov = r.rolling(60).cov(m)
        var = m.rolling(60).var().replace(0, np.nan)
        beta = cov / var
        resid = r - beta * m
        idio = resid.rolling(60).std()
        g["beta_60d"] = beta
        g["idio_vol_60d"] = idio
        return g

    panel = panel.groupby("stock_code", group_keys=False).apply(_stock_beta)
    return panel.drop(columns=["idx_ret"])


def _add_industry_neutral(panel: pd.DataFrame) -> pd.DataFrame:
    """Industry-relative residuals: feature - industry median per date."""
    if "industry" not in panel.columns:
        for c in ["ret_20d_indneut", "ret_60d_indneut", "mom_60_skip5_indneut"]:
            panel[c] = np.nan
        return panel

    for src, dst in [("ret_20d", "ret_20d_indneut"),
                     ("ret_60d", "ret_60d_indneut"),
                     ("mom_60_skip5", "mom_60_skip5_indneut")]:
        med = panel.groupby(["date", "industry"])[src].transform("median")
        panel[dst] = panel[src] - med
    return panel


def _add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily pct rank of selected features."""
    for c in RANKED_FEATURES:
        if c not in panel.columns:
            continue
        panel[f"{c}_rank"] = panel.groupby("date")[c].rank(method="average", pct=True)
    return panel


def build_features_pro(prices: pd.DataFrame, index_df: pd.DataFrame,
                        industries: pd.DataFrame | None) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["stock_code"] = p["stock_code"].astype(str).str.zfill(6)
    p = p.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # Per-stock features
    p = p.groupby("stock_code", group_keys=False).apply(_per_stock).reset_index(drop=True)

    # Beta / idio vol vs index
    p = _add_beta_and_idio(p, index_df)

    # Merge industry
    if industries is not None:
        ind = industries.copy()
        ind["stock_code"] = ind["stock_code"].astype(str).str.zfill(6)
        p = p.merge(ind[["stock_code", "industry"]], on="stock_code", how="left")

    # Industry-neutral residuals
    p = _add_industry_neutral(p)

    # Cross-sectional ranks (last so they see all base features)
    p = _add_cross_sectional_ranks(p)

    return p


def cross_sectional_winsorize_pro(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """MAD winsorize at ±3 std per date for non-rank features only."""
    p = panel.copy()
    for c in cols:
        if c not in p.columns or c.endswith("_rank"):
            continue
        med = p.groupby("date")[c].transform("median")
        mad = p.groupby("date")[c].transform(
            lambda x: (x - x.median()).abs().median()
        )
        std_proxy = mad * 1.4826
        upper = med + 3 * std_proxy
        lower = med - 3 * std_proxy
        p[c] = p[c].clip(lower=lower, upper=upper)
    return p
