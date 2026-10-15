"""Phase 2-Pro — XGB + LGB rank-averaged ensemble on the 61-feature stack."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    build_features_pro, cross_sectional_winsorize_pro,
    ALL_FEATURES, TARGET_COL,
)
from portfolio import (
    select_with_industry_cap, filter_suspended,
    EMBARGO_DAYS,
)

DATA_DIR = Path(__file__).parent / "data"
SUBMISSIONS_DIR = Path(__file__).parent / "submissions"

TRAIN_LOOKBACK_DAYS = 365
SEEDS_LGB = [42, 123, 456]
SEED_XGB = 42

LGB_PARAMS = dict(
    objective="regression",
    boosting_type="gbdt",
    n_estimators=200,
    learning_rate=0.063,
    num_leaves=31,
    max_depth=9,
    min_child_samples=180,
    subsample=0.73,
    subsample_freq=1,
    colsample_bytree=0.91,
    reg_alpha=0.02,
    reg_lambda=3.87,
    n_jobs=-1,
    verbose=-1,
)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.057,
    max_depth=7,
    min_child_weight=26,
    subsample=0.77,
    colsample_bytree=0.78,
    reg_alpha=0.32,
    reg_lambda=0.10,
    tree_method="hist",
    n_jobs=-1,
)


def train_xgb_lgb_rank_ensemble(
    train_panel: pd.DataFrame,
    pred_panel: pd.DataFrame,
    feat_cols: list[str],
) -> pd.Series:
    n_pred = len(pred_panel)

    xgb_params = {**XGB_PARAMS, "random_state": SEED_XGB}
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(train_panel[feat_cols], train_panel[TARGET_COL])
    xgb_pred = xgb_model.predict(pred_panel[feat_cols])
    xgb_rank = pd.Series(xgb_pred).rank(method="average", pct=True).values

    lgb_rank_sum = np.zeros(n_pred)
    for s in SEEDS_LGB:
        lgb_params = {**LGB_PARAMS, "random_state": s, "seed": s,
                      "feature_fraction_seed": s, "bagging_seed": s, "data_random_seed": s}
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(train_panel[feat_cols], train_panel[TARGET_COL])
        lp = m.predict(pred_panel[feat_cols])
        lgb_rank_sum += pd.Series(lp).rank(method="average", pct=True).values
    lgb_rank = lgb_rank_sum / len(SEEDS_LGB)

    blended = 0.5 * xgb_rank + 0.5 * lgb_rank

    return pd.Series(blended,
                     index=pred_panel["stock_code"].astype(str).str.zfill(6).values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default="20260508")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", default=str(SUBMISSIONS_DIR / "window2_pro.csv"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-per-industry", type=int, default=6)
    parser.add_argument("--train-days", type=int, default=TRAIN_LOOKBACK_DAYS)
    args = parser.parse_args()
    data_dir = args.data_dir
    if args.top_k < 30 or args.max_per_industry < 1 or args.train_days < 1:
        parser.error("top-k must be at least 30; industry cap and train-days must be positive")
    required = ["prices.parquet", "index.parquet", "industries.csv", "constituents.csv"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        parser.error("Missing input files: " + ", ".join(missing) + ". See data/README.md.")

    print("=" * 70)
    print("PHASE 2-PRO  — XGB+LGB ensemble, expanded feature stack")
    print(f"  features={len(ALL_FEATURES)}  top_k={args.top_k}  "
          f"max_per_ind={args.max_per_industry}  train_days={args.train_days}")
    print("=" * 70)

    print("\n[1/5] Loading data ...")
    prices = pd.read_parquet(data_dir / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    index_df = pd.read_parquet(data_dir / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])
    industries = pd.read_csv(data_dir / "industries.csv", dtype={"stock_code": str})
    industries["stock_code"] = industries["stock_code"].astype(str).str.zfill(6)
    ind_map = dict(zip(industries["stock_code"], industries["industry"]))
    constituents = pd.read_csv(data_dir / "constituents.csv", dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].astype(str).str.zfill(6)
    universe = set(constituents["stock_code"].tolist())

    print(f"\n[2/5] Building {len(ALL_FEATURES)}-feature panel ...")
    panel = build_features_pro(prices, index_df, industries)
    panel = cross_sectional_winsorize_pro(panel, ALL_FEATURES)
    feat_cols = [c for c in ALL_FEATURES if c in panel.columns]
    print(f"  panel rows: {len(panel):,}; features available: {len(feat_cols)}")

    as_of_ts = pd.Timestamp(args.as_of)
    trading_dates = sorted(panel["date"].unique())
    dates_le = [d for d in trading_dates if d <= as_of_ts]
    if len(dates_le) <= EMBARGO_DAYS:
        raise ValueError("Not enough trading history before --as-of")
    pred_date = dates_le[-1]
    train_cutoff = dates_le[-(EMBARGO_DAYS + 1)]
    train_start = pred_date - pd.Timedelta(days=args.train_days)

    train_panel = (
        panel[(panel["date"] >= train_start) & (panel["date"] <= train_cutoff)]
        .dropna(subset=feat_cols + [TARGET_COL]).copy()
    )
    pred_panel = panel[panel["date"] == pred_date].dropna(subset=feat_cols).copy()
    pred_panel = pred_panel[pred_panel["stock_code"].isin(universe)]

    if train_panel.empty or pred_panel.empty:
        raise ValueError("No usable training/prediction rows; inspect data coverage and missing features")

    print(f"\n[3/5] Training XGB + LGB ensemble (XGB×1 + LGB×{len(SEEDS_LGB)}) ...")
    print(f"  train: {len(train_panel):,} rows ({train_panel['date'].min().date()} -> {train_panel['date'].max().date()})")
    print(f"  predict as of {pred_date.date()} on {len(pred_panel)} stocks")

    scores = train_xgb_lgb_rank_ensemble(train_panel, pred_panel, feat_cols)

    print("\n[4/5] Building portfolio ...")
    candidates = filter_suspended(prices, scores.index.tolist(), pred_date)
    scores = scores.loc[candidates]
    picked = select_with_industry_cap(scores, ind_map, args.top_k, args.max_per_industry)
    if len(picked) < 30:
        raise ValueError("Fewer than 30 eligible stocks remain after filtering")
    weights = pd.Series(1.0 / len(picked), index=picked)

    star = sum(1 for c in picked if c.startswith("688"))
    print(f"  picked {len(picked)} stocks; STAR exposure: {star}/{len(picked)} ({star/len(picked)*100:.0f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stock_code": weights.index, "weight": weights.values}).to_csv(
        out_path, index=False
    )
    print(f"\n[5/5] Wrote {out_path}")
    print(f"  stocks: {len(weights)}  sum: {weights.sum():.6f}  max: {weights.max():.4f}")


if __name__ == "__main__":
    main()
