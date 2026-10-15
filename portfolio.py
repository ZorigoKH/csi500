"""Portfolio helpers from the submitted Phase 2 model."""
from __future__ import annotations

import pandas as pd

EMBARGO_DAYS = 5


def select_with_industry_cap(
    scores: pd.Series,
    industries_map: dict[str, str],
    top_k: int,
    max_per_industry: int,
) -> list[str]:
    sorted_codes = scores.sort_values(ascending=False).index.tolist()
    picked: list[str] = []
    industry_counts: dict[str, int] = {}

    for code in sorted_codes:
        if len(picked) >= top_k:
            break
        ind = industries_map.get(code, "_unknown")
        if industry_counts.get(ind, 0) >= max_per_industry:
            continue
        picked.append(code)
        industry_counts[ind] = industry_counts.get(ind, 0) + 1

    if len(picked) < top_k:
        for code in sorted_codes:
            if len(picked) >= top_k:
                break
            if code in picked:
                continue
            picked.append(code)

    return picked[:top_k]


def filter_suspended(prices: pd.DataFrame, candidates: list[str], as_of: pd.Timestamp) -> list[str]:
    recent = prices[prices["date"] <= as_of]
    last_dates = sorted(recent["date"].unique())[-3:]
    sub = recent[recent["date"].isin(last_dates)]
    bad = sub.groupby("stock_code")["volume"].apply(
        lambda v: (v == 0).any() or v.isna().any()
    )
    bad_codes = set(bad[bad].index.astype(str).str.zfill(6).tolist())
    return [c for c in candidates if c not in bad_codes]
