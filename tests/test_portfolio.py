import numpy as np
import pandas as pd

from portfolio import filter_suspended, select_with_industry_cap


def test_diversification_changes_the_top_ranked_selection():
    codes = [str(i).zfill(6) for i in range(80)]
    scores = pd.Series(np.arange(80, 0, -1), index=codes)
    industries = {code: f"industry-{i // 8}" for i, code in enumerate(codes)}
    picked = select_with_industry_cap(scores, industries, 50, 6)
    counts = pd.Series([industries[code] for code in picked]).value_counts()
    assert len(picked) == len(set(picked)) == 50
    assert counts.max() <= 6
    assert codes[6] not in picked  # A high-scoring seventh name is replaced by diversification.


def test_original_fallback_relaxes_an_infeasible_industry_cap():
    scores = pd.Series([3, 2, 1], index=["000001", "000002", "000003"])
    industries = {code: "same-industry" for code in scores.index}
    assert len(select_with_industry_cap(scores, industries, 3, 1)) == 3


def test_future_volume_cannot_change_suspension_filter():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-05-06", "2026-05-07", "2026-05-08", "2026-05-11"] * 2),
        "stock_code": ["000001"] * 4 + ["000002"] * 4,
        "volume": [100, 100, 100, 0, 100, 0, 100, 100],
    })
    assert filter_suspended(prices, ["000001", "000002"], pd.Timestamp("2026-05-08")) == ["000001"]
