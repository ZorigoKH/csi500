import numpy as np
import pandas as pd

from demo import synthetic_inputs
from features import ALL_FEATURES, build_features_pro, cross_sectional_winsorize_pro


def test_future_prices_do_not_change_past_features():
    prices, index, industries = synthetic_inputs(n_stocks=8, n_dates=190)
    cutoff = sorted(index.date)[-15]
    original = build_features_pro(prices, index, industries)
    changed_prices = prices.copy()
    changed_index = index.copy()
    future = changed_prices.date > cutoff
    changed_prices.loc[future, ["open", "high", "low", "close", "volume", "amount", "turnover"]] *= 10
    changed_index.loc[changed_index.date > cutoff, "close"] *= 4
    changed = build_features_pro(changed_prices, changed_index, industries)
    assert len(ALL_FEATURES) == 61
    keys = ["date", "stock_code"]
    before = original.loc[original.date <= cutoff, keys + ALL_FEATURES].set_index(keys).sort_index()
    after = changed.loc[changed.date <= cutoff, keys + ALL_FEATURES].set_index(keys).sort_index()
    pd.testing.assert_frame_equal(before, after)


def test_rank_features_stay_in_unit_interval_after_winsorization():
    prices, index, industries = synthetic_inputs(n_stocks=8, n_dates=190)
    panel = build_features_pro(prices, index, industries)
    panel = cross_sectional_winsorize_pro(panel, ALL_FEATURES)
    ranked = panel[[name for name in ALL_FEATURES if name.endswith("_rank")]].to_numpy()
    values = ranked[np.isfinite(ranked)]
    assert len(values) > 0
    assert np.all((values >= 0) & (values <= 1))
