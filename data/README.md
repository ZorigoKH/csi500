# Inputs

Supply these four files separately. This repository does not include downloaded
market data, course datasets or credentials.

| File | Required columns |
| --- | --- |
| `prices.parquet` | `date`, `stock_code`, `open`, `high`, `low`, `close`, `volume`, `amount`, `turnover` |
| `index.parquet` | `date`, `close` |
| `industries.csv` | `stock_code`, `industry` |
| `constituents.csv` | `stock_code` |

Use one price row per stock and trading date, and one index row per trading date.
Stock codes are read as strings and zero-padded to six characters. Dates must be
parseable by pandas. Prices must be positive and use a consistent adjustment
convention; the course used forward-adjusted stock prices. Volume and amount
should be nonnegative. The original input provider's turnover units must be
preserved for historical reproduction.

The source course repository documents its supplied snapshot and download
process: [NYUSH-ML/ml-competition-sp26](https://github.com/NYUSH-ML/ml-competition-sp26).
Industry metadata was obtained separately during the original project. Obtain
and use inputs under their applicable terms; a current constituent or industry
snapshot is not a point-in-time reconstruction of an earlier universe.

Allow at least 120 trading dates for feature warmup, then enough additional
history to fit the rolling model. Missing required feature values remove rows
from training and prediction. An entirely missing input such as turnover can
therefore leave no usable rows.

The original CLI prediction date was 8 May 2026. The 30-window archived results
used earlier historical dates. Without the same historical input snapshots,
the summary of saved results is reproducible but exact fitted portfolios are
not guaranteed to match.
