# CSI500: model, results and lessons

## The task and final model

The course task was to choose a long-only portfolio from a supplied CSI500
constituent list and compare its return with the index over a fixed window.

The final model uses 61 features: returns and momentum at several horizons,
volatility, intraday prices, moving averages, RSI and MACD, volume, liquidity,
turnover, market beta, industry-relative signals, and cross-sectional ranks.
The label is the next five trading days' close-to-close return.

One XGBoost regressor (seed 42) and three LightGBM regressors (seeds 42, 123 and
456) are trained from scratch on a rolling 365-calendar-day window. Five trading
days are removed from the end of the training rows so their forward labels are
known at the prediction date. LightGBM's prediction ranks are averaged, then
combined equally with XGBoost's ranks.

The pipeline chooses 50 names with equal weights. It first limits selections to
six per industry. The original implementation fills any shortfall from the
remaining ranked names, so this is a diversification preference rather than an
unconditional industry cap. It also removes stocks with zero or missing volume
in the latest three trading dates available in the input.

## What the results show

The archived output covers 30 historical prediction dates from 30 October 2025
to 28 April 2026. The final model averaged +1.006 percentage points above the
index per five-day evaluation window, with a median of +0.559. It beat the index
in 21 windows, or 70%. The worst window was -2.582 percentage points; the best
was +12.930.

That best window matters: removing it reduces the mean to +0.595 percentage
points. The full fold-level output is included so the average can be inspected,
not just quoted.

These figures are recomputed from saved output, not from a new training run.
The public extraction contains the final training pipeline and archived results;
it does not contain the complete experimental history or a claim that the
historical folds can be regenerated exactly without the original snapshots and
evaluation setup.

## What failed live

The original course report records +1.78% for the first submitted portfolio
against +4.13% for the index, a shortfall of 2.35 percentage points. Its positive
backtest did not carry into that live window.

The subsequent work compared a simpler feature set with the final XGBoost and
LightGBM ensemble, changed regularization and training lookback, and increased
diversification. The useful lesson was to inspect portfolio-level outcomes and
failure cases rather than rely on prediction scores or a positive historical
average. This is a development account, not evidence that the redesign solved
generalization.

No verified second live result is included in the saved material reviewed for
this portfolio version.

## Limits

- Many feature, model and portfolio variants were tried. The archived backtest
  influenced model selection; it is not a pristine final holdout. A confidence
  interval from those folds does not account for that search.
- Five-day evaluation windows can overlap, making fold returns dependent.
  Thirty folds are not thirty independent trials.
- Constituent and industry files are snapshots. Historical evaluation can have
  survivorship and point-in-time classification bias.
- The summary uses the original close-to-close, portfolio-minus-index convention.
  Transaction costs, slippage, market impact and a full execution simulation are
  not represented in those figures. Adjusted prices can also be revised.
- Fixed seeds improve repeatability within an environment. Different library
  versions, platforms or input snapshots can change fitted trees and weights.

## Publication notes

This short report was prepared from the saved course report and
`pro_30fold.csv`, renamed here to `results/walk_forward.csv`. The original course
report acknowledged Claude's coding and drafting assistance. This portfolio
extraction adds documentation, input checks, a synthetic demonstration and
automated checks; its model settings and ranking rule come from the submitted
Phase 2 implementation.
