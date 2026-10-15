# csi500

An XGBoost + LightGBM model for selecting stocks from the CSI500 universe,
built for NYU Shanghai's Spring 2026 machine-learning competition.

The final model combines 61 price, volume, risk and industry-relative features.
It averages the prediction ranks of one XGBoost model and three LightGBM
models, then selects 50 stocks with equal weights and an industry-diversification
rule. The prediction target is the next five trading days' return.

## Results

These are **historical backtest results**, from the saved 30-window output in
[`results/walk_forward.csv`](results/walk_forward.csv). They are not live returns.

| Measure | Final model |
| --- | ---: |
| Mean return above the index per five-day window | +1.006 percentage points |
| Median return above the index | +0.559 percentage points |
| Windows above the index | 21 / 30 (70%) |
| Worst window relative to the index | -2.582 percentage points |
| Best window relative to the index | +12.930 percentage points |

The first submitted model underperformed the index by 2.35 percentage points
in its live evaluation. That failure led to the final model's redesign.
The saved report does not establish the final model's second live result.
The development history and limits are explained in [REPORT.md](REPORT.md).

Recompute the table from the archived output, with Python's standard library:

```bash
python summarize_results.py
```

## Run the model

Python 3.12 is the verified environment. Create a virtual environment, then:

```bash
pip install -r requirements.txt

# Place your separately obtained market inputs in data/ first.
python train.py --as-of 20260508 --data-dir data --out submissions/portfolio.csv
```

On macOS, LightGBM also needs the OpenMP runtime (for example, `brew install libomp`).

The four required inputs and their schemas are described in
[`data/README.md`](data/README.md). Market datasets and course-provided starter
files are not redistributed here. Reproducing the original portfolio requires
the original input snapshot; revised data can produce different predictions.

To exercise feature construction, all four fitted models, and portfolio output
without downloading market data:

```bash
python demo.py
```

The demo creates artificial data in a temporary folder. Its output is a software
check, not a performance estimate.

## What is here

- `features.py`: the final feature engineering and per-date winsorization.
- `train.py`: the submitted model settings, fixed seeds, five-trading-day training
  embargo, rank averaging, and command-line entry point.
- `portfolio.py`: suspension filtering and the original industry-diversification rule.
- `results/`: the saved window-level returns and their column definitions.
- `tests/`: feature timing, portfolio constraints, and archived-result checks.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Origin and assistance

This is a focused portfolio extraction of my implementation for the
[original course competition](https://github.com/NYUSH-ML/ml-competition-sp26),
not the course starter repository. The task, universe and evaluation convention
come from the course. The published code is the added final-model implementation;
the staff baseline, scorer, validator, and supplied market data are excluded.

The original work used Claude for coding, feature ideas, experiments and report
drafting, as acknowledged in the course report. Codex helped prepare this portfolio
version, its documentation and checks. No language-model predictions or stock
recommendations enter the model.
