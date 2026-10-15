"""Run the full training CLI on artificial data; no market data is downloaded."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def synthetic_inputs(n_stocks: int = 80, n_dates: int = 280):
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    market = rng.normal(0.0002, 0.008, n_dates)
    index = pd.DataFrame({"date": dates, "close": 1000 * np.exp(market.cumsum())})
    rows = []
    codes = [str(number + 1).zfill(6) for number in range(n_stocks)]
    for number, code in enumerate(codes):
        changes = (0.5 + number / n_stocks) * market + rng.normal(0, 0.012, n_dates)
        close = (15 + number / 3) * np.exp(changes.cumsum())
        opened = close * np.exp(rng.normal(0, 0.003, n_dates))
        volume = rng.lognormal(12, 0.65, n_dates)
        rows.append(pd.DataFrame({
            "date": dates, "stock_code": code, "open": opened, "close": close,
            "high": np.maximum(opened, close) * (1 + rng.uniform(0.001, 0.02, n_dates)),
            "low": np.minimum(opened, close) * (1 - rng.uniform(0.001, 0.02, n_dates)),
            "volume": volume, "amount": volume * close,
            "turnover": rng.uniform(0.2, 4, n_dates),
        }))
    industries = pd.DataFrame({
        "stock_code": codes, "industry": [f"group-{number % 16}" for number in range(n_stocks)],
    })
    return pd.concat(rows, ignore_index=True), index, industries


def main() -> None:
    prices, index, industries = synthetic_inputs()
    with tempfile.TemporaryDirectory(prefix="csi500-demo-") as tmp:
        root = Path(tmp)
        prices.to_parquet(root / "prices.parquet", index=False)
        index.to_parquet(root / "index.parquet", index=False)
        industries.to_csv(root / "industries.csv", index=False)
        industries[["stock_code"]].to_csv(root / "constituents.csv", index=False)
        out = root / "portfolio.csv"
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("train.py")),
            "--data-dir", str(root), "--as-of", prices.date.max().strftime("%Y%m%d"),
            "--out", str(out),
        ], check=True)
        portfolio = pd.read_csv(out, dtype={"stock_code": str})
        assert len(portfolio) == 50 and portfolio.stock_code.is_unique
        assert set(portfolio.stock_code) <= set(industries.stock_code)
        assert np.isclose(portfolio.weight.sum(), 1.0)
        assert np.allclose(portfolio.weight, 0.02)
        joined = portfolio.merge(industries, on="stock_code", validate="one_to_one")
        assert joined.groupby("industry").size().max() <= 6
        print("Synthetic end-to-end check passed: 50 names, valid weights and industry counts.")


if __name__ == "__main__":
    main()
