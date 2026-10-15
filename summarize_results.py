"""Summarize the saved folds without market data or third-party packages."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def summarize(path: Path) -> dict[str, float | int]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [float(row["pro"]) for row in rows]
    if len(values) < 2:
        raise ValueError("At least two windows are required")
    remaining = values.copy()
    remaining.remove(max(values))
    return {
        "windows": len(values),
        "mean_excess_pp": mean(values),
        "median_excess_pp": median(values),
        "positive_windows": sum(value > 0 for value in values),
        "win_rate_percent": sum(value > 0 for value in values) / len(values) * 100,
        "worst_excess_pp": min(values),
        "best_excess_pp": max(values),
        "mean_without_best_pp": mean(remaining),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).parent / "results/walk_forward.csv")
    stats = summarize(parser.parse_args().csv)
    print("Historical backtest output; excess returns are percentage points.")
    for name, value in stats.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
