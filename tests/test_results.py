from pathlib import Path
import subprocess
import sys

import pytest

from summarize_results import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_readme_numbers_match_archived_windows():
    stats = summarize(ROOT / "results/walk_forward.csv")
    assert stats["windows"] == 30
    assert stats["positive_windows"] == 21
    assert stats["mean_excess_pp"] == pytest.approx(1.0063780740129367)
    assert stats["median_excess_pp"] == pytest.approx(0.5589891810956147)
    assert stats["worst_excess_pp"] == pytest.approx(-2.582197165616941)
    assert stats["best_excess_pp"] == pytest.approx(12.930052798922349)
    assert stats["mean_without_best_pp"] == pytest.approx(0.5952168766022674)


def test_cli_explains_missing_market_inputs(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "train.py"), "--data-dir", str(tmp_path),
    ], text=True, capture_output=True)
    assert result.returncode == 2
    assert "Missing input files:" in result.stderr
    assert "data/README.md" in result.stderr
