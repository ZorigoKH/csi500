# Archived results

`walk_forward.csv` is copied unchanged from the original project's
`pro_30fold.csv`. It contains aggregate return statistics, not underlying stock
price records or portfolio holdings.

| Column | Meaning |
| --- | --- |
| `as_of` | Historical prediction date |
| `bench` | CSI500 index return over the evaluation window, in percent |
| `p2` | Intermediate simplified model's return minus index return, in percentage points |
| `pro` | Final model's return minus index return, in percentage points |

For example, `pro = 1.0` means one percentage point above the index. It does not
mean the portfolio's absolute return was 1%, and it is not a decimal return of 1.

The arithmetic summary can be reproduced with `python summarize_results.py`.
See the report for model-selection, overlap and data-snapshot limitations.
