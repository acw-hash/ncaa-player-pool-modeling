# NCAA Tournament Player Pool — Draft Board Generator

A statistical draft board generator for NCAA Tournament player pool competitions, where participants draft individual players (not teams) and score points based on tournament performance. Built and iterated over 4 tournament seasons (2022–2025).

**Backtest results: 100th percentile (2024) · 93rd percentile (2025)**

---

## How It Works

You provide a pre-tournament cheat sheet (Excel file with player names, teams, seeds, and PPG). The model outputs a ranked draft board by predicting each player's expected tournament point total using:

- A per-game scoring model (Ridge regression on PPG + Seed)
- Historical seed advancement probabilities (40 years of data)
- Team-quality adjustments via BARTHAG and three-point rate
- Monte Carlo simulation with team-correlated advancement
- Two optimization strategies: concentrated (high upside) vs. diversified

The output is a formatted Excel file with your draft board and model diagnostics. Everything else (EDA, optimization comparisons, simulation stats) prints to the console.

---

## Quickstart

### Requirements

```bash
pip install pandas numpy openpyxl scikit-learn
```

### Run

```bash
python draft_board_v5.py your_cheat_sheet.xlsx
```

With a custom output name:
```bash
python draft_board_v5.py your_cheat_sheet.xlsx --output my_draft_board.xlsx
```

Adjust risk tolerance (default 0.0):
```bash
python draft_board_v5.py your_cheat_sheet.xlsx --risk 0.3
```

### Input Format

Your Excel file needs at minimum:

| Column | Notes |
|--------|-------|
| `Player` (or `Name`) | Player name |
| `Team` (or `School`) | Team name |
| `Seed` | Tournament seed (1–16) |
| `PPG` (or `PTS`, `Points`) | Regular season points per game |
| `Region` | Optional but recommended |
| `Rank` | Optional — cheat sheet's pre-tournament ranking |

Optional team-level columns that improve accuracy: `BARTHAG`, `ThreeRate`, `AdjEM`, `AdjO`, `Pace`, `Usage`. The model degrades gracefully if these are missing, and is fully backward-compatible with older input formats.

### Output

An Excel file with two sheets:
- **Draft Board** — Full ranked player list, top 8 highlighted
- **Model Info** — Model parameters, formulas, and backtest performance

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v1** | Baseline model: per-game linear regression + seed advancement lookup |
| **v2** | Ridge regression, basic Monte Carlo simulation |
| **v3** | Cross-validated Ridge, corrected Monte Carlo, Start-5 optimization |
| **v4** | Team-correlated advancement (players on same team advance together), per-seed-bucket shrinkage, win-probability pool simulation |
| **v5** *(current)* | BARTHAG advancement adjustment (replaces AdjEM as primary team-quality predictor), three-point rate (3PR) feature — the only statistically validated advancement predictor beyond seed across 10 analytical methods |

The `draft_board.py` (no version suffix) is a clean, standalone v1-style script that runs with no optional columns and minimal dependencies — good for a quick board without advanced metrics.

---

## Repository Contents

```
draft_board_v5.py              # Current model (recommended)
draft_board.py                 # Lightweight standalone version
v3_fitted_params.json          # Cached model parameters
v5_fitted_params.json
ncaa_player_pool_*.xlsx        # Historical cheat sheets (2022–2025)
NCAA_Pool_*_Report*.md         # Analysis reports by version
enhanced_model_analysis.py     # Model fitting and validation scripts
v3_model_fitting.py
v5_model_fitting.py
exhaustive_advancement_analysis.py
full_feature_analysis.py
```

---

## Key Findings

- **Seed is the dominant variable** — 1-seeds average 4.05 expected games vs. 1.44 for 10-seeds. Games played, not PPG, drives total points.
- **PPG correlates moderately with per-game scoring** (r = 0.45) but weakly with total points (r = 0.16).
- **Three-point rate** is the only team metric with a statistically reliable bootstrap CI for predicting advancement beyond seed.
- The optimal 8-player lineup is robust: stress-testing seed probabilities across 20 perturbation scenarios changed 0–1 lineup slots.
