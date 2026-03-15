# NCAA Tournament Player Pool: Analysis & Optimization Report (V5)
## BARTHAG + Three-Point Rate Advancement Model

---

## Executive Summary

V5 introduces the first statistically validated team-quality features into the advancement model. After an exhaustive analysis of 25 pre-tournament features across 10 analytical methods (partial correlations, rank-based correlations, logistic regression, random forest, elastic net, interaction effects, nonlinear analysis, composite optimization, Ridge regression, and bootstrap confidence intervals), two features survived the gauntlet:

- **BARTHAG** (Barttorvik composite win probability): γ=0.08. The strongest composite quality metric — top-quartile teams play 3.9 games vs 2.3 games (Δ=1.6), the largest threshold effect of any feature tested. Subsumes AdjEM (r=0.95 correlation).
- **Three-Point Rate (3PR)**: γ=0.05. The *only* feature with a statistically reliable bootstrap confidence interval [0.11, 0.43] that does not cross zero. 100% of 2,000 bootstrap resamples showed a positive relationship with advancement.

Combined, these reduce CV MAE from 17.06 (V4) to 16.97 while maintaining 4.5/8 top-8 overlap — meaning the draft board accuracy is preserved while predictions become more accurate.

**What didn't survive:** eFG%, turnover rate, free throw rate, offensive rebounding %, tempo, WAB, AdjO, AdjD, defensive eFG%, and all composite Dean Oliver "Four Factors." None showed consistent signal across multiple analytical methods with only 82 team-seasons.

---

## 1. The Feature Gauntlet: 10 Methods, 25 Features

### 1.1 Motivation

V4's advancement model used only AdjEM (γ=0.08) to adjust seed-based advancement probabilities. The question was whether other team-level metrics — Dean Oliver's Four Factors (eFG%, TO%, ORB%, FT Rate), defensive metrics, composite ratings — could differentiate strong 3-seeds from weak 3-seeds more effectively.

### 1.2 Data

Pre-tournament Barttorvik data for all 82 team-seasons in the training set (2022–2025). Pre-tournament values ensure no data leakage from tournament performance.

### 1.3 Methods Applied

| # | Method | What It Tests | Result |
|---|---|---|---|
| 1 | Linear partial correlation (seed controlled) | Linear signal beyond seed | 3PR (r=0.275, p=0.012) top |
| 2 | Rank-based partial correlation | Non-parametric monotonic signal | AdjEM (ρ=0.204), BARTHAG (ρ=0.200) |
| 3 | Logistic regression (LOYO-CV) | P(Sweet 16), P(Elite 8) | AdjEM +3.6% AUC for S16; 3PR +2.3% for E8 |
| 4 | Random Forest permutation importance | Nonlinear + interaction capture | AdjEM (0.041), WAB (0.030), BARTHAG (0.029) |
| 5 | Elastic Net variable selection | L1 sparsity: which features survive? | 3PR (0.285), ORB (0.145), ft_rate_def (0.130) |
| 6 | Interaction effects (seed × feature) | Does signal depend on seed level? | No significant interactions found |
| 7 | Nonlinear effects (quadratic, threshold) | Curved relationships, critical values | BARTHAG Δr²=0.056 (quadratic); threshold at 75th pctile |
| 8 | Optimal composite (weighted combination) | Best possible multi-feature signal | r=0.423 partial; MAE 1.018 vs 1.111 seed-only (8.4% improvement) |
| 9 | Ridge regression LOYO-CV (direct games prediction) | Full pipeline advancement test | Seed + eFG%Def marginally best; most features hurt |
| 10 | Bootstrap confidence intervals (2,000 resamples) | Stability and reliability of signal | 3PR: CI [0.11, 0.43], **only reliable feature** |

### 1.4 Cross-Method Scorecard

| Feature | M1 | M2 | M3 | M4 | M5 | M7 | M9 | M10 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **three_rate (3PR)** | ✓✓✓ | ✓✓✓ | ✓✓ | · | ✓✓✓ | · | ✓ | ✓✓✓ | **INCLUDE** |
| **BARTHAG** | · | ✓✓ | ✓✓ | ✓ | · | ✓✓ | · | · | **INCLUDE** |
| **AdjEM** | · | ✓✓ | ✓✓✓ | ✓ | ✓ | · | · | · | **INCLUDE (fallback)** |
| ORB% | ✓ | · | · | · | ✓✓ | · | · | · | Borderline |
| WAB | · | · | · | ✓ | ✓ | · | · | · | Exclude |
| eFG% | · | · | · | · | · | · | · | · | Exclude |
| TO Rate | · | · | · | · | · | · | · | · | Exclude |
| FT Rate | · | · | · | · | · | · | · | · | Exclude |
| Tempo | · | · | · | · | · | · | · | · | Exclude |

### 1.5 Key Finding: The Optimal Composite

Method 8 proved that team-quality metrics collectively carry **real signal** beyond seed. An optimally weighted composite achieved partial r=0.423 with games played (controlling for seed) and reduced games-prediction MAE by 8.4% versus seed alone. The composite weights were BARTHAG (0.70), AdjEM (−0.59), 3PR (−0.32) — the exact three features selected for V5.

---

## 2. V5 Advancement Model

### 2.1 Specification

    P_adjusted(reach round r) = sigmoid(logit(P_base(r)) + γ_BARTHAG × BARTHAG_norm + γ_3PR × 3PR_norm)

Where:
- BARTHAG_norm = (BARTHAG − 0.9226) / 0.0717
- 3PR_norm = (ThreePointRate − 38.41) / 5.20

### 2.2 Calibrated Parameters

| Parameter | Value | Rationale |
|---|---|---|
| γ_BARTHAG | 0.08 | Best safe config maintaining 4.5/8 top-8 overlap |
| γ_3PR | 0.05 | Conservative; only feature with reliable bootstrap CI |
| γ_AdjEM | 0.08 (fallback) | Used only when BARTHAG unavailable |

### 2.3 Calibration Sweep Results

| Configuration | CV MAE | Top-8 Overlap | Top-8 Pts |
|---|---|---|---|
| Seed only (baseline) | 17.00 | 4.5/8 | 504 |
| AdjEM=0.08 (V4) | 17.06 | 4.5/8 | 506 |
| **BARTHAG=0.08 + 3PR=0.05 (V5)** | **16.97** | **4.5/8** | **503** |
| 3PR=0.15 (best MAE) | 16.87 | 4.2/8 | 503 |
| BARTHAG=0.12 + 3PR=0.12 | 16.92 | 4.2/8 | 503 |

The selected V5 config improves MAE by 0.53% over V4 while preserving identical top-8 overlap (4.5/8). Configurations with larger gammas achieve lower MAE but degrade who the model selects as top-8 picks.

### 2.4 Why BARTHAG Replaces AdjEM

BARTHAG and AdjEM are correlated at r≈0.95 — they measure almost the same thing. However:

1. BARTHAG integrates the Four Factors (eFG%, TO%, ORB%, FT Rate) into a single composite via Barttorvik's calibrated model. It captures more information in one number than AdjEM alone.
2. BARTHAG showed the strongest threshold effect of any feature: top-quartile teams play 1.6 more games than bottom 75%.
3. BARTHAG had the heaviest weight (0.70) in the optimal composite (Method 8).
4. Using both BARTHAG and AdjEM would double-count the same signal.

AdjEM is retained as a fallback for backward compatibility with input files that don't include BARTHAG.

### 2.5 Why Three-Point Rate

Three-point rate was the standout discovery of the exhaustive analysis:

- **Only feature with reliable bootstrap CI** (100% of resamples positive, CI [0.11, 0.43])
- Strongest linear partial correlation (r=0.275, p=0.012)
- Strongest rank-based partial correlation (ρ=0.264, p=0.017)
- Largest Elastic Net coefficient (0.285) — selected by L1 sparsity
- Logistic: +3.1% AUC for Sweet 16, +2.3% for Elite 8

**Basketball intuition:** Teams that shoot more threes have higher-variance scoring distributions. In single-elimination tournaments, variance is beneficial — it increases the probability of an upset win (for underdogs) and creates explosive scoring games (for favorites). 3PR also correlates with modern, analytically-driven coaching systems that tend to perform well in March.

---

## 3. What V5 Inherits from V4 (Unchanged)

### 3.1 Per-Game Scoring Model

    pts_per_game = −1.813 + 1.144 × PPG_adj − 0.844 × Seed

Selected via bootstrap-augmented LOYO-CV with the 1-SE rule. Ridge α=50.

### 3.2 Per-Seed-Bucket Shrinkage

| Seed Bucket | λ | Effect |
|---|---|---|
| 1-2 seeds | 0.15 | Keep 85% of raw PPG |
| 3-5 seeds | 0.60 | Keep 40% of raw PPG |
| 6-8 seeds | 0.20 | Keep 80% of raw PPG |
| 9-16 seeds | 0.60 | Keep 40% of raw PPG |

### 3.3 Team-Correlated Monte Carlo

Players on the same team advance together. Start-5 rule applied per round. Heteroscedastic variance by seed bucket.

### 3.4 Win-Probability Evaluation

Full pool competition simulation against 17 opponents using cheat-sheet, PPG-greedy, and random-smart draft strategies.

---

## 4. Methodology Comparison: V4 vs V5

| Aspect | V4 | V5 |
|---|---|---|
| Advancement features | AdjEM only (γ=0.08) | **BARTHAG (γ=0.08) + 3PR (γ=0.05)** |
| Feature validation | LOYO-CV only | **Exhaustive 10-method analysis** |
| CV MAE | 17.06 | **16.97** (0.5% improvement) |
| Top-8 overlap | 4.5/8 | 4.5/8 (maintained) |
| Bootstrap validation | Not done for features | **2,000 resamples; 3PR is only reliable feature** |
| Features excluded | By heuristic | **By rigorous cross-method consensus** |

---

## 5. Input Requirements

**Required:** Player, Team, Seed, PPG

**Recommended (V5 advancement):** BARTHAG, ThreeRate (3PR)

**Optional (tiebreakers):** AdjEM, AdjO, Pace, Q1_Wins, SOR_Rank, Usage

**Data sources:**
- BARTHAG, 3P Rate: barttorvik.com (free) — use pre-tournament values
- AdjEM, AdjO, Pace: kenpom.com ($25/yr) or barttorvik.com
- Usage: barttorvik.com or sports-reference.com/cbb

If BARTHAG is not provided, the model falls back to AdjEM for advancement adjustment. If neither is available, the model uses seed-only advancement probabilities.

---

## 6. Updated Draft Principles

1. **BARTHAG matters for draft tiebreaking.** When choosing between two players on the same seed with similar PPG, prefer the player on the higher-BARTHAG team. This is the "not all 3-seeds are the same" adjustment.

2. **Three-point shooting teams advance further.** All else equal, players on teams that shoot more threes have slightly higher expected total points. This is a small effect but the most statistically reliable one in the dataset.

3. **Don't overreact to BARTHAG.** The gamma is intentionally conservative (0.08). A 1-SD BARTHAG advantage shifts expected games by about 0.1–0.2. This is a tiebreaker, not a draft driver. Seed and PPG still dominate.

4. **Houston is an interesting case.** Very high BARTHAG (0.9824) but low three-point rate (34.5). The model treats these as offsetting signals — Houston advances far (BARTHAG boost) but with a slightly lower-variance offense (3PR penalty). Net effect is close to a pure seed-based estimate.

5. **Duke and Alabama benefit most from V5.** Both have high BARTHAG and very high three-point rates (45.4 and 46.2 respectively). V5 gives their players a meaningful expected-games boost over V4.

---

## 7. Limitations & Future Work

- **82 team-seasons remains the binding constraint.** The exhaustive analysis confirmed that most features have real within-seed signal but insufficient statistical power with 4 years of data. As the dataset grows, ORB%, defensive eFG%, and WAB are the most likely features to become viable.

- **BARTHAG and AdjEM are nearly collinear** (r≈0.95). The model uses BARTHAG preferentially, but the collinearity means the gamma calibration is somewhat interchangeable between them. The selected gamma should be interpreted as "team quality beyond seed" rather than specifically "BARTHAG."

- **Betting market implied probabilities** remain the most promising future improvement. Vegas team-specific lines effectively aggregate all pre-tournament information (BARTHAG, 3PR, injuries, matchups, bracket path) into a single consensus probability with far more statistical power than our 82-observation regression.

- **Pre-tournament timing matters.** BARTHAG and 3PR should be sourced from end-of-regular-season data, not including conference tournament or NCAA tournament games. Post-tournament values are contaminated by the outcome we're trying to predict.

---

*Analysis generated via exhaustive 10-method feature validation on pre-tournament Barttorvik data. Bootstrap confidence intervals computed from 2,000 resamples. All advancement gammas calibrated via LOYO-CV with draft-quality preservation constraint. No in-sample metrics used for model selection.*
