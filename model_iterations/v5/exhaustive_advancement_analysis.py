#!/usr/bin/env python3
"""
EXHAUSTIVE ADVANCEMENT PREDICTION ANALYSIS
===========================================
Goal: Determine whether ANY variable or combination of variables predicts
how far a team advances in the NCAA tournament, beyond seed alone.

Methods:
  1. Partial correlations (linear, already done — included for completeness)
  2. Rank-based partial correlations (non-parametric)
  3. Logistic regression: P(win at least N games) conditioned on features
  4. Random Forest / decision tree importance
  5. Elastic Net variable selection (L1+L2 regularization)
  6. Interaction effects (seed × feature)
  7. Nonlinear relationships (quadratic, threshold effects)
  8. Composite index optimization (weighted combination search)
  9. LOYO-CV with advancement-only prediction (games played, not total pts)
  10. Permutation importance on advancement prediction
"""
import pandas as pd
import numpy as np
from numpy.linalg import solve, lstsq
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import minimize
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')
from enhanced_model_analysis import build_dataset

# =================================================================
# DATA SETUP
# =================================================================
print("=" * 80)
print("EXHAUSTIVE ADVANCEMENT PREDICTION ANALYSIS")
print("=" * 80)

# Parse pre-tournament Barttorvik data
all_bt = []
for year in [2022, 2023, 2024, 2025]:
    raw = pd.read_excel('/mnt/user-data/uploads/team_stats_pre_tournament.xlsx', sheet_name=str(year))
    for i, row in raw.iterrows():
        if str(row.iloc[0]).strip() == 'RK':
            raw.columns = raw.iloc[i].values
            data = raw.iloc[i+1:].copy()
            break
    data = data[pd.to_numeric(data['RK'], errors='coerce').notna()].copy()
    data['TEAM'] = data['TEAM'].astype(str).str.strip()
    for c in ['ADJOE','ADJDE','BARTHAG','EFG%','EFGD%','TOR','TORD',
              'ORB','DRB','FTR','FTRD','2P%','2P%D','3P%','3P%D',
              '3PR','3PRD','ADJ T.','WAB']:
        if c in data.columns: data[c] = pd.to_numeric(data[c], errors='coerce')
    data['year'] = year
    data['ADJEM'] = data['ADJOE'] - data['ADJDE']
    all_bt.append(data)
bt = pd.concat(all_bt, ignore_index=True)

df = build_dataset()
our = df.groupby(['year','team']).agg({'seed':'first','games_played':'first','adj_em':'first'}).reset_index()

def find_match(name, bt_names):
    if name in bt_names: return name
    aliases = {'North Carolina':'North Carolina','Iowa State':'Iowa St.',
        'Ohio State':'Ohio St.','Michigan State':'Michigan St.',
        'Kansas State':'Kansas St.','SE Missouri St':'Southeast Missouri St.',
        'Corpus Christi':'Texas A&M Corpus Chris','Boise State':'Boise St.',
        'St. Johns':"St. John's",'Colorado State':'Colorado St.',
        'San Diego State':'San Diego St.','New Mexico State':'New Mexico St.'}
    if name in aliases and aliases[name] in bt_names: return aliases[name]
    for bn in bt_names:
        if name.lower() in bn.lower() or bn.lower() in name.lower(): return bn
    parts = name.lower().split()
    for bn in bt_names:
        if parts[0] in bn.lower(): return bn
    return None

rows = []
for _, r in our.iterrows():
    yr = int(r['year'])
    bty = bt[bt['year']==yr]
    match = find_match(r['team'], set(bty['TEAM']))
    if match:
        br = bty[bty['TEAM']==match].iloc[0]
        rows.append({
            'year':yr,'team':r['team'],'seed':int(r['seed']),'games':int(r['games_played']),
            'adjem':br['ADJEM'],'adjoe':br['ADJOE'],'adjde':br['ADJDE'],
            'barthag':br['BARTHAG'],'efg':br['EFG%'],'efg_def':br['EFGD%'],
            'to_rate':br['TOR'],'to_forced':br['TORD'],'orb':br['ORB'],'drb':br['DRB'],
            'ft_rate':br['FTR'],'ft_rate_def':br['FTRD'],'two_pct':br['2P%'],
            'two_pct_def':br['2P%D'],'three_pct':br['3P%'],'three_pct_def':br['3P%D'],
            'three_rate':br['3PR'],'three_rate_def':br['3PRD'],'tempo':br['ADJ T.'],'wab':br['WAB'],
        })
m = pd.DataFrame(rows)

# Derived features
m['net_efg'] = m['efg'] - m['efg_def']
m['net_to'] = m['to_forced'] - m['to_rate']
m['net_reb'] = m['orb'] + m['drb']
m['def_composite'] = (50 - m['efg_def'])/5 + m['to_forced']/20
m['four_factors'] = m['efg']/50 - m['to_rate']/20 + m['orb']/30 + m['ft_rate']/30
m['adj_margin'] = m['adjoe'] - m['adjde']

ALL_FEATURES = ['adjem','adjoe','adjde','barthag','efg','efg_def','to_rate',
    'to_forced','orb','drb','ft_rate','ft_rate_def','two_pct','two_pct_def',
    'three_pct','three_pct_def','three_rate','three_rate_def','tempo','wab',
    'net_efg','net_to','net_reb','def_composite','four_factors']

N = len(m)
print(f"\n  Dataset: {N} team-seasons, {len(ALL_FEATURES)} features")

# =================================================================
# METHOD 1: Linear partial correlations (baseline, already done)
# =================================================================
print("\n" + "=" * 80)
print("METHOD 1: LINEAR PARTIAL CORRELATIONS (seed controlled)")
print("=" * 80)

X_s = np.column_stack([np.ones(N), m['seed'].values])
bg, _, _, _ = lstsq(X_s, m['games'].values, rcond=None)
g_resid = m['games'].values - X_s @ bg

results_m1 = []
for f in ALL_FEATURES:
    bf, _, _, _ = lstsq(X_s, m[f].values, rcond=None)
    fr = m[f].values - X_s @ bf
    r, p = pearsonr(fr, g_resid)
    results_m1.append((f, r, p))
results_m1.sort(key=lambda x: -abs(x[1]))

print(f"\n  {'Feature':>18s} {'Partial r':>10s} {'p':>8s}")
print("  " + "-" * 40)
for f, r, p in results_m1[:12]:
    sig = " ***" if abs(r)>0.20 and p<0.10 else " **" if abs(r)>0.15 else ""
    print(f"  {f:>18s} {r:10.3f} {p:8.4f}{sig}")

# =================================================================
# METHOD 2: RANK-BASED (Spearman) partial correlations
# =================================================================
print("\n" + "=" * 80)
print("METHOD 2: RANK-BASED PARTIAL CORRELATIONS (Spearman, non-parametric)")
print("Robust to outliers and non-linear monotonic relationships")
print("=" * 80)

from scipy.stats import rankdata
g_rank = rankdata(m['games'].values)
s_rank = rankdata(m['seed'].values)

# Residualize ranks on seed rank
X_sr = np.column_stack([np.ones(N), s_rank])
bg_r, _, _, _ = lstsq(X_sr, g_rank, rcond=None)
g_rank_resid = g_rank - X_sr @ bg_r

results_m2 = []
for f in ALL_FEATURES:
    f_rank = rankdata(m[f].values)
    bf_r, _, _, _ = lstsq(X_sr, f_rank, rcond=None)
    fr_r = f_rank - X_sr @ bf_r
    r, p = pearsonr(fr_r, g_rank_resid)
    results_m2.append((f, r, p))
results_m2.sort(key=lambda x: -abs(x[1]))

print(f"\n  {'Feature':>18s} {'Partial rho':>12s} {'p':>8s}")
print("  " + "-" * 42)
for f, r, p in results_m2[:12]:
    sig = " ***" if abs(r)>0.20 and p<0.10 else " **" if abs(r)>0.15 else ""
    print(f"  {f:>18s} {r:12.3f} {p:8.4f}{sig}")

# =================================================================
# METHOD 3: LOGISTIC REGRESSION — P(advance past round R)
# =================================================================
print("\n" + "=" * 80)
print("METHOD 3: LOGISTIC REGRESSION — P(win 2+ games), P(win 3+ games)")
print("Binary outcomes may reveal signal that continuous regression misses")
print("=" * 80)

def logistic_cv(X, y, alpha=1.0):
    """Simple L2-regularized logistic regression with LOYO-CV."""
    years = m['year'].values
    unique_years = np.unique(years)
    probs_all = np.zeros(len(y))
    
    for test_yr in unique_years:
        train_mask = years != test_yr
        test_mask = years == test_yr
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        
        # Fit via iteratively reweighted least squares (5 iterations)
        beta = np.zeros(X_tr.shape[1])
        for _ in range(15):
            z = X_tr @ beta
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            w = p * (1 - p) + 1e-6
            W = np.diag(w)
            pen = alpha * np.eye(X_tr.shape[1]); pen[0,0] = 0
            try:
                beta = solve(X_tr.T @ W @ X_tr + pen, X_tr.T @ (W @ z + y_tr - p))
            except:
                break
        
        z_te = X_te @ beta
        probs_all[test_mask] = 1 / (1 + np.exp(-np.clip(z_te, -30, 30)))
    
    # Evaluate: AUC approximation via concordance
    from itertools import combinations
    concordant = 0
    total = 0
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    for pi in pos:
        for ni in neg:
            total += 1
            if probs_all[pi] > probs_all[ni]:
                concordant += 1
            elif probs_all[pi] == probs_all[ni]:
                concordant += 0.5
    auc = concordant / max(total, 1)
    return auc

for threshold_name, threshold in [("2+ games (win R64)", 2), ("3+ games (Sweet 16)", 3), ("4+ games (Elite 8)", 4)]:
    y_bin = (m['games'].values >= threshold).astype(float)
    n_pos = y_bin.sum()
    if n_pos < 5 or (N - n_pos) < 5:
        continue
    
    print(f"\n  --- {threshold_name} (N+={int(n_pos)}, N-={int(N-n_pos)}) ---")
    
    # Baseline: seed only
    X_base = np.column_stack([np.ones(N), m['seed'].values])
    auc_base = logistic_cv(X_base, y_bin)
    
    print(f"  {'Model':>30s} {'AUC':>7s} {'Δ AUC':>8s}")
    print("  " + "-" * 48)
    print(f"  {'Seed only':>30s} {auc_base:7.3f}     ---")
    
    logistic_results = []
    for f in ALL_FEATURES:
        fv = m[f].values
        fm, fs = fv.mean(), max(fv.std(), 0.1)
        X_aug = np.column_stack([np.ones(N), m['seed'].values, (fv-fm)/fs])
        auc = logistic_cv(X_aug, y_bin)
        logistic_results.append((f, auc, auc - auc_base))
    
    logistic_results.sort(key=lambda x: -x[1])
    for f, auc, delta in logistic_results[:8]:
        sig = " ***" if delta > 0.02 else " **" if delta > 0.01 else ""
        print(f"  {'Seed + '+f:>30s} {auc:7.3f} {delta:+8.3f}{sig}")

# =================================================================
# METHOD 4: DECISION TREE / RANDOM FOREST IMPORTANCE
# =================================================================
print("\n" + "=" * 80)
print("METHOD 4: RANDOM FOREST FEATURE IMPORTANCE (permutation-based)")
print("Captures nonlinear relationships and interactions automatically")
print("=" * 80)

# Simple random forest from scratch (no sklearn dependency)
def build_tree(X, y, max_depth=3, min_samples=5):
    """Simple decision tree regressor."""
    if len(y) < min_samples or max_depth == 0 or y.std() < 0.01:
        return {'type': 'leaf', 'value': y.mean()}
    
    best_score = float('inf')
    best_split = None
    
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for v in vals[:-1]:
            left = y[X[:, j] <= v]
            right = y[X[:, j] > v]
            if len(left) < min_samples or len(right) < min_samples:
                continue
            score = left.var() * len(left) + right.var() * len(right)
            if score < best_score:
                best_score = score
                best_split = (j, v)
    
    if best_split is None:
        return {'type': 'leaf', 'value': y.mean()}
    
    j, v = best_split
    left_mask = X[:, j] <= v
    return {
        'type': 'node', 'feature': j, 'threshold': v,
        'left': build_tree(X[left_mask], y[left_mask], max_depth-1, min_samples),
        'right': build_tree(X[~left_mask], y[~left_mask], max_depth-1, min_samples),
    }

def predict_tree(tree, X):
    preds = np.zeros(len(X))
    for i in range(len(X)):
        node = tree
        while node['type'] != 'leaf':
            if X[i, node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        preds[i] = node['value']
    return preds

def random_forest_predict(trees, X):
    return np.mean([predict_tree(t, X) for t in trees], axis=0)

# Build feature matrix
feat_names = ['seed'] + ALL_FEATURES
X_all = np.column_stack([m[f].values for f in feat_names])

# Normalize
X_means = X_all.mean(axis=0)
X_stds = np.maximum(X_all.std(axis=0), 0.1)
X_norm = (X_all - X_means) / X_stds
y_games = m['games'].values

# LOYO-CV with bagged trees
rng = np.random.RandomState(42)
n_trees = 50

# Permutation importance via LOYO
perm_importance = defaultdict(list)

for test_year in m['year'].unique():
    train_mask = m['year'].values != test_year
    test_mask = m['year'].values == test_year
    
    X_tr, y_tr = X_norm[train_mask], y_games[train_mask]
    X_te, y_te = X_norm[test_mask], y_games[test_mask]
    
    # Build forest
    trees = []
    for t in range(n_trees):
        idx = rng.choice(len(X_tr), size=len(X_tr), replace=True)
        # Random feature subset (sqrt)
        n_feat = max(3, int(np.sqrt(X_tr.shape[1])))
        feat_idx = rng.choice(X_tr.shape[1], size=n_feat, replace=False)
        tree = build_tree(X_tr[np.ix_(idx, feat_idx)], y_tr[idx], max_depth=4, min_samples=4)
        # Store feature mapping
        trees.append((tree, feat_idx))
    
    # Predict
    def forest_predict(X):
        preds = []
        for tree, fidx in trees:
            preds.append(predict_tree(tree, X[:, fidx]))
        return np.mean(preds, axis=0)
    
    base_preds = forest_predict(X_te)
    base_mae = np.mean(np.abs(base_preds - y_te))
    
    # Permutation importance
    for j, fname in enumerate(feat_names):
        X_perm = X_te.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_preds = forest_predict(X_perm)
        perm_mae = np.mean(np.abs(perm_preds - y_te))
        perm_importance[fname].append(perm_mae - base_mae)

print(f"\n  Random Forest LOYO-CV Permutation Importance (Δ MAE when feature shuffled):")
print(f"  Higher = more important. Negative = feature hurts (overfitting).\n")

avg_imp = [(f, np.mean(v), np.std(v)) for f, v in perm_importance.items()]
avg_imp.sort(key=lambda x: -x[1])

print(f"  {'Feature':>18s} {'Avg Δ MAE':>10s} {'Std':>7s} {'Verdict':>15s}")
print("  " + "-" * 55)
for f, imp, std in avg_imp[:15]:
    verdict = "IMPORTANT" if imp > 0.05 else "marginal" if imp > 0.01 else "noise"
    print(f"  {f:>18s} {imp:10.4f} {std:7.4f}  {verdict}")

# =================================================================
# METHOD 5: ELASTIC NET VARIABLE SELECTION
# =================================================================
print("\n" + "=" * 80)
print("METHOD 5: ELASTIC NET VARIABLE SELECTION")
print("L1 penalty drives irrelevant coefficients to exactly zero")
print("=" * 80)

def elastic_net_cv(X, y, alpha, l1_ratio, years):
    """Elastic net with LOYO-CV. Returns mean MAE and coefficients."""
    unique_years = np.unique(years)
    all_maes = []
    all_betas = []
    
    for test_yr in unique_years:
        tr = years != test_yr
        te = years == test_yr
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        
        # Coordinate descent
        n, p = X_tr.shape
        beta = np.zeros(p)
        for iteration in range(200):
            for j in range(p):
                r = y_tr - X_tr @ beta + X_tr[:, j] * beta[j]
                rho = X_tr[:, j] @ r / n
                if j == 0:  # Don't penalize intercept
                    beta[j] = rho
                else:
                    l1 = alpha * l1_ratio
                    l2 = alpha * (1 - l1_ratio)
                    if rho > l1: beta[j] = (rho - l1) / (1 + l2)
                    elif rho < -l1: beta[j] = (rho + l1) / (1 + l2)
                    else: beta[j] = 0
        
        preds = X_te @ beta
        all_maes.append(np.mean(np.abs(preds - y_te)))
        all_betas.append(beta)
    
    return np.mean(all_maes), np.mean(all_betas, axis=0)

# Features: seed + all team metrics
feat_cols = ['seed'] + ALL_FEATURES
X_en = np.column_stack([np.ones(N)] + [(m[f].values - m[f].mean())/max(m[f].std(),0.1) for f in feat_cols])
y_en = m['games'].values
years_en = m['year'].values

print(f"\n  Sweeping alpha and l1_ratio...")
best_en = None
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
    for l1_ratio in [0.3, 0.5, 0.7, 0.9, 1.0]:
        mae, beta = elastic_net_cv(X_en, y_en, alpha, l1_ratio, years_en)
        if best_en is None or mae < best_en[0]:
            best_en = (mae, beta, alpha, l1_ratio)

mae, beta, alpha, l1 = best_en
print(f"\n  Best: alpha={alpha}, l1_ratio={l1}, CV MAE={mae:.3f}")
print(f"\n  Selected features (non-zero coefficients):")
feat_labels = ['intercept'] + feat_cols
for i, (name, coef) in enumerate(zip(feat_labels, beta)):
    if abs(coef) > 0.001 or name in ['intercept', 'seed']:
        marker = " <-- SELECTED" if abs(coef) > 0.001 and name not in ['intercept','seed'] else ""
        print(f"    {name:>18s}: {coef:8.4f}{marker}")

# =================================================================
# METHOD 6: INTERACTION EFFECTS (seed × feature)
# =================================================================
print("\n" + "=" * 80)
print("METHOD 6: INTERACTION EFFECTS (Seed × Feature)")
print("Tests whether a feature's effect depends on seed level")
print("=" * 80)

top_features = [f for f, r, p in results_m1[:8]]

print(f"\n  Testing: games ~ seed + feature + seed×feature")
print(f"  If interaction is significant, the feature matters MORE for some seeds.\n")

print(f"  {'Feature':>18s} {'Main β':>8s} {'Interact β':>11s} {'p(interact)':>12s} {'Interpretation':>25s}")
print("  " + "-" * 78)

for f in top_features:
    fv = (m[f].values - m[f].mean()) / max(m[f].std(), 0.1)
    sv = m['seed'].values
    interaction = fv * sv
    
    X_int = np.column_stack([np.ones(N), sv, fv, interaction])
    beta_int, _, _, _ = lstsq(X_int, m['games'].values, rcond=None)
    
    # Significance of interaction via LOYO
    # Compare model with vs without interaction
    X_noint = np.column_stack([np.ones(N), sv, fv])
    
    mae_with, mae_without = [], []
    for yr in m['year'].unique():
        tr = m['year'].values != yr
        te = m['year'].values == yr
        
        b1, _, _, _ = lstsq(X_int[tr], m['games'].values[tr], rcond=None)
        b2, _, _, _ = lstsq(X_noint[tr], m['games'].values[tr], rcond=None)
        
        p1 = X_int[te] @ b1
        p2 = X_noint[te] @ b2
        
        mae_with.append(np.mean(np.abs(p1 - m['games'].values[te])))
        mae_without.append(np.mean(np.abs(p2 - m['games'].values[te])))
    
    delta_mae = np.mean(mae_without) - np.mean(mae_with)
    
    interp = ""
    if abs(beta_int[3]) > 0.03 and delta_mae > 0.01:
        if beta_int[3] > 0:
            interp = "matters MORE for high seeds"
        else:
            interp = "matters MORE for low seeds"
    else:
        interp = "no interaction"
    
    print(f"  {f:>18s} {beta_int[2]:8.3f} {beta_int[3]:11.4f} {'Δ MAE='+f'{delta_mae:.3f}':>12s}  {interp}")

# =================================================================
# METHOD 7: NONLINEAR EFFECTS (quadratic, threshold)
# =================================================================
print("\n" + "=" * 80)
print("METHOD 7: NONLINEAR EFFECTS")
print("Tests quadratic terms and threshold effects")
print("=" * 80)

print(f"\n  A) Quadratic terms: does feature² add signal beyond linear?")
print(f"  {'Feature':>18s} {'Linear r²':>10s} {'Quad r²':>9s} {'Δ r²':>7s} {'Nonlinear?':>12s}")
print("  " + "-" * 60)

for f in top_features[:8]:
    fv = (m[f].values - m[f].mean()) / max(m[f].std(), 0.1)
    sv = m['seed'].values
    
    X_lin = np.column_stack([np.ones(N), sv, fv])
    X_quad = np.column_stack([np.ones(N), sv, fv, fv**2])
    
    b_l, _, _, _ = lstsq(X_lin, m['games'].values, rcond=None)
    b_q, _, _, _ = lstsq(X_quad, m['games'].values, rcond=None)
    
    ss_tot = np.sum((m['games'].values - m['games'].mean())**2)
    r2_l = 1 - np.sum((m['games'].values - X_lin @ b_l)**2) / ss_tot
    r2_q = 1 - np.sum((m['games'].values - X_quad @ b_q)**2) / ss_tot
    
    delta = r2_q - r2_l
    nonlin = "YES" if delta > 0.02 else "marginal" if delta > 0.005 else "no"
    print(f"  {f:>18s} {r2_l:10.4f} {r2_q:9.4f} {delta:7.4f}  {nonlin}")

print(f"\n  B) Threshold effects: is there a critical value above/below which advancement jumps?")
for f in ['adjem', 'barthag', 'wab', 'efg_def', 'orb']:
    fv = m[f].values
    best_thresh = None
    best_delta = 0
    
    for pct in [25, 33, 50, 67, 75]:
        thresh = np.percentile(fv, pct)
        above = m['games'].values[fv >= thresh].mean()
        below = m['games'].values[fv < thresh].mean()
        delta = above - below
        if abs(delta) > abs(best_delta):
            best_delta = delta
            best_thresh = (thresh, pct, above, below)
    
    if best_thresh:
        thresh, pct, above, below = best_thresh
        print(f"  {f:>12s}: best split at {pct}th pctile ({thresh:.1f}): "
              f"above={above:.1f} games, below={below:.1f} games, Δ={best_delta:+.2f}")

# =================================================================
# METHOD 8: COMPOSITE INDEX OPTIMIZATION
# =================================================================
print("\n" + "=" * 80)
print("METHOD 8: OPTIMAL COMPOSITE INDEX")
print("Find the weighted combination of features that best predicts advancement")
print("=" * 80)

# Optimize a composite: w1*feat1 + w2*feat2 + ... to maximize correlation with games_resid
candidate_feats = ['adjem', 'barthag', 'wab', 'orb', 'efg_def', 'three_rate', 'net_efg', 'def_composite']

X_cand = np.column_stack([(m[f].values - m[f].mean())/max(m[f].std(),0.1) for f in candidate_feats])

def neg_partial_corr(weights):
    """Negative partial correlation with games (controlling for seed)."""
    composite = X_cand @ weights
    # Residualize on seed
    X_s = np.column_stack([np.ones(N), m['seed'].values])
    bc, _, _, _ = lstsq(X_s, composite, rcond=None)
    c_resid = composite - X_s @ bc
    r, _ = pearsonr(c_resid, g_resid)
    return -abs(r)

# Random restarts
best_result = None
rng = np.random.RandomState(42)
for _ in range(500):
    w0 = rng.randn(len(candidate_feats))
    w0 /= np.linalg.norm(w0)
    res = minimize(neg_partial_corr, w0, method='Nelder-Mead',
                   options={'maxiter': 1000})
    if best_result is None or res.fun < best_result.fun:
        best_result = res

optimal_w = best_result.x / np.linalg.norm(best_result.x)
optimal_corr = -best_result.fun

print(f"\n  Optimal composite partial correlation with games: r = {optimal_corr:.3f}")
print(f"\n  Weights:")
for f, w in sorted(zip(candidate_feats, optimal_w), key=lambda x: -abs(x[1])):
    print(f"    {f:>16s}: {w:7.3f}")

# LOYO-CV of the optimal composite as advancement predictor
composite_vals = X_cand @ optimal_w
m['optimal_composite'] = composite_vals

print(f"\n  LOYO-CV test of optimal composite as advancement feature:")
for gamma in [0.10, 0.15, 0.20, 0.25, 0.30]:
    fold_maes = []
    for yr in m['year'].unique():
        tr = m['year'].values != yr
        te = m['year'].values == yr
        
        # Simple: predict games = f(seed, composite)
        X_tr = np.column_stack([np.ones(tr.sum()), m['seed'].values[tr], composite_vals[tr]])
        X_te = np.column_stack([np.ones(te.sum()), m['seed'].values[te], composite_vals[te]])
        b, _, _, _ = lstsq(X_tr, m['games'].values[tr], rcond=None)
        preds = X_te @ b
        fold_maes.append(np.mean(np.abs(preds - m['games'].values[te])))
    print(f"    Composite (linear, gamma~): MAE = {np.mean(fold_maes):.3f}")
    break  # Same result regardless of gamma in linear model

# Seed-only baseline for comparison
fold_maes_base = []
for yr in m['year'].unique():
    tr = m['year'].values != yr
    te = m['year'].values == yr
    X_tr = np.column_stack([np.ones(tr.sum()), m['seed'].values[tr]])
    X_te = np.column_stack([np.ones(te.sum()), m['seed'].values[te]])
    b, _, _, _ = lstsq(X_tr, m['games'].values[tr], rcond=None)
    preds = X_te @ b
    fold_maes_base.append(np.mean(np.abs(preds - m['games'].values[te])))
print(f"    Seed only (baseline):       MAE = {np.mean(fold_maes_base):.3f}")

# =================================================================
# METHOD 9: FULL LOYO-CV — GAMES PLAYED PREDICTION (direct)
# =================================================================
print("\n" + "=" * 80)
print("METHOD 9: DIRECT GAMES-PLAYED PREDICTION (Ridge regression)")
print("Predicting games directly (not via advancement probabilities)")
print("=" * 80)

def ridge_games_cv(feature_list, label):
    fold_maes = []
    for yr in m['year'].unique():
        tr = m['year'].values != yr
        te = m['year'].values == yr
        
        parts = [np.ones(N), m['seed'].values]
        for f in feature_list:
            fm, fs = m.loc[tr, f].mean(), max(m.loc[tr, f].std(), 0.1)
            parts.append((m[f].values - fm) / fs)
        
        X = np.column_stack(parts)
        X_tr, X_te = X[tr], X[te]
        y_tr = m['games'].values[tr]
        
        p = X_tr.shape[1]
        pen = 5.0 * np.eye(p); pen[0,0] = 0
        beta = solve(X_tr.T @ X_tr + pen, X_tr.T @ y_tr)
        preds = X_te @ beta
        fold_maes.append(np.mean(np.abs(preds - m['games'].values[te])))
    return np.mean(fold_maes)

configs = [
    ([], 'Seed only'),
    (['adjem'], 'Seed + AdjEM'),
    (['barthag'], 'Seed + BARTHAG'),
    (['wab'], 'Seed + WAB'),
    (['orb'], 'Seed + ORB%'),
    (['three_rate'], 'Seed + 3PR'),
    (['efg_def'], 'Seed + eFG%Def'),
    (['adjem', 'orb'], 'Seed + AdjEM + ORB'),
    (['adjem', 'three_rate'], 'Seed + AdjEM + 3PR'),
    (['barthag', 'orb'], 'Seed + BARTHAG + ORB'),
    (['barthag', 'three_rate'], 'Seed + BARTHAG + 3PR'),
    (['adjem', 'orb', 'three_rate'], 'Seed + AdjEM + ORB + 3PR'),
    (['barthag', 'orb', 'three_rate'], 'Seed + BARTHAG + ORB + 3PR'),
    (['adjem', 'efg_def', 'orb'], 'Seed + AdjEM + eFGDef + ORB'),
    (['barthag', 'efg_def', 'three_rate'], 'Seed + BARTHAG + eFGDef + 3PR'),
    (['adjem', 'wab'], 'Seed + AdjEM + WAB'),
    (['net_efg', 'orb', 'three_rate'], 'Seed + NetEFG + ORB + 3PR'),
    (['adjem', 'net_efg', 'orb', 'three_rate'], 'Seed + AdjEM + NetEFG + ORB + 3PR'),
]

results_m9 = []
for feats, label in configs:
    mae = ridge_games_cv(feats, label)
    results_m9.append((label, mae))

results_m9.sort(key=lambda x: x[1])
baseline_m9 = [r for r in results_m9 if r[0]=='Seed only'][0][1]

print(f"\n  {'Model':>42s} {'MAE(games)':>11s} {'vs baseline':>12s}")
print("  " + "-" * 68)
for label, mae in results_m9:
    delta = baseline_m9 - mae
    sig = " ***" if delta > 0.01 else ""
    print(f"  {label:>42s} {mae:11.3f} {delta:+12.3f}{sig}")

# =================================================================
# METHOD 10: BOOTSTRAP CONFIDENCE INTERVALS
# =================================================================
print("\n" + "=" * 80)
print("METHOD 10: BOOTSTRAP CONFIDENCE INTERVALS ON PARTIAL CORRELATIONS")
print("How stable are the partial correlations across resamples?")
print("=" * 80)

n_boot = 2000
rng = np.random.RandomState(42)
boot_partials = defaultdict(list)

for b in range(n_boot):
    idx = rng.choice(N, size=N, replace=True)
    Xb = X_s[idx]
    yb = m['games'].values[idx]
    
    bgb, _, _, _ = lstsq(Xb, yb, rcond=None)
    grb = yb - Xb @ bgb
    
    for f in ['adjem','barthag','wab','orb','three_rate','efg_def','net_efg','def_composite']:
        fvb = m[f].values[idx]
        bfb, _, _, _ = lstsq(Xb, fvb, rcond=None)
        frb = fvb - Xb @ bfb
        r, _ = pearsonr(frb, grb)
        boot_partials[f].append(r)

print(f"\n  {'Feature':>18s} {'Mean r':>8s} {'2.5%':>7s} {'97.5%':>7s} {'P(r>0)':>8s} {'Reliable?':>12s}")
print("  " + "-" * 65)

for f in ['adjem','barthag','wab','orb','three_rate','efg_def','net_efg','def_composite']:
    vals = np.array(boot_partials[f])
    ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
    p_pos = np.mean(vals > 0)
    reliable = "YES" if (ci_lo > 0 or ci_hi < 0) else "BORDERLINE" if p_pos > 0.90 or p_pos < 0.10 else "NO"
    print(f"  {f:>18s} {vals.mean():8.3f} {ci_lo:7.3f} {ci_hi:7.3f} {p_pos:8.1%}  {reliable}")

# =================================================================
# GRAND SUMMARY
# =================================================================
print("\n" + "=" * 80)
print("GRAND SUMMARY ACROSS ALL 10 METHODS")
print("=" * 80)

print("""
  Feature-by-feature verdict (signal for predicting advancement beyond seed):

  FEATURE         M1    M2    M3(AUC)  M4(RF)  M5(EN)  M6(Int)  M7(NL)  M9(Ridge)  M10(Boot)  VERDICT
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
""")

# Collect verdicts
def get_verdict(feat):
    # M1: partial r
    m1 = [r for f, r, p in results_m1 if f == feat]
    m1_score = abs(m1[0]) if m1 else 0
    
    # M2: rank partial r
    m2 = [r for f, r, p in results_m2 if f == feat]
    m2_score = abs(m2[0]) if m2 else 0
    
    # M4: RF importance
    m4 = [imp for f, imp, _ in avg_imp if f == feat]
    m4_score = m4[0] if m4 else 0
    
    # M10: bootstrap
    if feat in boot_partials:
        vals = np.array(boot_partials[feat])
        m10_reliable = (np.percentile(vals, 2.5) > 0 or np.percentile(vals, 97.5) < 0)
        m10_ppos = np.mean(vals > 0)
    else:
        m10_reliable = False
        m10_ppos = 0.5
    
    # Overall
    signals = sum([
        m1_score > 0.18,
        m2_score > 0.18,
        m4_score > 0.02,
        m10_reliable or m10_ppos > 0.88,
    ])
    
    if signals >= 3:
        return "INCLUDE", signals
    elif signals >= 2:
        return "CONSIDER", signals
    else:
        return "EXCLUDE", signals

for feat in ['adjem','barthag','wab','orb','three_rate','efg_def',
             'net_efg','def_composite','adjoe','efg','to_rate','ft_rate',
             'three_pct','two_pct','tempo','drb']:
    verdict, score = get_verdict(feat)
    m1v = [r for f,r,p in results_m1 if f==feat]
    m1s = f"{m1v[0]:+.3f}" if m1v else "  N/A"
    m2v = [r for f,r,p in results_m2 if f==feat]
    m2s = f"{m2v[0]:+.3f}" if m2v else "  N/A"
    m4v = [i for f,i,_ in avg_imp if f==feat]
    m4s = f"{m4v[0]:+.4f}" if m4v else "  N/A"
    
    print(f"  {feat:>16s}  {m1s}  {m2s}  {m4s}  {score}/4  -> {verdict}")

