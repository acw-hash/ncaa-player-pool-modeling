#!/usr/bin/env python3
"""
Comprehensive feature analysis using PRE-TOURNAMENT Barttorvik data.
Tests eFG%, TO%, ORB%, FTR, BARTHAG, WAB, defensive metrics, etc.
"""
import pandas as pd
import numpy as np
from numpy.linalg import solve, lstsq
from scipy.stats import pearsonr, spearmanr
import warnings; warnings.filterwarnings('ignore')
from enhanced_model_analysis import build_dataset

# =================================================================
# PARSE PRE-TOURNAMENT BARTTORVIK DATA
# =================================================================
print("=" * 80)
print("COMPREHENSIVE FEATURE ANALYSIS — PRE-TOURNAMENT DATA")
print("=" * 80)

all_bt = []
for year in [2022, 2023, 2024, 2025]:
    raw = pd.read_excel('/mnt/user-data/uploads/team_stats_pre_tournament.xlsx', sheet_name=str(year))
    # Find header
    for i, row in raw.iterrows():
        if str(row.iloc[0]).strip() == 'RK':
            raw.columns = raw.iloc[i].values
            data = raw.iloc[i+1:].copy()
            break
    # Filter out seed/tournament description rows (NaN in RK)
    data = data[pd.to_numeric(data['RK'], errors='coerce').notna()].copy()
    data['TEAM'] = data['TEAM'].astype(str).str.strip()
    
    num_cols = ['ADJOE','ADJDE','BARTHAG','EFG%','EFGD%','TOR','TORD',
                'ORB','DRB','FTR','FTRD','2P%','2P%D','3P%','3P%D',
                '3PR','3PRD','ADJ T.','WAB']
    for c in num_cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors='coerce')
    
    data['year'] = year
    data['ADJEM'] = data['ADJOE'] - data['ADJDE']
    all_bt.append(data)
    print(f"  {year}: {len(data)} teams (pre-tournament)")

bt = pd.concat(all_bt, ignore_index=True)
print(f"  Total: {len(bt)} team-seasons")

# =================================================================
# MERGE WITH TRAINING DATA
# =================================================================
df = build_dataset()
our_teams = df.groupby(['year','team']).agg({
    'seed':'first','games_played':'first','adj_em':'first',
    'adj_o':'first','pace':'first','q1_wins':'first',
}).reset_index()

# Name matching
def find_match(our_name, bt_names_series):
    """Match our team name to Barttorvik name."""
    bt_list = bt_names_series.tolist()
    # Direct
    if our_name in bt_list: return our_name
    # Common abbreviations
    aliases = {
        'North Carolina':'North Carolina', 'Iowa State':'Iowa St.',
        'Ohio State':'Ohio St.', 'Michigan State':'Michigan St.',
        'Texas Tech':'Texas Tech', 'Kansas State':'Kansas St.',
        'SE Missouri St':'Southeast Missouri St.', 
        'Corpus Christi':'Texas A&M Corpus Chris',
        'Boise State':'Boise St.', 'St. Johns':"St. John's",
        'Colorado State':'Colorado St.', 'San Diego State':'San Diego St.',
        'New Mexico State':'New Mexico St.',
    }
    if our_name in aliases and aliases[our_name] in bt_list:
        return aliases[our_name]
    # Substring
    for bn in bt_list:
        if our_name.lower() in bn.lower() or bn.lower() in our_name.lower():
            return bn
    # First word
    parts = our_name.lower().split()
    for bn in bt_list:
        if parts[0] in bn.lower():
            return bn
    return None

merged_rows = []
for _, row in our_teams.iterrows():
    yr = int(row['year'])
    bt_yr = bt[bt['year']==yr]
    match = find_match(row['team'], bt_yr['TEAM'])
    if match:
        br = bt_yr[bt_yr['TEAM']==match].iloc[0]
        merged_rows.append({
            'year':yr, 'team':row['team'], 'seed':int(row['seed']),
            'games_played':int(row['games_played']),
            'adj_em_ours':row['adj_em'],
            # Barttorvik pre-tournament metrics
            'adjoe':br['ADJOE'], 'adjde':br['ADJDE'], 'adjem':br['ADJEM'],
            'barthag':br['BARTHAG'],
            'efg':br['EFG%'], 'efg_def':br['EFGD%'],
            'to_rate':br['TOR'], 'to_forced':br['TORD'],
            'orb':br['ORB'], 'drb':br['DRB'],
            'ft_rate':br['FTR'], 'ft_rate_def':br['FTRD'],
            'two_pct':br['2P%'], 'two_pct_def':br['2P%D'],
            'three_pct':br['3P%'], 'three_pct_def':br['3P%D'],
            'three_rate':br['3PR'], 'three_rate_def':br['3PRD'],
            'tempo':br['ADJ T.'], 'wab':br['WAB'],
        })

m = pd.DataFrame(merged_rows)
print(f"\n  Merged: {len(m)}/82 teams")

# Verify pre-tournament data
print(f"\n  Data accuracy (our AdjEM vs Barttorvik pre-tourney AdjEM):")
m['em_diff'] = abs(m['adj_em_ours'] - m['adjem'])
print(f"  Correlation: {m[['adj_em_ours','adjem']].corr().iloc[0,1]:.4f}")
print(f"  Mean |diff|: {m['em_diff'].mean():.2f}")
print(f"  Max |diff|:  {m['em_diff'].max():.2f}")
# Show worst
for _, r in m.nlargest(3, 'em_diff').iterrows():
    print(f"    {r['team']:20s} ({r['year']}): Ours={r['adj_em_ours']:.1f}, BT={r['adjem']:.1f}")

# =================================================================
# PART 1: RAW CORRELATIONS
# =================================================================
print("\n" + "=" * 80)
print("PART 1: RAW CORRELATIONS WITH GAMES PLAYED (N=82 team-seasons)")
print("=" * 80)

features = ['adjem','adjoe','adjde','barthag','efg','efg_def','to_rate',
            'to_forced','orb','drb','ft_rate','ft_rate_def',
            'two_pct','two_pct_def','three_pct','three_pct_def',
            'three_rate','tempo','wab','seed']

print(f"\n  {'Feature':>16s} {'r':>7s} {'p':>8s} {'|r|':>5s}  Interpretation")
print("  " + "-" * 68)
raw_results = []
for f in features:
    r, p = pearsonr(m[f], m['games_played'])
    raw_results.append((f, r, p))
raw_results.sort(key=lambda x: -abs(x[1]))
for f, r, p in raw_results:
    strength = "***STRONG***" if abs(r)>0.35 else "**moderate**" if abs(r)>0.20 else "weak"
    print(f"  {f:>16s} {r:7.3f} {p:8.4f} {abs(r):5.3f}  {strength}")

# =================================================================
# PART 2: PARTIAL CORRELATIONS (seed controlled, linear)
# =================================================================
print("\n" + "=" * 80)
print("PART 2: PARTIAL CORRELATIONS — controlling for seed (linear)")
print("The key question: signal BEYOND what seed already tells us")
print("=" * 80)

X_seed = np.column_stack([np.ones(len(m)), m['seed'].values])
beta_g, _, _, _ = lstsq(X_seed, m['games_played'].values, rcond=None)
games_resid = m['games_played'].values - X_seed @ beta_g
var_exp = 1 - games_resid.var() / m['games_played'].var()
print(f"\n  Seed explains {var_exp:.1%} of games-played variance")

test_features = [f for f in features if f != 'seed']
print(f"\n  {'Feature':>16s} {'Raw r':>7s} {'Partial r':>10s} {'p':>8s}  {'Signal?':>22s}")
print("  " + "-" * 72)

partial_results = []
for f in test_features:
    raw_r, _ = pearsonr(m[f], m['games_played'])
    # Residualize feature on seed
    beta_f, _, _, _ = lstsq(X_seed, m[f].values, rcond=None)
    feat_resid = m[f].values - X_seed @ beta_f
    pr, pp = pearsonr(feat_resid, games_resid)
    partial_results.append((f, raw_r, pr, pp))

partial_results.sort(key=lambda x: -abs(x[2]))
for f, raw_r, pr, pp in partial_results:
    sig = "*** YES ***" if abs(pr)>0.20 and pp<0.10 else "** borderline **" if abs(pr)>0.15 else "weak" if abs(pr)>0.10 else "no"
    print(f"  {f:>16s} {raw_r:7.3f} {pr:10.3f} {pp:8.4f}  {sig}")

# =================================================================
# PART 3: WITHIN-SEED DRILL-DOWN
# =================================================================
print("\n" + "=" * 80)
print("PART 3: WITHIN-SEED ANALYSIS (the 'not all 3-seeds are equal' test)")
print("=" * 80)

# Focus on features with partial r > 0.10
promising = [f for f, _, pr, pp in partial_results if abs(pr) > 0.10]
promising_labels = promising[:8]  # Top 8

for seed_val in [1, 2, 3, 4, 5]:
    sub = m[m['seed'] == seed_val]
    if len(sub) < 6:
        continue
    print(f"\n  === Seed {seed_val} ({len(sub)} teams, games range: {sub['games_played'].min()}-{sub['games_played'].max()}) ===")
    
    for f in promising_labels:
        r, p = pearsonr(sub[f], sub['games_played'])
        marker = " <-- USEFUL" if abs(r)>0.30 and p<0.20 else " <-- hint" if abs(r)>0.20 else ""
        print(f"    {f:>14s}: r={r:6.3f}, p={p:.3f}{marker}")

# =================================================================
# PART 4: COMPOSITE FEATURE ENGINEERING
# =================================================================
print("\n" + "=" * 80)
print("PART 4: COMPOSITE FEATURES — combining metrics for stronger signal")
print("=" * 80)

# Create composites
m['net_efg'] = m['efg'] - m['efg_def']  # Offensive - Defensive eFG gap
m['net_to'] = m['to_forced'] - m['to_rate']  # Turnover margin
m['net_reb'] = m['orb'] + m['drb']  # Total rebounding (off + def reb %)
m['four_factors'] = (m['efg']/50 - m['to_rate']/20 + m['orb']/30 + m['ft_rate']/30)  # Dean Oliver composite
m['def_quality'] = (50 - m['efg_def'])/5 + m['to_forced']/20  # Defensive composite
m['off_efficiency'] = m['efg'] + m['ft_rate']/3 - m['to_rate']/2  # Offensive composite

composites = ['net_efg', 'net_to', 'net_reb', 'four_factors', 'def_quality', 'off_efficiency']

print(f"\n  {'Composite':>18s} {'Raw r':>7s} {'Partial r':>10s} {'p':>8s}")
print("  " + "-" * 48)
for f in composites:
    raw_r, _ = pearsonr(m[f], m['games_played'])
    beta_f, _, _, _ = lstsq(X_seed, m[f].values, rcond=None)
    feat_resid = m[f].values - X_seed @ beta_f
    pr, pp = pearsonr(feat_resid, games_resid)
    print(f"  {f:>18s} {raw_r:7.3f} {pr:10.3f} {pp:8.4f}")

# =================================================================
# PART 5: FULL PIPELINE LOYO-CV — Per-Game Scoring
# =================================================================
print("\n" + "=" * 80)
print("PART 5: FULL PIPELINE TEST — Per-Game Scoring + Total Points")
print("The definitive test: add features to scoring model, measure impact")
print("=" * 80)

# Merge BT data into player-level dataset
df_full = df.merge(
    m[['year','team','adjem','barthag','efg','efg_def','to_rate','to_forced',
       'orb','drb','ft_rate','wab','two_pct','three_pct','tempo',
       'net_efg','def_quality','four_factors','adjoe','adjde']],
    on=['year','team'], how='left'
)
print(f"  Player-level: {len(df_full)} rows, {df_full['efg'].notna().sum()} with BT metrics")

SEED_PROBS = {
    1:[1,0.993,0.850,0.600,0.390,0.220,0.130],2:[1,0.940,0.670,0.400,0.220,0.110,0.060],
    3:[1,0.850,0.510,0.250,0.120,0.050,0.020],4:[1,0.790,0.430,0.200,0.090,0.040,0.015],
    5:[1,0.640,0.330,0.140,0.060,0.025,0.010],6:[1,0.630,0.310,0.120,0.050,0.020,0.008],
    7:[1,0.600,0.230,0.080,0.030,0.012,0.005],8:[1,0.500,0.200,0.070,0.025,0.010,0.004],
    9:[1,0.500,0.170,0.060,0.020,0.008,0.003],10:[1,0.390,0.130,0.040,0.015,0.006,0.002],
    11:[1,0.370,0.110,0.035,0.012,0.005,0.002],12:[1,0.360,0.090,0.025,0.008,0.003,0.001],
    13:[1,0.210,0.050,0.010,0.003,0.001,0.000],14:[1,0.150,0.020,0.005,0.001,0.000,0.000],
    15:[1,0.070,0.010,0.002,0.000,0.000,0.000],16:[1,0.010,0.002,0.000,0.000,0.000,0.000],
}

def shrink(seed):
    if seed <= 2: return 0.15
    elif seed <= 5: return 0.60
    elif seed <= 8: return 0.20
    else: return 0.60

def run_cv(df_in, extra_scoring_feats, adv_feat=None, adv_gamma=0.0, label=''):
    """
    Full LOYO-CV pipeline.
    extra_scoring_feats: list of column names to add to per-game scoring model
    adv_feat: column name to use for advancement adjustment
    adv_gamma: gamma for advancement logistic adjustment
    """
    fold_maes, fold_overlaps, fold_pts, fold_rank_corrs = [], [], [], []
    
    for test_year in df_in['year'].unique():
        train = df_in[df_in['year'] != test_year].dropna(subset=extra_scoring_feats + ([adv_feat] if adv_feat else []))
        test = df_in[df_in['year'] == test_year].dropna(subset=extra_scoring_feats + ([adv_feat] if adv_feat else []))
        if len(test) < 5: continue
        
        mu = train['ppg_reg'].mean()
        
        # Design matrix
        ppg_tr = train.apply(lambda r: (1-shrink(r['seed']))*r['ppg_reg']+shrink(r['seed'])*mu, axis=1)
        ppg_te = test.apply(lambda r: (1-shrink(r['seed']))*r['ppg_reg']+shrink(r['seed'])*mu, axis=1)
        
        parts_tr = [np.ones(len(train)), ppg_tr.values, train['seed'].values]
        parts_te = [np.ones(len(test)), ppg_te.values, test['seed'].values]
        
        for fc in extra_scoring_feats:
            fm, fs = train[fc].mean(), max(train[fc].std(), 0.1)
            parts_tr.append(((train[fc]-fm)/fs).values)
            parts_te.append(((test[fc]-fm)/fs).values)
        
        X_tr = np.column_stack(parts_tr)
        X_te = np.column_stack(parts_te)
        y_tr = train['pts_per_game_scoring'].values
        
        p = X_tr.shape[1]
        pen = 50.0 * np.eye(p); pen[0,0]=0
        beta = solve(X_tr.T@X_tr + pen, X_tr.T @ y_tr)
        ppg_pred = np.maximum(X_te @ beta, 2.0)
        
        # Advancement
        if adv_feat and adv_gamma > 0:
            adv_mean = train[adv_feat].mean()
            adv_std = max(train[adv_feat].std(), 0.1)
        
        preds = []
        for i, (_, row) in enumerate(test.iterrows()):
            probs = list(SEED_PROBS.get(row['seed'], SEED_PROBS[16]))
            
            if adv_feat and adv_gamma > 0 and not np.isnan(row[adv_feat]):
                norm = (row[adv_feat] - adv_mean) / adv_std
                adj = [probs[0]]
                for r in range(1, len(probs)):
                    bp = max(min(probs[r], 0.999), 0.001)
                    logit = np.log(bp/(1-bp)) + adv_gamma * norm
                    ap = 1/(1+np.exp(-logit))
                    ap = min(ap, adj[-1])
                    adj.append(ap)
                probs = adj
            
            eg = sum(probs[:6])
            preds.append(ppg_pred[i] * eg + row['seed'])
        
        preds = np.array(preds)
        actuals = test['tourney_total'].values
        fold_maes.append(np.mean(np.abs(preds - actuals)))
        rho, _ = spearmanr(preds, actuals)
        fold_rank_corrs.append(rho)
        
        tw = test.copy(); tw['pred'] = preds
        a8 = set(tw.nlargest(8,'tourney_total')['player'])
        p8 = set(tw.nlargest(8,'pred')['player'])
        fold_overlaps.append(len(a8&p8))
        fold_pts.append(tw.nlargest(8,'pred')['tourney_total'].sum())
    
    return {
        'label':label, 'cv_mae':np.mean(fold_maes), 'overlap':np.mean(fold_overlaps),
        'actual_pts':np.mean(fold_pts), 'rank_rho':np.mean(fold_rank_corrs),
        'fold_maes':fold_maes,
    }

# ---------- A) SCORING MODEL FEATURES ----------
print("\n  === A) SCORING MODEL: Which features improve per-game scoring prediction? ===\n")

scoring_configs = [
    ([], None, 0, 'BASELINE: PPG + Seed'),
    (['efg'], None, 0, '+ eFG%'),
    (['efg_def'], None, 0, '+ eFG% Defense'),
    (['to_rate'], None, 0, '+ Turnover Rate'),
    (['orb'], None, 0, '+ Off Reb %'),
    (['ft_rate'], None, 0, '+ FT Rate'),
    (['barthag'], None, 0, '+ BARTHAG'),
    (['wab'], None, 0, '+ WAB'),
    (['net_efg'], None, 0, '+ Net eFG (Off-Def)'),
    (['def_quality'], None, 0, '+ Defensive Composite'),
    (['two_pct', 'three_pct'], None, 0, '+ 2P% + 3P%'),
    (['efg', 'to_rate'], None, 0, '+ eFG% + TO Rate'),
    (['efg', 'to_rate', 'orb'], None, 0, '+ eFG% + TO + ORB'),
    (['efg', 'to_rate', 'orb', 'ft_rate'], None, 0, '+ Four Factors'),
    (['usage_pct'], None, 0, '+ Usage Rate'),
    (['usage_pct', 'efg'], None, 0, '+ Usage + eFG%'),
    (['usage_pct', 'efg', 'to_rate'], None, 0, '+ Usage + eFG% + TO'),
    (['net_efg', 'usage_pct'], None, 0, '+ Net eFG + Usage'),
]

print(f"  {'Config':>35s} {'MAE':>7s} {'Rho':>6s} {'OL':>5s} {'Pts':>6s}")
print("  " + "-" * 64)

scoring_results = []
for feats, af, ag, label in scoring_configs:
    r = run_cv(df_full, feats, af, ag, label)
    scoring_results.append(r)

scoring_results.sort(key=lambda x: x['cv_mae'])
for r in scoring_results:
    mk = " ***" if r == scoring_results[0] else ""
    print(f"  {r['label']:>35s} {r['cv_mae']:7.2f} {r['rank_rho']:6.3f} {r['overlap']:5.1f} {r['actual_pts']:6.0f}{mk}")

# ---------- B) ADVANCEMENT MODEL FEATURES ----------
print("\n  === B) ADVANCEMENT MODEL: Which features improve expected-games prediction? ===\n")

adv_configs = [
    ([], None, 0, 'BASELINE: Seed only'),
    ([], 'adjem', 0.08, 'AdjEM γ=0.08 (V4 current)'),
    ([], 'adjem', 0.15, 'AdjEM γ=0.15'),
    ([], 'adjem', 0.25, 'AdjEM γ=0.25'),
    ([], 'adjem', 0.40, 'AdjEM γ=0.40'),
    ([], 'barthag', 0.08, 'BARTHAG γ=0.08'),
    ([], 'barthag', 0.15, 'BARTHAG γ=0.15'),
    ([], 'barthag', 0.25, 'BARTHAG γ=0.25'),
    ([], 'barthag', 0.40, 'BARTHAG γ=0.40'),
    ([], 'wab', 0.08, 'WAB γ=0.08'),
    ([], 'wab', 0.15, 'WAB γ=0.15'),
    ([], 'wab', 0.25, 'WAB γ=0.25'),
    ([], 'def_quality', 0.15, 'DefQuality γ=0.15'),
    ([], 'def_quality', 0.25, 'DefQuality γ=0.25'),
    ([], 'efg_def', 0.15, 'eFG%Def γ=0.15'),
    ([], 'efg_def', 0.25, 'eFG%Def γ=0.25'),
    ([], 'net_efg', 0.15, 'NetEFG γ=0.15'),
    ([], 'net_efg', 0.25, 'NetEFG γ=0.25'),
]

print(f"  {'Config':>35s} {'MAE':>7s} {'Rho':>6s} {'OL':>5s} {'Pts':>6s}")
print("  " + "-" * 64)

adv_results = []
for feats, af, ag, label in adv_configs:
    r = run_cv(df_full, feats, af, ag, label)
    adv_results.append(r)

adv_results.sort(key=lambda x: x['cv_mae'])
for r in adv_results:
    mk = " ***" if r == adv_results[0] else ""
    print(f"  {r['label']:>35s} {r['cv_mae']:7.2f} {r['rank_rho']:6.3f} {r['overlap']:5.1f} {r['actual_pts']:6.0f}{mk}")

# ---------- C) COMBINED: Best scoring + best advancement ----------
print("\n  === C) COMBINED: Best scoring features + best advancement features ===\n")

# Get top scoring features (those that beat baseline)
baseline_mae = [r for r in scoring_results if 'BASELINE' in r['label']][0]['cv_mae']
good_scoring = [r for r in scoring_results if r['cv_mae'] < baseline_mae - 0.02]

# Get top advancement configs
baseline_adv = [r for r in adv_results if 'BASELINE' in r['label']][0]
good_adv = adv_results[:5]

combined_configs = [
    ([], None, 0, 'BASELINE'),
]

# Best scoring + best advancement combos
best_scoring_feats = [
    (['efg'], '+ eFG%'),
    (['two_pct', 'three_pct'], '+ 2P%+3P%'),
    (['net_efg'], '+ NetEFG'),
    (['usage_pct', 'efg'], '+ Usg+eFG'),
]

best_adv_options = [
    ('barthag', 0.15, 'BARTHAG-0.15'),
    ('barthag', 0.25, 'BARTHAG-0.25'),
    ('adjem', 0.15, 'AdjEM-0.15'),
    ('adjem', 0.25, 'AdjEM-0.25'),
    ('wab', 0.15, 'WAB-0.15'),
]

for sf, sl in best_scoring_feats:
    for af, ag, al in best_adv_options:
        combined_configs.append((sf, af, ag, f'{sl} | ADV:{al}'))

# Also test advancement alone with baseline scoring
for af, ag, al in best_adv_options:
    combined_configs.append(([], af, ag, f'Base scoring | ADV:{al}'))

print(f"  {'Config':>45s} {'MAE':>7s} {'Rho':>6s} {'OL':>5s} {'Pts':>6s}")
print("  " + "-" * 74)

combined_results = []
for feats, af, ag, label in combined_configs:
    r = run_cv(df_full, feats, af, ag, label)
    combined_results.append(r)

combined_results.sort(key=lambda x: x['cv_mae'])
for r in combined_results[:20]:
    mk = " ***" if r == combined_results[0] else ""
    print(f"  {r['label']:>45s} {r['cv_mae']:7.2f} {r['rank_rho']:6.3f} {r['overlap']:5.1f} {r['actual_pts']:6.0f}{mk}")

# Sort by draft quality
print(f"\n  --- Top 10 by draft quality (actual pts from top-8 picks) ---")
combined_by_pts = sorted(combined_results, key=lambda x: -x['actual_pts'])
for r in combined_by_pts[:10]:
    print(f"  {r['label']:>45s} MAE={r['cv_mae']:.2f} OL={r['overlap']:.1f} Pts={r['actual_pts']:.0f}")

# =================================================================
# FINAL VERDICT
# =================================================================
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

bl = [r for r in combined_results if r['label']=='BASELINE'][0]
best_mae = combined_results[0]
best_pts = combined_by_pts[0]

print(f"""
  BASELINE (PPG + Seed only):
    MAE={bl['cv_mae']:.2f}, Rank rho={bl['rank_rho']:.3f}, Top-8 overlap={bl['overlap']:.1f}/8, Pts={bl['actual_pts']:.0f}
    
  BEST BY MAE: {best_mae['label']}
    MAE={best_mae['cv_mae']:.2f}, Rank rho={best_mae['rank_rho']:.3f}, Top-8 overlap={best_mae['overlap']:.1f}/8, Pts={best_mae['actual_pts']:.0f}
    Improvement: {bl['cv_mae'] - best_mae['cv_mae']:.2f} MAE ({(bl['cv_mae']-best_mae['cv_mae'])/bl['cv_mae']*100:.1f}%)
    
  BEST BY DRAFT QUALITY: {best_pts['label']}
    MAE={best_pts['cv_mae']:.2f}, Rank rho={best_pts['rank_rho']:.3f}, Top-8 overlap={best_pts['overlap']:.1f}/8, Pts={best_pts['actual_pts']:.0f}
    Points improvement: {best_pts['actual_pts'] - bl['actual_pts']:.0f} pts ({(best_pts['actual_pts']-bl['actual_pts'])/bl['actual_pts']*100:.1f}%)
""")

