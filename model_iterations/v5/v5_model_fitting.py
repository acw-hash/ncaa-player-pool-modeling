#!/usr/bin/env python3
"""
V5 Calibration: Find optimal gammas for BARTHAG + 3PR in advancement model.
Since AdjEM and BARTHAG are ~0.95 correlated, test whether BARTHAG replaces
AdjEM or whether a composite works better.
"""
import pandas as pd, numpy as np
from numpy.linalg import solve, lstsq
from scipy.stats import pearsonr
import json, sys, os, warnings
warnings.filterwarnings('ignore')
from enhanced_model_analysis import build_dataset

# Parse pre-tournament BT data
all_bt = []
for year in [2022,2023,2024,2025]:
    raw = pd.read_excel('/mnt/user-data/uploads/team_stats_pre_tournament.xlsx', sheet_name=str(year))
    for i, row in raw.iterrows():
        if str(row.iloc[0]).strip()=='RK':
            raw.columns=raw.iloc[i].values; data=raw.iloc[i+1:].copy(); break
    data=data[pd.to_numeric(data['RK'],errors='coerce').notna()].copy()
    data['TEAM']=data['TEAM'].astype(str).str.strip()
    for c in ['ADJOE','ADJDE','BARTHAG','EFG%','EFGD%','TOR','TORD','ORB','DRB','FTR','FTRD','2P%','2P%D','3P%','3P%D','3PR','3PRD','ADJ T.','WAB']:
        if c in data.columns: data[c]=pd.to_numeric(data[c],errors='coerce')
    data['year']=year; data['ADJEM']=data['ADJOE']-data['ADJDE']
    all_bt.append(data)
bt=pd.concat(all_bt,ignore_index=True)

df=build_dataset()
our=df.groupby(['year','team']).agg({'seed':'first','games_played':'first','adj_em':'first'}).reset_index()

def find_match(name, bt_names):
    if name in bt_names: return name
    aliases={'North Carolina':'North Carolina','Iowa State':'Iowa St.','Ohio State':'Ohio St.',
        'Michigan State':'Michigan St.','Kansas State':'Kansas St.',
        'SE Missouri St':'Southeast Missouri St.','Corpus Christi':'Texas A&M Corpus Chris',
        'Boise State':'Boise St.','St. Johns':"St. John's",'Colorado State':'Colorado St.',
        'San Diego State':'San Diego St.','New Mexico State':'New Mexico St.'}
    if name in aliases and aliases[name] in bt_names: return aliases[name]
    for bn in bt_names:
        if name.lower() in bn.lower() or bn.lower() in name.lower(): return bn
    parts=name.lower().split()
    for bn in bt_names:
        if parts[0] in bn.lower(): return bn
    return None

team_rows=[]
for _,r in our.iterrows():
    yr=int(r['year']); bty=bt[bt['year']==yr]
    match=find_match(r['team'],set(bty['TEAM']))
    if match:
        br=bty[bty['TEAM']==match].iloc[0]
        team_rows.append({'year':yr,'team':r['team'],'seed':int(r['seed']),
            'games_played':int(r['games_played']),'adj_em_ours':r['adj_em'],
            'barthag':br['BARTHAG'],'three_rate':br['3PR'],'adjem_bt':br['ADJEM'],
            'adjoe':br['ADJOE'],'efg_def':br['EFGD%'],'orb':br['ORB']})
team_df=pd.DataFrame(team_rows)

# Merge BT data into player-level
df_full=df.merge(team_df[['year','team','barthag','three_rate','adjem_bt','adjoe','efg_def','orb']],
                  on=['year','team'],how='left')

print("=" * 80)
print("V5 CALIBRATION: BARTHAG + 3PR ADVANCEMENT MODEL")
print("=" * 80)
print(f"Players: {len(df_full)}, Teams: {len(team_df)}")

SEED_PROBS={1:[1,0.993,0.850,0.600,0.390,0.220,0.130],2:[1,0.940,0.670,0.400,0.220,0.110,0.060],
    3:[1,0.850,0.510,0.250,0.120,0.050,0.020],4:[1,0.790,0.430,0.200,0.090,0.040,0.015],
    5:[1,0.640,0.330,0.140,0.060,0.025,0.010],6:[1,0.630,0.310,0.120,0.050,0.020,0.008],
    7:[1,0.600,0.230,0.080,0.030,0.012,0.005],8:[1,0.500,0.200,0.070,0.025,0.010,0.004],
    9:[1,0.500,0.170,0.060,0.020,0.008,0.003],10:[1,0.390,0.130,0.040,0.015,0.006,0.002],
    11:[1,0.370,0.110,0.035,0.012,0.005,0.002],12:[1,0.360,0.090,0.025,0.008,0.003,0.001],
    13:[1,0.210,0.050,0.010,0.003,0.001,0.000],14:[1,0.150,0.020,0.005,0.001,0.000,0.000],
    15:[1,0.070,0.010,0.002,0.000,0.000,0.000],16:[1,0.010,0.002,0.000,0.000,0.000,0.000]}

def shrink(seed):
    if seed<=2: return 0.15
    elif seed<=5: return 0.60
    elif seed<=8: return 0.20
    else: return 0.60

def get_adj_probs_multi(seed, feat_dict, gammas, means, stds):
    probs=list(SEED_PROBS.get(seed,SEED_PROBS[16]))
    total_adj=0
    for fname,gamma in gammas.items():
        val=feat_dict.get(fname)
        if val is not None and not np.isnan(val) and gamma!=0:
            total_adj += gamma * (val - means[fname]) / stds[fname]
    if abs(total_adj)>0.001:
        adjusted=[probs[0]]
        for r in range(1,len(probs)):
            bp=max(min(probs[r],0.999),0.001)
            logit=np.log(bp/(1-bp))+total_adj
            ap=1/(1+np.exp(-logit))
            ap=min(ap,adjusted[-1])
            adjusted.append(ap)
        return adjusted
    return probs

def run_full_cv(df_in, adv_gammas, label):
    fold_maes,fold_overlaps,fold_pts=[],[],[]
    for test_year in df_in['year'].unique():
        train=df_in[df_in['year']!=test_year]
        test=df_in[df_in['year']==test_year]
        # Check we have the features
        for f in adv_gammas:
            if f not in test.columns or test[f].isna().all(): continue
        
        mu=train['ppg_reg'].mean()
        ppg_tr=train.apply(lambda r:(1-shrink(r['seed']))*r['ppg_reg']+shrink(r['seed'])*mu,axis=1)
        ppg_te=test.apply(lambda r:(1-shrink(r['seed']))*r['ppg_reg']+shrink(r['seed'])*mu,axis=1)
        X_tr=np.column_stack([np.ones(len(train)),ppg_tr.values,train['seed'].values])
        X_te=np.column_stack([np.ones(len(test)),ppg_te.values,test['seed'].values])
        y_tr=train['pts_per_game_scoring'].values
        pen=50.0*np.eye(3);pen[0,0]=0
        beta=solve(X_tr.T@X_tr+pen,X_tr.T@y_tr)
        ppg_pred=np.maximum(X_te@beta,2.0)
        
        means={f:train[f].mean() for f in adv_gammas if f in train.columns}
        stds={f:max(train[f].std(),0.1) for f in adv_gammas if f in train.columns}
        
        preds=[]
        for i,(_,row) in enumerate(test.iterrows()):
            fd={f:row[f] for f in adv_gammas if f in row.index}
            probs=get_adj_probs_multi(row['seed'],fd,adv_gammas,means,stds)
            eg=sum(probs[:6])
            preds.append(ppg_pred[i]*eg+row['seed'])
        
        preds=np.array(preds)
        actuals=test['tourney_total'].values
        fold_maes.append(np.mean(np.abs(preds-actuals)))
        tw=test.copy();tw['pred']=preds
        a8=set(tw.nlargest(8,'tourney_total')['player'])
        p8=set(tw.nlargest(8,'pred')['player'])
        fold_overlaps.append(len(a8&p8))
        fold_pts.append(tw.nlargest(8,'pred')['tourney_total'].sum())
    
    return {'label':label,'cv_mae':np.mean(fold_maes),'overlap':np.mean(fold_overlaps),
            'actual_pts':np.mean(fold_pts),'fold_maes':fold_maes}

# =================================================================
# SWEEP: All combinations of BARTHAG, 3PR, and AdjEM gammas
# =================================================================
print("\n  Sweeping advancement model configurations...\n")

configs = [
    ({}, 'Seed only (baseline)'),
    ({'adj_em': 0.08}, 'AdjEM=0.08 (V4)'),
    # BARTHAG alone
    ({'barthag': 0.08}, 'BARTHAG=0.08'),
    ({'barthag': 0.10}, 'BARTHAG=0.10'),
    ({'barthag': 0.12}, 'BARTHAG=0.12'),
    ({'barthag': 0.15}, 'BARTHAG=0.15'),
    ({'barthag': 0.20}, 'BARTHAG=0.20'),
    # 3PR alone
    ({'three_rate': 0.05}, '3PR=0.05'),
    ({'three_rate': 0.08}, '3PR=0.08'),
    ({'three_rate': 0.10}, '3PR=0.10'),
    ({'three_rate': 0.12}, '3PR=0.12'),
    ({'three_rate': 0.15}, '3PR=0.15'),
    # BARTHAG + 3PR
    ({'barthag': 0.08, 'three_rate': 0.05}, 'BARTHAG=0.08 + 3PR=0.05'),
    ({'barthag': 0.08, 'three_rate': 0.08}, 'BARTHAG=0.08 + 3PR=0.08'),
    ({'barthag': 0.08, 'three_rate': 0.10}, 'BARTHAG=0.08 + 3PR=0.10'),
    ({'barthag': 0.10, 'three_rate': 0.05}, 'BARTHAG=0.10 + 3PR=0.05'),
    ({'barthag': 0.10, 'three_rate': 0.08}, 'BARTHAG=0.10 + 3PR=0.08'),
    ({'barthag': 0.10, 'three_rate': 0.10}, 'BARTHAG=0.10 + 3PR=0.10'),
    ({'barthag': 0.12, 'three_rate': 0.05}, 'BARTHAG=0.12 + 3PR=0.05'),
    ({'barthag': 0.12, 'three_rate': 0.08}, 'BARTHAG=0.12 + 3PR=0.08'),
    ({'barthag': 0.12, 'three_rate': 0.10}, 'BARTHAG=0.12 + 3PR=0.10'),
    ({'barthag': 0.15, 'three_rate': 0.05}, 'BARTHAG=0.15 + 3PR=0.05'),
    ({'barthag': 0.15, 'three_rate': 0.08}, 'BARTHAG=0.15 + 3PR=0.08'),
    ({'barthag': 0.15, 'three_rate': 0.10}, 'BARTHAG=0.15 + 3PR=0.10'),
    # AdjEM + 3PR (no BARTHAG, to test if 3PR adds to V4)
    ({'adj_em': 0.08, 'three_rate': 0.08}, 'AdjEM=0.08 + 3PR=0.08'),
    ({'adj_em': 0.08, 'three_rate': 0.10}, 'AdjEM=0.08 + 3PR=0.10'),
    # All three (careful of collinearity)
    ({'barthag': 0.06, 'adj_em': 0.04, 'three_rate': 0.08}, 'BARTHAG=0.06+AdjEM=0.04+3PR=0.08'),
    ({'barthag': 0.08, 'adj_em': 0.04, 'three_rate': 0.06}, 'BARTHAG=0.08+AdjEM=0.04+3PR=0.06'),
    # Replace AdjEM entirely with BARTHAG
    ({'barthag': 0.15, 'three_rate': 0.12}, 'BARTHAG=0.15 + 3PR=0.12'),
    ({'barthag': 0.12, 'three_rate': 0.12}, 'BARTHAG=0.12 + 3PR=0.12'),
]

# Use adjem_bt from Barttorvik for configs that use adj_em
# We need to map adj_em column properly
df_test = df_full.copy()
df_test['adj_em'] = df_test['adjem_bt'].fillna(df_test['adj_em'])

results = []
for gammas, label in configs:
    # Map feature names to columns
    mapped_gammas = {}
    for k, v in gammas.items():
        if k == 'adj_em':
            mapped_gammas['adjem_bt'] = v
        else:
            mapped_gammas[k] = v
    r = run_full_cv(df_test, mapped_gammas, label)
    results.append(r)

results.sort(key=lambda x: x['cv_mae'])
baseline = [r for r in results if 'baseline' in r['label']][0]
v4 = [r for r in results if 'V4' in r['label']][0]

print(f"  {'Config':>42s} {'MAE':>7s} {'OL':>5s} {'Pts':>6s} {'vs Base':>8s} {'vs V4':>7s}")
print("  " + "-" * 80)
for r in results:
    d_base = baseline['cv_mae'] - r['cv_mae']
    d_v4 = v4['cv_mae'] - r['cv_mae']
    mk = " ***" if r == results[0] else ""
    print(f"  {r['label']:>42s} {r['cv_mae']:7.2f} {r['overlap']:5.1f} {r['actual_pts']:6.0f} "
          f"{d_base:+8.2f} {d_v4:+7.2f}{mk}")

best = results[0]
# Also find best by draft quality
results_by_pts = sorted(results, key=lambda x: (-x['actual_pts'], -x['overlap'], x['cv_mae']))
best_pts = results_by_pts[0]

# Find best that maintains 4.5/8 overlap
good_overlap = [r for r in results if r['overlap'] >= 4.4]
good_overlap.sort(key=lambda x: x['cv_mae'])
best_safe = good_overlap[0] if good_overlap else best

print(f"\n  BEST MAE:            {best['label']} (MAE={best['cv_mae']:.2f}, OL={best['overlap']:.1f}, Pts={best['actual_pts']:.0f})")
print(f"  BEST DRAFT QUALITY:  {best_pts['label']} (MAE={best_pts['cv_mae']:.2f}, OL={best_pts['overlap']:.1f}, Pts={best_pts['actual_pts']:.0f})")
print(f"  BEST SAFE (OL>=4.5): {best_safe['label']} (MAE={best_safe['cv_mae']:.2f}, OL={best_safe['overlap']:.1f}, Pts={best_safe['actual_pts']:.0f})")
print(f"  V4 CURRENT:          {v4['label']} (MAE={v4['cv_mae']:.2f}, OL={v4['overlap']:.1f}, Pts={v4['actual_pts']:.0f})")
print(f"  BASELINE:            {baseline['label']} (MAE={baseline['cv_mae']:.2f}, OL={baseline['overlap']:.1f}, Pts={baseline['actual_pts']:.0f})")

# Save selected config
# Choose the best config that doesn't degrade draft quality
selected = best_safe
print(f"\n  SELECTED FOR V5: {selected['label']}")

# Extract gammas from selected
for gammas, label in configs:
    if label == selected['label']:
        selected_gammas = gammas
        break

# Now get normalization constants for BARTHAG and 3PR
print(f"\n  Normalization constants:")
for f in ['barthag', 'three_rate']:
    vals = df_test[f].dropna()
    print(f"    {f}: mean={vals.mean():.4f}, std={vals.std():.4f}")

# Save params
params = {
    'advancement_gammas': selected_gammas,
    'barthag_mean': round(float(df_test['barthag'].mean()), 4),
    'barthag_std': round(float(df_test['barthag'].std()), 4),
    'three_rate_mean': round(float(df_test['three_rate'].mean()), 2),
    'three_rate_std': round(float(df_test['three_rate'].std()), 2),
    'selected_config': selected['label'],
    'cv_mae': round(float(selected['cv_mae']), 4),
    'overlap': float(selected['overlap']),
    'actual_pts': float(selected['actual_pts']),
}

with open('v5_advancement_params.json', 'w') as f:
    json.dump(params, f, indent=2)
print(f"\n  Saved to v5_advancement_params.json")
print(f"  Gammas: {selected_gammas}")

