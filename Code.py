# Importing Libraries

import RNA, numpy as np, pandas as pd
from collections import Counter, defaultdict
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
import os, warnings, time
warnings.filterwarnings("ignore")

# Path configuration
IS_COLAB  = os.path.exists('/content')
IS_KAGGLE = os.path.exists('/kaggle/working')

if IS_COLAB:
    TRAIN_PATH   = '/content/train.csv'
    TEST_PATH    = '/content/test.csv'
    MIRBASE_PATH = '/content/mature.fa'
    # ← IMPORTANT: use submission_r1.csv (OOF=0.836, gap=0.79, NOT round 4)
    PREV_SUB     = '/content/submission_r1.csv'
    OUT_DIR      = '/content/outputs'
elif IS_KAGGLE:
    import glob
    TRAIN_PATH   = glob.glob('/kaggle/input/**/train.csv',  recursive=True)[0]
    TEST_PATH    = glob.glob('/kaggle/input/**/test.csv',   recursive=True)[0]
    MIRBASE_PATH = glob.glob('/kaggle/input/**/mature.fa',  recursive=True)[0]
    PREV_SUB     = '/kaggle/working/submission_r1.csv'
    OUT_DIR      = '/kaggle/working/'
else:
    TRAIN_PATH='Inputs/train.csv'
    TEST_PATH='Inputs/test.csv'
    MIRBASE_PATH='Inputs/mature.fa'
    PREV_SUB='Inputs/submission_r1.csv'   # ADD THIS LINE
    OUT_DIR='Outputs'
    os.makedirs(OUT_DIR, exist_ok=True)

_rng = np.random.default_rng(42)
NUCS = list("ACGU")
K3   = ["".join(p) for p in product(NUCS, repeat=3)]
K4   = ["".join(p) for p in product(NUCS, repeat=4)]
RC_T = str.maketrans('ACGU','UGCA')
def rev_comp(s): return s.translate(RC_T)[::-1]

# Loading Input files 
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
X_raw = np.array(train["Sequence"].str.upper().str.replace("T","U").tolist())
y     = train["Label"].values
X_tst = np.array(test["Sequence"].str.upper().str.replace("T","U").tolist())
print(f"Train: {len(X_raw)}  |  Test: {len(X_tst)}")

# Mirbase feature Extraction
def load_mirbase(path):
    seqs = []
    with open(path) as f:
        buf = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if buf and 14 <= len(buf) <= 35: seqs.append(buf)
                buf = ''
            else:
                buf += line.upper().replace('T','U')
        if buf and 14 <= len(buf) <= 35: seqs.append(buf)
    return seqs

print("Loading miRBase...")
mirbase = load_mirbase(MIRBASE_PATH)
print(f"Loaded {len(mirbase):,}")

def auto_k(n_seqs, avg_len):
    for k in range(5,16):
        if min((n_seqs*max(avg_len-k+1,1))/(4**k)*100,100) < 20: return k
    return 13

K_BASE = auto_k(len(mirbase), 22)
K_LIST = [k for k in [K_BASE-1, K_BASE, K_BASE+1, K_BASE+2] if k >= 9]
print(f"k_base={K_BASE}  K_LIST={K_LIST}")

print(f"Building k-mer indices...")
kidx = {}
for k in K_LIST:
    idx = defaultdict(set)
    for i,s in enumerate(mirbase):
        for j in range(len(s)-k+1): idx[s[j:j+k]].add(i)
    kidx[k] = idx

# Also build k=7 index specifically for seed candidates
kidx7 = defaultdict(set)
for i,s in enumerate(mirbase):
    for j in range(len(s)-6): kidx7[s[j:j+7]].add(i)
print("Done.")

# SMITH-WATERMAN ALIGNMENT FEATURES Extraction 

_MATCH_SCORE   =  2
_MISMATCH_SCORE= -1
_GAP_SCORE     = -2

def sw_score_norm(s1, s2):
    """Normalized SW local alignment score (0–1)."""
    m, n  = len(s1), len(s2)
    s1b   = np.frombuffer(s1.encode(), dtype=np.uint8)
    s2b   = np.frombuffer(s2.encode(), dtype=np.uint8)
    prev  = np.zeros(n+1, dtype=np.float32)
    best  = 0.0
    for i in range(m):
        sc   = np.where(s2b == s1b[i], _MATCH_SCORE, _MISMATCH_SCORE).astype(np.float32)
        curr = np.zeros(n+1, dtype=np.float32)
        for j in range(1, n+1):
            v = max(0.0, prev[j-1]+sc[j-1], prev[j]+_GAP_SCORE, curr[j-1]+_GAP_SCORE)
            curr[j] = v
            if v > best: best = v
        prev = curr
    # Normalize by max possible score (perfect match of shorter sequence)
    perfect = min(m, n) * _MATCH_SCORE
    return float(best / perfect) if perfect > 0 else 0.0

def get_sw_candidates(seq, n_cand=60):
    """Use k-mer seeds to find ~60 nearest miRBase candidates."""
    k  = K_LIST[0]   # smallest k → most candidates found
    qk = set(seq[i:i+k] for i in range(len(seq)-k+1))
    cands = set()
    for km in qk: cands |= kidx[k].get(km, set())
    # Also add seed-based candidates (k=7)
    if len(seq) >= 7:
        seed = seq[1:8]
        cands |= kidx7.get(seed, set())
    if not cands:
        return []
    # Rank candidates by k-mer overlap (cheap), keep top n_cand
    def cov(c):
        mk = set(mirbase[c][i:i+k] for i in range(len(mirbase[c])-k+1))
        return len(qk & mk)
    ranked = sorted(cands, key=cov, reverse=True)[:n_cand]
    return ranked

def sw_features(seq):
    """
    Smith-Waterman alignment features vs nearest miRBase sequences.
    Returns 6 features:
      sw_best:    best normalized SW score (0-1)
      sw_mean3:   mean of top-3 SW scores
      sw_mean10:  mean of top-10 SW scores
      sw_gt90:    fraction of candidates with SW > 0.90
      sw_gt80:    fraction of candidates with SW > 0.80
      sw_gt70:    fraction of candidates with SW > 0.70
    """
    cands = get_sw_candidates(seq, n_cand=60)
    if not cands:
        return [0.0] * 6
    scores = sorted([sw_score_norm(seq, mirbase[c]) for c in cands], reverse=True)
    n = len(scores)
    return [
        scores[0],
        float(np.mean(scores[:3])),
        float(np.mean(scores[:min(10,n)])),
        float(sum(1 for s in scores if s > 0.90) / n),
        float(sum(1 for s in scores if s > 0.80) / n),
        float(sum(1 for s in scores if s > 0.70) / n),
    ]

def batch_sw_features(seqs):
    """Compute SW features for all sequences with progress reporting."""
    t = time.time()
    out = []
    for i, seq in enumerate(seqs):
        out.append(sw_features(seq))
        if (i+1) % 200 == 0:
            elapsed = time.time() - t
            eta = elapsed / (i+1) * (len(seqs) - i - 1)
            print(f"  {i+1}/{len(seqs)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    print(f"  Done in {time.time()-t:.0f}s")
    return np.array(out, dtype=np.float32)

# K-MER COVERAGE FEATURES Extraction 

def coverage_features(seq):
    THRS = [0.3, 0.5, 0.7]
    out  = []
    for k in K_LIST:
        qk = set(seq[i:i+k] for i in range(len(seq)-k+1))
        if not qk:
            out += [0.0] * len(THRS); continue
        cands = set()
        for km in qk: cands |= kidx[k].get(km, set())
        if not cands:
            out += [0.0] * len(THRS); continue
        for thr in THRS:
            n = sum(1 for c in cands
                    if len(qk & set(mirbase[c][i:i+k]
                                     for i in range(len(mirbase[c])-k+1)))/len(qk) >= thr)
            out.append(float(np.log1p(n)))
    return out

# VIENNARNA + K-MER BASE FEATURES EXTRACTION 
def dinuc_odds(seq):
    n=max(len(seq),1); mono={nuc:seq.count(nuc)/n for nuc in NUCS}; out=[]
    for a,b in product(NUCS,repeat=2):
        obs=sum(1 for i in range(len(seq)-1) if seq[i]==a and seq[i+1]==b)/max(n-1,1)
        out.append(obs/max(mono[a]*mono[b],1e-6))
    return out

def extract_base(seq):
    seq=seq.upper().replace("T","U"); n=len(seq)
    fc=RNA.fold_compound(seq); ss,mfe=fc.mfe(); _,efe=fc.pf(); bpp=fc.bpp()
    unp=[max(0.0,1.0-(sum(bpp[i][j] for j in range(n+1))+sum(bpp[j][i] for j in range(i))))
         for i in range(1,n+1)]
    s_list=list(seq); sh=[]
    for _ in range(25): _rng.shuffle(s_list); _,m=RNA.fold("".join(s_list)); sh.append(m)
    z=(mfe-np.mean(sh))/max(np.std(sh),1e-4)
    bp=ss.count("("); ms=cur=0
    for c in ss:
        if c=="(": cur+=1; ms=max(ms,cur)
        else: cur=0
    seed_unp=unp[1:7]; end_unp=unp[-8:]; unp_pad=(unp+[0.5]*(26-n)) if n<26 else unp[:26]
    def kf(k,km): t=max(n-k+1,1); c=Counter(seq[i:i+k] for i in range(n-k+1)); return [c.get(m,0)/t for m in km]
    gc=(seq.count("G")+seq.count("C"))/n; pur=(seq.count("A")+seq.count("G"))/n
    tr=sum(seq[i]!=seq[i+1] for i in range(n-1))/max(n-1,1)
    seed=seq[1:8]; sn=max(len(seed),1); seed_gc=(seed.count("G")+seed.count("C"))/sn
    gu=sum(1 for i in range(n-1) if seq[i]=="G" and seq[i+1]=="U")/max(n-1,1)
    return np.array(
        [mfe,efe,mfe/n,efe-mfe,bp/n,ss.count(".")/n,ms,
         float(ss[0]=="("),float(ss[-1]==")"),z,
         float(np.mean(unp)),float(np.std(unp)),
         float(np.mean(seed_unp)),float(np.min(seed_unp)),float(np.max(seed_unp)),
         float(np.mean(end_unp)),float(unp[0]),float(unp[-1]),
         sum(p>0.7 for p in unp)/n,sum(p<0.3 for p in unp)/n,
         gc,pur,tr,float(seq[0]=="U"),float(seq[-1]=="U"),
         float(np.log1p(n)),seed_gc,gu]
        +unp_pad+dinuc_odds(seq)+kf(3,K3)+kf(4,K4), dtype=np.float32)

def all_features(seq):
    base = extract_base(seq)
    cov  = np.array(coverage_features(seq), dtype=np.float32)
    sw   = np.array(sw_features(seq), dtype=np.float32)
    return np.concatenate([base, cov, sw])

# ── BUILDLING FEATURE MATRICES FROM THE FEATURES
print("\nExtracting ViennaRNA + k-mer base features (train)...")
X_tr_base = np.array([extract_base(s) for s in X_raw])
print("Extracting ViennaRNA + k-mer base features (test)...")
X_te_base = np.array([extract_base(s) for s in X_tst])

print("\nComputing k-mer coverage features (train)...")
X_tr_cov = np.array([coverage_features(s) for s in X_raw], dtype=np.float32)
print("Computing k-mer coverage features (test)...")
X_te_cov = np.array([coverage_features(s) for s in X_tst], dtype=np.float32)

print("\nComputing Smith-Waterman alignment features (train, ~2 min)...")
X_tr_sw = batch_sw_features(X_raw)
print("Computing Smith-Waterman alignment features (test, ~30s)...")
X_te_sw = batch_sw_features(X_tst)

# Checking SW features discriminative powers

print("\nSW feature AUCs:")
sw_names = ['sw_best','sw_mean3','sw_mean10','sw_gt90','sw_gt80','sw_gt70']
for i, name in enumerate(sw_names):
    feat = X_tr_sw[:, i]
    if feat.std() > 0:
        a = roc_auc_score(y, feat)
        a = max(a, 1-a)
        print(f"  {name:12s}: {a:.5f}  pos_mean={feat[y==1].mean():.3f}  neg_mean={feat[y==0].mean():.3f}")

X_tr = np.hstack([X_tr_base, X_tr_cov, X_tr_sw])
X_te = np.hstack([X_te_base, X_te_cov, X_te_sw])
X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
print(f"\nFull feature matrix: {X_tr.shape}")

# PSEUDO-LABELING (Round 1, gap > 0.5 only) 
n_orig   = len(X_raw)
X_pseudo = np.zeros((0, X_tr.shape[1]), dtype=np.float32)
y_pseudo = np.array([], dtype=np.float32)

if os.path.exists(PREV_SUB):
    preds = pd.read_csv(PREV_SUB)["Label"].values
    if not (0.3 < preds.mean() < 0.7):
        print(f"\nWARNING: {PREV_SUB} mean={preds.mean():.3f} — skipping")
    else:
        idx_s = np.argsort(preds)
        N = 50
        neg_idx, pos_idx = idx_s[:N], idx_s[-N:]
        gap = preds[pos_idx].min() - preds[neg_idx].max()
        print(f"\nPseudo-label source: gap={gap:.4f}")
        if gap > 0.50:
            print(f"Adding {N*2} pseudo-labels (gap > 0.50)...")
            pl_seqs = [X_tst[i] for i in pos_idx] + [X_tst[i] for i in neg_idx]
            pl_base = np.array([extract_base(s) for s in pl_seqs])
            pl_cov  = np.array([coverage_features(s) for s in pl_seqs], dtype=np.float32)
            pl_sw   = batch_sw_features(pl_seqs)
            X_pseudo = np.nan_to_num(np.hstack([pl_base, pl_cov, pl_sw]),
                                      nan=0.0, posinf=0.0, neginf=0.0)
            y_pseudo = np.array([1.0]*N + [0.0]*N, dtype=np.float32)
            print(f"Augmented: {n_orig} + {N*2} = {n_orig+N*2}")
        else:
            print(f"Gap {gap:.4f} < 0.50 → skipping (too noisy)")
else:
    print(f"\n{PREV_SUB} not found → no pseudo-labels")
    print("→ Point PREV_SUB to submission_r1.csv for +0.003-0.005 gain")

X_tr_aug = np.vstack([X_tr, X_pseudo]) if len(X_pseudo) else X_tr
y_aug    = np.concatenate([y, y_pseudo]) if len(y_pseudo) else y

# SCALING FEATURE VALUES 
sc       = StandardScaler()
sc.fit(X_tr_aug)
X_tr_sc  = sc.transform(X_tr)
X_aug_sc = sc.transform(X_tr_aug)
X_te_sc  = sc.transform(X_te)
X_ps     = X_tr_aug[n_orig:]
X_ps_sc  = X_aug_sc[n_orig:]

# ENSEMBLE: XGB + LGB + CB TRANING
CV   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_te = len(X_tst)

oof_xgb=np.zeros(n_orig); tst_xgb=np.zeros(n_te)
oof_lgb=np.zeros(n_orig); tst_lgb=np.zeros(n_te)
oof_cb =np.zeros(n_orig); tst_cb =np.zeros(n_te)

def make_fold(tr_i):
    if len(X_ps) == 0:
        return X_tr[tr_i], y[tr_i]
    return (np.vstack([X_tr[tr_i], X_ps]),
            np.concatenate([y[tr_i], y_pseudo]))

# XGB: 30 seeding 

print(f"\nXGB (30 seeds)...")
for seed in range(30):
    oof_s=np.zeros(n_orig); tst_s=np.zeros(n_te)
    m=xgb.XGBClassifier(n_estimators=1200,learning_rate=0.01,max_depth=5,
        subsample=0.8,colsample_bytree=0.7,reg_alpha=0.3,reg_lambda=1.0,
        eval_metric="auc",random_state=seed,verbosity=0)
    for tr_i,va_i in CV.split(X_tr,y):
        Xf,yf=make_fold(tr_i)
        m.fit(Xf,yf)
        oof_s[va_i]=m.predict_proba(X_tr[va_i])[:,1]
        tst_s+=m.predict_proba(X_te)[:,1]
    oof_xgb+=oof_s; tst_xgb+=tst_s/5
    if (seed+1)%10==0:
        print(f"  {seed+1}/30: {roc_auc_score(y,oof_xgb/(seed+1)):.5f}")
oof_xgb/=30; tst_xgb/=30
print(f"  XGB: {roc_auc_score(y,oof_xgb):.5f}")

# LGB: 25 seeding 

print(f"LGB (25 seeds)...")
for seed in range(25):
    oof_s=np.zeros(n_orig); tst_s=np.zeros(n_te)
    m=lgb.LGBMClassifier(n_estimators=1200,learning_rate=0.01,max_depth=5,
        num_leaves=31,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.3,
        random_state=seed,verbose=-1)
    for tr_i,va_i in CV.split(X_tr,y):
        Xf,yf=make_fold(tr_i)
        m.fit(Xf,yf)
        oof_s[va_i]=m.predict_proba(X_tr[va_i])[:,1]
        tst_s+=m.predict_proba(X_te)[:,1]
    oof_lgb+=oof_s; tst_lgb+=tst_s/5
    if (seed+1)%10==0:
        print(f"  {seed+1}/25: {roc_auc_score(y,oof_lgb/(seed+1)):.5f}")
oof_lgb/=25; tst_lgb/=25
print(f"  LGB: {roc_auc_score(y,oof_lgb):.5f}")

# CatBoost: 3 seeding for diversity 

print(f"CatBoost (3 seeds × 5 folds)...")
for seed in range(3):
    oof_s=np.zeros(n_orig); tst_s=np.zeros(n_te)
    for tr_i,va_i in CV.split(X_tr,y):
        Xf,yf=make_fold(tr_i)
        m=CatBoostClassifier(iterations=1500,learning_rate=0.008,depth=7,
            l2_leaf_reg=2,random_seed=seed,verbose=0)
        m.fit(Xf,yf)
        oof_s[va_i]=m.predict_proba(X_tr[va_i])[:,1]
        tst_s+=m.predict_proba(X_te)[:,1]
    oof_cb+=oof_s; tst_cb+=tst_s/5
oof_cb/=3; tst_cb/=3
print(f"  CB: {roc_auc_score(y,oof_cb):.5f}")

# FINAL ENSEMBLE TRAINING 
final_oof = (oof_xgb + oof_lgb + oof_cb) / 3
final_tst = (tst_xgb + tst_lgb + tst_cb) / 3
final_auc = roc_auc_score(y, final_oof)

assert 0 <= final_tst.min() and final_tst.max() <= 1
assert 0.3 < final_tst.mean() < 0.7

print(f"\n{'='*60}")
print(f"  XGB (30-seed) : {roc_auc_score(y,oof_xgb):.5f}")
print(f"  LGB (25-seed) : {roc_auc_score(y,oof_lgb):.5f}")
print(f"  CatBoost      : {roc_auc_score(y,oof_cb):.5f}")
print(f"  Ensemble      : {final_auc:.5f}")
print(f"  Pred scale    : mean={final_tst.mean():.4f}  "
      f"min={final_tst.min():.4f}  max={final_tst.max():.4f}")
print(f"  Expected Kaggle: {final_auc-0.013:.4f}–{final_auc:.4f}")
print(f"{'='*60}")

out_path = os.path.join(OUT_DIR, "submission_file.csv")
pd.DataFrame({"ID": test["ID"], "Label": final_tst}).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

