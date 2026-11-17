#!/usr/bin/env python3
"""
run_benchmark.py - M2OR-Provenance, single entry point.

    python run_benchmark.py --stage folds      # build and export fold membership
    python run_benchmark.py --stage audit      # leakage audit over the exported folds
    python run_benchmark.py --stage evaluate   # train and score the reference models
    python run_benchmark.py --stage report     # regenerate every number quoted in the paper
    python run_benchmark.py --stage all

Fold membership is written out as record ids BEFORE any model is fitted, so a
reader can inspect or replace the splits without running the models, and can
verify that no label was constructed from a test source.

Outputs (under --out, default ./bench_out):
    records.csv        one row per (pair, source): record id, receptor, InChI key,
                       scaffold, source DOI, observed call
    folds_T1.csv       record id, repeat, fold   (scaffold-disjoint, 5 repeats x 5)
    folds_T2.csv       record id, fold           (publication-disjoint)
    folds_T3.csv       record id, fold           (publication-disjoint, test restricted
                                                  to pairs called positive by >=1
                                                  TRAINING source of that fold)
    labels_<task>.csv  per fold: how training labels were built and how many differ
                       from the observed call; test sources never inform a label
    leakage.csv        per fold: pairs / molecules / scaffolds / sources shared
    results.csv        per fold and model: AUC with bootstrap CI, PR-AUC, prevalence
    report.txt         the manuscript's numbers, regenerated

Reference models are library defaults with no tuning, so differences between
them are differences between defaults.
"""
import argparse, os, hashlib
import numpy as np
import pandas as pd

SEEDS = [1, 2, 3, 4, 5]
MIN_TEST, MIN_POS, MIN_SOURCE_RECORDS = 30, 5, 40
FP_BITS = 1024


# --------------------------------------------------------------- stage: folds
def build_records(m2or_path, smiles_path):
    d = pd.read_csv(m2or_path, sep=';', low_memory=False)
    d = d[d['species'].astype(str).str.contains('homo', case=False, na=False)]
    d = d[d['Mutation'].isna()]
    d = d[d['Mixture'].astype(str).str.lower() != 'mixture']
    d = d[d['InChI Key'].notna() & d['Responsive'].notna()].copy()
    # species prefix normalisation: hOR5AN1 and OR5AN1 are one receptor
    d['rec'] = d['Gene ID'].astype(str).str.replace(r'^h(?=OR)', '', regex=True).str.upper()
    d['key'] = d['rec'] + '|' + d['InChI Key'].astype(str)
    smi = pd.read_csv(smiles_path).drop_duplicates('InChIKey')
    d = d.merge(smi[['InChIKey', 'SMILES']], left_on='InChI Key', right_on='InChIKey', how='inner')

    R = (d.groupby(['key', 'DOI'], as_index=False)
           .agg(rec=('rec', 'first'), ikey=('InChI Key', 'first'), smi=('SMILES', 'first'),
                y=('Responsive', lambda s: int(s.dropna().mode().iloc[0]))))
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    sc = {}
    for s in R.smi.unique():
        mol = Chem.MolFromSmiles(s)
        sc[s] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None
    R = R[R.smi.map(sc).notna()].copy()
    R['scaf'] = R.smi.map(sc)
    R = R.sort_values(['key', 'DOI']).reset_index(drop=True)
    R.insert(0, 'record_id', ['r%06d' % i for i in range(len(R))])
    return R


def folds_T1(R):
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        u = np.array(R.scaf.unique(), dtype=object)
        rng.shuffle(u)
        f = R.scaf.map({v: i % 5 for i, v in enumerate(u)}).values
        for k in range(5):
            out.append(pd.DataFrame(dict(record_id=R.record_id.values[f == k],
                                         repeat=seed, fold='seed%d-fold%d' % (seed, k))))
    return pd.concat(out, ignore_index=True)


def eligible_sources(R):
    cnt = R.DOI.value_counts()
    return [s for s in cnt.index
            if cnt[s] >= MIN_SOURCE_RECORDS and R[R.DOI == s].y.nunique() == 2
            and R[R.DOI == s].y.sum() >= MIN_POS]


def folds_T2(R):
    return pd.concat([pd.DataFrame(dict(record_id=R.record_id.values[R.DOI.values == s],
                                        fold=s.split('/')[-1]))
                      for s in eligible_sources(R)], ignore_index=True)


def folds_T3(R):
    """Test records: pairs called positive by at least one source that stays in
    the training set for that fold."""
    rows = []
    for s in eligible_sources(R):
        tr = R[R.DOI != s]
        pos_keys = set(tr.key[tr.y == 1])
        sel = R[(R.DOI == s) & (R.key.isin(pos_keys))]
        if len(sel) >= MIN_TEST and sel.y.nunique() > 1 and sel.y.sum() >= MIN_POS:
            rows.append(pd.DataFrame(dict(record_id=sel.record_id.values,
                                          fold=s.split('/')[-1])))
    return pd.concat(rows, ignore_index=True)


def doi_of(R, fold):
    hits = [d for d in R.DOI.unique() if d.split('/')[-1] == fold]
    return hits[0] if hits else None


def train_index(R, folds, task, fold, te):
    if task == 'T1':
        return np.setdiff1d(np.arange(len(R)), te)
    return np.where(R.DOI.values != doi_of(R, fold))[0]


# --------------------------------------------------------------- stage: audit
def audit(R, folds, task):
    pos = {r: i for i, r in enumerate(R.record_id)}
    rows = []
    for fold, g in folds.groupby('fold'):
        te = np.array([pos[r] for r in g.record_id])
        tr = train_index(R, folds, task, fold, te)
        a, b = R.iloc[tr], R.iloc[te]
        rows.append(dict(task=task, fold=fold, n_train=len(a), n_test=len(b),
                         pairs_shared=len(set(a.key) & set(b.key)),
                         mols_shared=len(set(a.ikey) & set(b.ikey)),
                         scaffolds_shared=len(set(a.scaf) & set(b.scaf)),
                         sources_shared=len(set(a.DOI) & set(b.DOI))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ stage: evaluate
def featurise(R):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    fp = {s: np.array(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.uint8)
          for s in R.smi.unique()}
    F = np.stack(R.smi.map(fp).values).astype(np.float32)
    recs = sorted(R.rec.unique())
    Rm = np.zeros((len(R), len(recs)), dtype=np.float32)
    Rm[np.arange(len(R)), R.rec.map({r: i for i, r in enumerate(recs)})] = 1
    return np.hstack([F, Rm])


def reference_models():
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    return {
        'RandomForest': lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                                       n_jobs=-1, random_state=0,
                                                       class_weight='balanced'),
        'Logistic': lambda: LogisticRegression(max_iter=1500, C=0.1, class_weight='balanced'),
        'HistGB': lambda: HistGradientBoostingClassifier(max_iter=150, random_state=0),
        'ReceptorPrior': 'prior',
    }


def boot_ci(y, p, n=400, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def evaluate(R, X, folds, task, use_consensus, outdir):
    from sklearn.metrics import roc_auc_score, average_precision_score
    pos = {r: i for i, r in enumerate(R.record_id)}
    y = R.y.values
    rows, labrows = [], []
    for fold, g in folds.groupby('fold'):
        te = np.array([pos[r] for r in g.record_id])
        tr = train_index(R, folds, task, fold, te)
        if len(te) < MIN_TEST or len(np.unique(y[te])) < 2 or y[te].sum() < MIN_POS:
            continue
        if use_consensus:
            cons = pd.Series(y[tr], index=R.key.values[tr]).groupby(level=0).mean()
            ytr = (pd.Series(R.key.values[tr]).map(cons).values > 0.5).astype(int)
            src = 'pair consensus rebuilt from training sources only'
        else:
            ytr = y[tr]
            src = 'observed call of the training source'
        labrows.append(dict(task=task, fold=fold, n_train=len(tr), label_source=src,
                            n_label_differs_from_observed=int((ytr != y[tr]).sum())))
        for name, mk in reference_models().items():
            if mk == 'prior':
                rate = pd.Series(ytr, index=R.rec.values[tr]).groupby(level=0).mean()
                p = np.array([rate.get(r, ytr.mean()) for r in R.rec.values[te]])
            else:
                mdl = mk()
                mdl.fit(X[tr], ytr)
                p = mdl.predict_proba(X[te])[:, 1]
            lo, hi = boot_ci(y[te], p)
            rows.append(dict(task=task, fold=fold, model=name, n_test=len(te),
                             prev=float(y[te].mean()), auc=roc_auc_score(y[te], p),
                             auc_lo=lo, auc_hi=hi, pr=average_precision_score(y[te], p),
                             pr_baseline=float(y[te].mean())))
        print('  %-3s %-26s n=%6d prev=%.3f' % (task, fold[:26], len(te), y[te].mean()), flush=True)
    pd.DataFrame(labrows).to_csv(os.path.join(outdir, 'labels_%s.csv' % task), index=False)
    return pd.DataFrame(rows)


# -------------------------------------------------------------- stage: report
def report(outdir):
    B = pd.read_csv(os.path.join(outdir, 'results.csv'))
    L = []
    P = B.pivot_table(index='task', columns='model', values='auc')
    cols = [c for c in ['RandomForest', 'Logistic', 'HistGB', 'ReceptorPrior'] if c in P.columns]
    L.append('Table 1 - mean AUC by task and model')
    L.append(P[cols].round(3).to_string())
    B = B.assign(width=B.auc_hi - B.auc_lo, lift=B.pr / B.prev)
    for t in sorted(B.task.unique()):
        w = B[B.task == t].pivot(index='fold', columns='model', values='auc')
        A = B[B.task == t]
        L.append('\n%s: %d folds' % (t, len(w)))
        for m in ['Logistic', 'HistGB', 'RandomForest']:
            d = w.ReceptorPrior - w[m]
            L.append('  prior - %-13s mean %+.3f, prior ahead in %d/%d folds'
                     % (m, d.mean(), (d > 0).sum(), len(d)))
        # The reporting unit is the fold, so both summaries below are medians
        # over folds, never ratios of means. Interval width is pooled over all
        # four reference models so that it rests on no choice of model; PR lift
        # is given per model, because averaging a lift over a deliberately
        # chance-level baseline would not mean anything.
        L.append('  median CI width %.3f (all four models pooled)'
                 % A.width.median())
        L.append('  median per-fold PR-AUC / prevalence: '
                 + ', '.join('%s %.2f' % (m, A[A.model == m].lift.median())
                             for m in cols))
        L.append('  AUC range %.3f-%.3f' % (A.auc.min(), A.auc.max()))
    txt = '\n'.join(L)
    open(os.path.join(outdir, 'report.txt'), 'w').write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m2or', default='M2OR_20230428.csv')
    ap.add_argument('--smiles', default='ikey_smiles.csv')
    ap.add_argument('--out', default='bench_out')
    ap.add_argument('--stage', default='all',
                    choices=['folds', 'audit', 'evaluate', 'report', 'all'])
    ap.add_argument('--tasks', default='T1,T2,T3')
    global MIN_TEST, MIN_POS, MIN_SOURCE_RECORDS
    ap.add_argument('--min-test', type=int, default=MIN_TEST,
                    help='minimum scored records in a test fold (default 30)')
    ap.add_argument('--min-pos', type=int, default=MIN_POS,
                    help='minimum positives in a test fold (default 5)')
    ap.add_argument('--min-source-records', type=int, default=MIN_SOURCE_RECORDS,
                    help='minimum records for a source to form a fold (default 40)')
    a = ap.parse_args()
    # the eligibility thresholds are conventions; varying them here is what the
    # supplement's threshold-sensitivity table reports.
    MIN_TEST, MIN_POS, MIN_SOURCE_RECORDS = (a.min_test, a.min_pos,
                                             a.min_source_records)
    if (MIN_TEST, MIN_POS, MIN_SOURCE_RECORDS) != (30, 5, 40):
        print('eligibility thresholds: min_test=%d min_pos=%d min_source=%d'
              % (MIN_TEST, MIN_POS, MIN_SOURCE_RECORDS))
    os.makedirs(a.out, exist_ok=True)
    rec_path = os.path.join(a.out, 'records.csv')
    want = [t.strip() for t in a.tasks.split(',') if t.strip()]

    if a.stage in ('folds', 'all'):
        R = build_records(a.m2or, a.smiles)
        R.to_csv(rec_path, index=False)
        folds_T1(R).to_csv(os.path.join(a.out, 'folds_T1.csv'), index=False)
        folds_T2(R).to_csv(os.path.join(a.out, 'folds_T2.csv'), index=False)
        folds_T3(R).to_csv(os.path.join(a.out, 'folds_T3.csv'), index=False)
        print('records %d  sha256 %s'
              % (len(R), hashlib.sha256(open(rec_path, 'rb').read()).hexdigest()[:16]))

    if a.stage == 'folds':
        return

    R = pd.read_csv(rec_path)
    F = {t: pd.read_csv(os.path.join(a.out, 'folds_%s.csv' % t)) for t in ['T1', 'T2', 'T3']}

    if a.stage in ('audit', 'all'):
        pd.concat([audit(R, F[t], t) for t in want], ignore_index=True) \
          .to_csv(os.path.join(a.out, 'leakage.csv'), index=False)
        print('leakage audit written')

    if a.stage in ('evaluate', 'all'):
        X = featurise(R)
        parts = []
        for t in want:
            print('evaluating %s ...' % t, flush=True)
            parts.append(evaluate(R, X, F[t], t, t == 'T1', a.out))
            pd.concat(parts, ignore_index=True).to_csv(os.path.join(a.out, 'results.csv'), index=False)

    if a.stage in ('report', 'all'):
        report(a.out)


if __name__ == '__main__':
    main()
