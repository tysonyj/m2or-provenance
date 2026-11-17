"""Two controls on the new-source result, released with their data.

Both ask whether the drop in performance when a whole publication is held out
has an explanation other than the one the paper gives.

  design   Does it depend on what kind of study the held-out source is? Sources
           are split by how many distinct molecules they contribute, a property
           of the source and not of any result, and the held-out-source
           evaluation is reported within each group.

  latent   Does it go away if the labelling process is modelled? A Dawid-Skene
           latent-class model is fitted to the training sources' calls, and a
           learner is trained on the posterior probability that a pair is a true
           agonist instead of on the observed calls.

Run after `run_benchmark.py --stage folds`:
    python3 controls.py [--out bench_out] [--model HistGB]

Writes controls_design.csv and controls_latent.csv into --out.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import run_benchmark as rb

# A source is a broad screen if it contributes at least this many distinct
# molecules. Fixed before the evaluation below is run; the sensitivity of the
# split to this number is reported by --design-threshold.
BROAD_MOLECULES = 50


# ------------------------------------------------------------ Dawid and Skene
def dawid_skene(obs, n_iter=200, tol=1e-7, seed=0):
    """Two-class Dawid-Skene EM on a pair-by-source call matrix.

    obs: integer array, 0 or 1 for a call and -1 for "this source did not
    measure this pair". Returns the posterior probability that each pair is a
    true positive, together with each source's sensitivity and specificity.
    """
    obs = np.asarray(obs)
    n_items, n_src = obs.shape
    seen = obs >= 0
    rng = np.random.default_rng(seed)
    # initialise on the observed majority, with ties broken at random
    with np.errstate(invalid="ignore"):
        frac = np.where(seen.sum(1) > 0,
                        np.where(seen, obs, 0).sum(1) / np.maximum(seen.sum(1), 1),
                        0.5)
    q = np.clip(frac + 1e-3 * rng.standard_normal(n_items), 1e-3, 1 - 1e-3)
    prev = -np.inf
    for _ in range(n_iter):
        # M step: class prior, and each source's sensitivity and specificity
        pi = float(np.clip(q.mean(), 1e-6, 1 - 1e-6))
        sens = np.empty(n_src)
        spec = np.empty(n_src)
        for j in range(n_src):
            m = seen[:, j]
            qj, oj = q[m], obs[m, j]
            sens[j] = np.clip(((qj * (oj == 1)).sum() + 1.0)
                              / (qj.sum() + 2.0), 1e-3, 1 - 1e-3)
            spec[j] = np.clip((((1 - qj) * (oj == 0)).sum() + 1.0)
                              / ((1 - qj).sum() + 2.0), 1e-3, 1 - 1e-3)
        # E step
        lp = np.full(n_items, np.log(pi))
        ln = np.full(n_items, np.log1p(-pi))
        for j in range(n_src):
            m = seen[:, j]
            o = obs[m, j]
            lp[m] += np.where(o == 1, np.log(sens[j]), np.log1p(-sens[j]))
            ln[m] += np.where(o == 1, np.log1p(-spec[j]), np.log(spec[j]))
        mx = np.maximum(lp, ln)
        q = np.exp(lp - mx) / (np.exp(lp - mx) + np.exp(ln - mx))
        ll = float((mx + np.log(np.exp(lp - mx) + np.exp(ln - mx))).sum())
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    return q, sens, spec


def latent_posteriors(train):
    """Posterior P(true agonist) for every pair in a training fold."""
    call = (train.groupby(["key", "DOI"]).y
            .agg(lambda s: int(s.mean() > 0.5)).reset_index())
    keys = sorted(call.key.unique())
    srcs = sorted(call.DOI.unique())
    ki = {k: i for i, k in enumerate(keys)}
    si = {s: i for i, s in enumerate(srcs)}
    obs = np.full((len(keys), len(srcs)), -1, dtype=int)
    obs[call.key.map(ki).values, call.DOI.map(si).values] = call.y.values
    q, sens, spec = dawid_skene(obs)
    return pd.Series(q, index=keys), len(srcs), sens, spec


# ------------------------------------------------------------------ the runs
def fit_predict(name, Xtr, ytr, Xte, weight=None):
    mk = rb.reference_models()[name]
    m = mk()
    if weight is None:
        m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr, sample_weight=weight)
    return m.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--model", default="HistGB")
    ap.add_argument("--design-threshold", type=int, default=BROAD_MOLECULES)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    R = pd.read_csv(os.path.join(a.out, "records.csv"))
    folds = pd.read_csv(os.path.join(a.out, "folds_T2.csv"))
    X = rb.featurise(R)
    rid = {r: i for i, r in enumerate(R.record_id)}

    design, latent = [], []
    print("held-out source evaluation, model = %s" % a.model)
    for fold in sorted(folds.fold.unique()):
        te_ids = folds.record_id[folds.fold == fold].values
        te = np.array([rid[r] for r in te_ids])
        tr = np.setdiff1d(np.arange(len(R)), te)
        ytr, yte = R.y.values[tr], R.y.values[te]
        if len(np.unique(yte)) < 2:
            continue
        train, test = R.iloc[tr], R.iloc[te]

        # ---- control 1: what kind of study is the held-out source?
        n_mol = test.key.str.split("|").str[1].nunique()
        n_rec = test.rec.nunique()
        p_obs = fit_predict(a.model, X[tr], ytr, X[te])
        auc_obs = roc_auc_score(yte, p_obs)
        design.append(dict(fold=fold, molecules=n_mol, receptors=n_rec,
                           n_test=len(te), prevalence=float(yte.mean()),
                           group="broad screen" if n_mol >= a.design_threshold
                           else "targeted study", auc=auc_obs))
        print("  %-28s mol %4d  rec %3d  AUC %.3f  (%s)"
              % (fold[:28], n_mol, n_rec, auc_obs, design[-1]["group"]))

        # ---- control 2: train on latent-truth posteriors instead
        # The model is fitted to the training sources only, so nothing from the
        # held-out source reaches it.
        q, n_src, sens, spec = latent_posteriors(train)
        soft = train.key.map(q)
        if soft.isna().any() or n_src < 3:
            continue
        # A posterior is a soft label: fit the hard label it implies, weighted
        # by how confident the latent model is.
        y_lat = (soft.values > 0.5).astype(int)
        w = np.abs(soft.values - 0.5) * 2 + 1e-3
        if len(np.unique(y_lat)) < 2:
            continue
        p_lat = fit_predict(a.model, X[tr], y_lat, X[te], weight=w)
        latent.append(dict(fold=fold, n_train_sources=n_src,
                           n_relabelled=int((y_lat != ytr).sum()),
                           auc_observed=auc_obs,
                           auc_latent=roc_auc_score(yte, p_lat),
                           median_sensitivity=float(np.median(sens)),
                           median_specificity=float(np.median(spec))))
        print("      latent: relabelled %5d of %5d, AUC %.3f -> %.3f"
              % (latent[-1]["n_relabelled"], len(tr), auc_obs,
                 latent[-1]["auc_latent"]))

    D = pd.DataFrame(design)
    L = pd.DataFrame(latent)
    D.to_csv(os.path.join(a.out, "controls_design.csv"), index=False)
    L.to_csv(os.path.join(a.out, "controls_latent.csv"), index=False)

    print("\n== study design ==")
    for g, s in D.groupby("group"):
        print("  %-15s n=%2d  mean AUC %.3f  median %.3f"
              % (g, len(s), s.auc.mean(), s.auc.median()))
    if D.group.nunique() == 2:
        from scipy.stats import mannwhitneyu
        b = D.auc[D.group == "broad screen"]
        t = D.auc[D.group == "targeted study"]
        print("  Mann-Whitney p = %.3f" % mannwhitneyu(b, t).pvalue)

    print("\n== latent-truth relabelling ==")
    if len(L):
        d = L.auc_latent - L.auc_observed
        from scipy.stats import wilcoxon
        print("  folds %d, observed %.3f, latent %.3f, mean change %+.3f"
              % (len(L), L.auc_observed.mean(), L.auc_latent.mean(), d.mean()))
        print("  latent better in %d of %d, Wilcoxon p = %.3f"
              % (int((d > 0).sum()), len(d), wilcoxon(d).pvalue))
    print("\nwrote controls_design.csv and controls_latent.csv to %s" % a.out)


if __name__ == "__main__":
    main()
