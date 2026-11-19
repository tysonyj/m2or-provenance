"""O4: cross-document label agreement on ChEMBL, and the pre-registered
STANDARDISED vs EXPLORATORY comparison. No model fitting."""
import json, os, random, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/repo/m2or-provenance")

STANDARDISED = {"KCNH2", "CYP3A4", "CYP2C9", "CYP2D6"}

def kappa(pairs):
    n = len(pairs)
    if not n: return float("nan"), float("nan")
    a = sum(1 for x, y in pairs if x == 1 and y == 1)
    b = sum(1 for x, y in pairs if x == 1 and y == 0)
    c = sum(1 for x, y in pairs if x == 0 and y == 1)
    d = n - a - b - c
    po = (a + d) / n
    p1 = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    return (po - p1) / (1 - p1) if p1 < 1 else float("nan"), 1 - po

def collect_pairs(R):
    """One modal call per (pair, document); all ordered-free comparisons."""
    call = (R.groupby(["rec", "key", "DOI"]).y
             .agg(lambda s: int(s.mean() > 0.5)).reset_index())
    by_rec = {}
    for (rec, key), g in call.groupby(["rec", "key"]):
        v = g.y.tolist()
        if len(v) < 2: continue
        cmps = [(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
        by_rec.setdefault(rec, []).extend(cmps)
    return by_rec

def summarise(by_rec, label, nboot=1000, seed=0):
    allp = [p for g in by_rec.values() for p in g]
    if len(allp) < 2:
        print("  %-26s too few comparisons (%d)" % (label, len(allp))); return None
    k, disc = kappa(allp)
    recs = list(by_rec); rng = random.Random(seed); ks, ds = [], []
    for _ in range(nboot):
        drawn = [by_rec[recs[rng.randrange(len(recs))]] for _ in recs]
        pp = [p for g in drawn for p in g]
        if not pp: continue
        kk, dd = kappa(pp)
        if kk == kk: ks.append(kk); ds.append(dd)
    q = lambda v, t: sorted(v)[max(0, min(len(v) - 1, int(t * (len(v) - 1))))]
    print("  %-26s targets %2d  comparisons %6d  discordance %.4f [%.4f, %.4f]  "
          "kappa %.4f [%.4f, %.4f]"
          % (label, len(recs), len(allp), disc, q(ds, .025), q(ds, .975),
             k, q(ks, .025), q(ks, .975)))
    return dict(group=label, targets=len(recs), comparisons=len(allp),
                discordance=disc, disc_lo=q(ds, .025), disc_hi=q(ds, .975),
                kappa=k, kappa_lo=q(ks, .025), kappa_hi=q(ks, .975))

if __name__ == "__main__":
    R = pd.read_csv("out/records.csv")
    print("records %d, targets %d, documents %d" % (len(R), R.rec.nunique(), R.DOI.nunique()))
    by_rec = collect_pairs(R)
    print("\nO4  cross-document agreement (receptor-clustered bootstrap, 1000 replicates)")
    rows = [summarise(by_rec, "all targets")]
    std = {k: v for k, v in by_rec.items() if k in STANDARDISED}
    exp = {k: v for k, v in by_rec.items() if k not in STANDARDISED}
    rows.append(summarise(std, "STANDARDISED"))
    rows.append(summarise(exp, "EXPLORATORY"))
    print("\n  per target:")
    per = []
    for rec in sorted(by_rec):
        r = summarise({rec: by_rec[rec]}, "  " + rec, nboot=400)
        if r: r["group"] = rec; per.append(r)
    pd.DataFrame([r for r in rows if r] + per).to_csv("out/o4_agreement.csv", index=False)
    ks = {r["group"]: r["kappa"] for r in per}
    s = [ks[t] for t in ks if t in STANDARDISED]; e = [ks[t] for t in ks if t not in STANDARDISED]
    if s and e:
        from scipy.stats import mannwhitneyu
        p = mannwhitneyu(s, e, alternative="greater").pvalue
        print("\n  pre-registered one-sided test (STANDARDISED kappa > EXPLORATORY):")
        print("    medians %.3f (n=%d) vs %.3f (n=%d), Mann-Whitney p = %.3f"
              % (np.median(s), len(s), np.median(e), len(e), p))
