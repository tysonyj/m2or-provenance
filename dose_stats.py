"""Model-free dose-dependence and label-composition statistics.

Recomputes, from the M2OR release alone, the figures quoted in the manuscript
section "Why provenance has to be kept" and in SI "Composition of the audited
label set" and "Concentration analyses".  Every count printed here is quoted in
the text, so a reader can check the text against this file's output.

Run:  python3 dose_stats.py
Env:  M2OR_CSV  path to the release (default M2OR_20230428.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

SRC = os.environ.get("M2OR_CSV", "M2OR_20230428.csv")


def analysis_set(path):
    """The analysis set, built by exactly the filters run_benchmark.py applies."""
    d = pd.read_csv(path, sep=";", low_memory=False)
    n_release = len(d)
    d = d[d["species"].astype(str).str.contains("homo", case=False, na=False)]
    d = d[d["Mutation"].isna()]
    d = d[d["Mixture"].astype(str).str.lower() != "mixture"]
    d = d[d["InChI Key"].notna() & d["Responsive"].notna()].copy()
    # hOR5AN1 and OR5AN1 are one receptor; see SI "receptor identifier normalisation"
    d["rec"] = d["Gene ID"].astype(str).str.replace(r"^h(?=OR)", "", regex=True).str.upper()
    d["key"] = d["rec"] + "|" + d["InChI Key"].astype(str)
    d["y"] = pd.to_numeric(d["Responsive"], errors="coerce")
    d["unit"] = d["Unit_Screen"]
    d["val"] = pd.to_numeric(d["Value_Screen"], errors="coerce")
    return d, n_release


def unanimity(frame):
    """Replication status of each pair, by record unanimity.

    A replicated pair is unanimously negative only if no contributing source
    produced a single positive record, and unanimously positive only if every
    record from every source is positive.  Anything else is contested.  This is
    deliberately conservative: a pair on which one source contradicts itself is
    contested, not unanimous.
    """
    by_pair = frame.groupby("key").y
    all_neg = by_pair.max() == 0
    all_pos = by_pair.min() == 1
    return all_neg, all_pos


def main():
    if not os.path.exists(SRC):
        sys.exit("M2OR release not found at %s (set M2OR_CSV)" % SRC)
    d, n_release = analysis_set(SRC)
    N = len(d)

    print("== analysis set ==")
    print("release records                     %6d" % n_release)
    print("analysis-set records                %6d" % N)
    print("receptor identifiers                %6d" % d["rec"].nunique())
    print("molecules                           %6d" % d["InChI Key"].nunique())
    print("receptor-molecule pairs             %6d" % d["key"].nunique())
    print("source publications                 %6d" % d["DOI"].nunique())
    print("record-level positive rate          %.4f" % d.y.mean())

    print()
    print("== screening-concentration fields ==")
    nounit = int(d.unit.isna().sum())
    unit_noval = int((d.unit.notna() & d.val.isna()).sum())
    both = d.unit.notna() & d.val.notna()
    nonpos = int((both & (d.val <= 0)).sum())
    usable = both & (d.val > 0)
    print("unit absent                         %6d" % nounit)
    print("unit present, value absent          %6d" % unit_noval)
    print("unit and value present              %6d  (%.1f%% of analysis set)"
          % (int(both.sum()), 100.0 * both.sum() / N))
    print("  of which value <= 0               %6d" % nonpos)
    print("unit, value present and positive    %6d  (%.1f%% of analysis set)"
          % (int(usable.sum()), 100.0 * usable.sum() / N))
    print("unit labels: " + ", ".join("%s %d" % (k, v)
          for k, v in d.unit.fillna("[absent]").value_counts().items()))
    print()
    print("  the %d records with a non-positive value are vehicle or blank rows:"
          % nonpos)
    print("    their positive rate is %.4f"
          % d.y[both & (d.val <= 0)].mean())
    print("    they are EXCLUDED: a 0 uM row is not a test concentration, and")
    print("    log10(0) is undefined, so the log-odds models cannot use them.")
    print("  the %d records with no unit or no value are largely dose-response"
          % (nounit + unit_noval))
    print("    experiments; their positive rate is %.4f"
          % d.y[d.unit.isna() | (d.unit.notna() & d.val.isna())].mean())
    print()
    print("  the concentration analyses use the %d records with a positive value:"
          % int(usable.sum()))
    w, wo = d[usable], d[~usable]
    print("  positive rate   with %.4f   without %.4f" % (w.y.mean(), wo.y.mean()))
    print("  receptors/molecules  with %d/%d   without %d/%d"
          % (w["rec"].nunique(), w["InChI Key"].nunique(),
             wo["rec"].nunique(), wo["InChI Key"].nunique()))

    # ---- model-free dose dependence -------------------------------------
    dd = d[usable].copy()
    dd["uM"] = np.where(dd.unit == "mM", dd.val * 1000.0, dd.val)
    dd["ctx"] = (dd.Cell_line.astype(str) + "|" + dd.Assay.astype(str)
                 + "|" + dd.Gprotein.astype(str))
    G = ["key", "DOI", "ctx"]
    nconc = dd.groupby(G).uM.nunique()
    multi = nconc[nconc > 1].index

    print()
    print("== model-free dose dependence ==")
    print("pair x source x assay-context groups            %6d" % len(nconc))
    print("  tested at more than one concentration         %6d" % len(multi))
    span = np.log10(dd.groupby(G).uM.max() / dd.groupby(G).uM.min())[multi]
    print("  median log10 concentration span               %6.2f" % span.median())
    print("  distinct concentration values in the database %6d" % dd.uM.nunique())

    def changed(frame, label):
        """frame: one row per (group, concentration). Count groups whose call differs."""
        f = frame.set_index(G).loc[multi]
        nun = f.groupby(level=[0, 1, 2]).y.nunique()
        ch = nun[nun > 1].index
        ordered = 0
        for _, s in f[f.index.isin(ch)].groupby(level=[0, 1, 2]):
            pos, neg = s.uM[s.y == 1], s.uM[s.y == 0]
            if len(pos) and len(neg) and pos.min() > neg.max():
                ordered += 1
        n = len(ch)
        print("  %-44s %6d  (%.1f%% of %d)" % (label + ", call changed", n,
                                               100.0 * n / len(multi), len(multi)))
        print("  %-44s %6d  (%.1f%% of those)"
              % (label + ", change in dose order", ordered,
                 100.0 * ordered / n if n else float("nan")))
        return n, ordered

    print()
    # Primary: every record on its own, over genuine test concentrations.
    changed(dd[G + ["uM", "y"]].copy(), "primary (record level)")
    print()
    # Sensitivity: a source's call at one concentration reduced to its modal
    # call there, so a source contradicting itself at a single concentration is
    # not counted as a dose effect.
    prim = dd.groupby(G + ["uM"]).y.agg(lambda s: int(s.mean() > 0.5)).reset_index()
    changed(prim, "sensitivity (modal call per concentration)")
    print()
    print("  for reference, including the %d vehicle rows inflates these to:" % nonpos)
    dz = d[both].copy()
    dz["uM"] = np.where(dz.unit == "mM", dz.val * 1000.0, dz.val)
    dz["ctx"] = (dz.Cell_line.astype(str) + "|" + dz.Assay.astype(str)
                 + "|" + dz.Gprotein.astype(str))
    nz = dz.groupby(G).uM.nunique()
    mz = nz[nz > 1].index
    fz = dz.set_index(G).loc[mz]
    nunz = fz.groupby(level=[0, 1, 2]).y.nunique()
    chz = nunz[nunz > 1].index
    oz = 0
    for _, s in fz[fz.index.isin(chz)].groupby(level=[0, 1, 2]):
        pos, neg = s.uM[s.y == 1], s.uM[s.y == 0]
        if len(pos) and len(neg) and pos.min() > neg.max():
            oz += 1
    print("    call changed %d, in dose order %d -- but a vehicle row is" % (len(chz), oz))
    print("    negative by construction, so these are not evidence of a")
    print("    dependence on which test concentration was chosen.")

    # ---- audited label set ----------------------------------------------
    print()
    print("== composition of the audited label set ==")
    nsrc = d.groupby("key").DOI.nunique()
    rep = nsrc[nsrc >= 2].index
    print("pairs                               %6d" % len(nsrc))
    print("  unreplicated (one source)         %6d" % int((nsrc < 2).sum()))
    print("  replicated (>=2 sources)          %6d" % len(rep))
    all_neg, all_pos = unanimity(d[d.key.isin(rep)])
    print("  of the replicated pairs:")
    print("    unanimously negative            %6d" % int(all_neg.sum()))
    print("    unanimously positive            %6d" % int(all_pos.sum()))
    print("    contested                       %6d"
          % (len(rep) - int(all_neg.sum()) - int(all_pos.sum())))
    print("  record-level positive rate, replicated pairs  %.4f"
          % d.y[d.key.isin(rep)].mean())
    print("  record-level positive rate, all records       %.4f" % d.y.mean())


def odds_ratios(dd):
    """Within-group logistic regression of the call on log10 concentration.

    Three nested specifications.  Only groups with variation in both the call
    and the concentration identify the slope, so each model is fitted on those,
    with group fixed effects and standard errors clustered on
    receptor-molecule pair.  The specification matters: the tighter the group,
    the larger the estimate and the fewer the observations that identify it.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  [statsmodels not installed -- odds ratios skipped]")
        return
    dd = dd.copy()
    dd["lc"] = np.log10(dd.uM)
    specs = [("within pair", ["key"]),
             ("within pair and source", ["key", "DOI"]),
             ("within pair, source and assay context", ["key", "DOI", "ctx"])]
    for label, keys in specs:
        g = dd.groupby(keys)
        ok = g.y.transform("nunique").gt(1) & g.lc.transform("nunique").gt(1)
        s = dd[ok]
        if not len(s):
            print("  %-38s no identifying groups" % label)
            continue
        gid = s.groupby(keys).ngroup()
        X = pd.get_dummies(gid, prefix="g", drop_first=True, dtype=float)
        X.index = s.index
        X.insert(0, "lc", s.lc.values)
        try:
            m = sm.Logit(s.y.values, sm.add_constant(X)).fit(
                disp=0, method="lbfgs", maxiter=800,
                cov_type="cluster", cov_kwds={"groups": s.key.values})
            b, se = m.params["lc"], m.bse["lc"]
            print("  %-38s OR %6.2f  (95%% CI %.2f-%.2f)  n=%d, groups=%d"
                  % (label, np.exp(b), np.exp(b - 1.96 * se),
                     np.exp(b + 1.96 * se), len(s), gid.nunique()))
        except Exception as exc:                        # pragma: no cover
            print("  %-38s fit failed: %s" % (label, exc))


if __name__ == "__main__":
    main()
    _d, _ = analysis_set(SRC)
    _ok = _d.unit.notna() & _d.val.notna() & (_d.val > 0)
    _dd = _d[_ok].copy()
    _dd["uM"] = np.where(_dd.unit == "mM", _dd.val * 1000.0, _dd.val)
    _dd["ctx"] = (_dd.Cell_line.astype(str) + "|" + _dd.Assay.astype(str)
                  + "|" + _dd.Gprotein.astype(str))
    print()
    print("== dose dependence, within-group logistic models ==")
    odds_ratios(_dd)
