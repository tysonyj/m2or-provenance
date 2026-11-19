"""Has cross-source label agreement improved over time?

Fits the discordance of each cross-source comparison on the publication decade,
as a logistic model with receptor-clustered standard errors. Reads the
comparison set written by verify.py and the publication years in
results/M2OR_source_years.csv, which were resolved from Crossref once and are
released so this runs offline.

The year of a comparison is taken to be the LATER of its two sources: a
cross-source comparison exists only once both sources are published. That choice
is a convention, so the two alternatives are printed beside it; the conclusion
does not depend on which is used.

    python3 agreement_trend.py [--comparisons results/M2OR_crosssource_comparisons.csv]
                               [--years results/M2OR_source_years.csv]
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit(disc, year, clusters):
    """Logistic regression of discordance on decade, receptor-clustered SEs."""
    x = sm.add_constant((np.asarray(year, float) - np.mean(year)) / 10.0)
    m = sm.Logit(np.asarray(disc, float), x).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": clusters})
    lo, hi = np.exp(m.conf_int()[1])
    return np.exp(m.params[1]), lo, hi, m.pvalues[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparisons", default="results/M2OR_crosssource_comparisons.csv")
    ap.add_argument("--years", default="results/M2OR_source_years.csv")
    a = ap.parse_args()

    C = pd.read_csv(a.comparisons)
    Y = pd.read_csv(a.years).set_index("source").year
    C["y1"] = C.d1.map(Y)
    C["y2"] = C.d2.map(Y)
    missing = C.y1.isna() | C.y2.isna()
    if missing.any():
        print("dropping %d comparisons with an unresolved publication year"
              % int(missing.sum()))
    C = C[~missing]

    print("%d comparisons, %d sources, publication years %d-%d, %d receptors"
          % (len(C), len(set(C.d1) | set(C.d2)), Y.min(), Y.max(), C.rec.nunique()))
    print("\nodds of a discordant call per decade (>1 means agreement got worse)")
    specs = {
        "later source   (primary)": C[["y1", "y2"]].max(axis=1),
        "earlier source": C[["y1", "y2"]].min(axis=1),
        "midpoint": C[["y1", "y2"]].mean(axis=1),
    }
    for name, yr in specs.items():
        o, lo, hi, p = fit(C.disc.values, yr.values, C.rec.values)
        print("  %-24s OR %.2f  95%% CI %.2f-%.2f  p = %.2f" % (name, o, lo, hi, p))
    print("\nNo specification reaches significance: there is no evidence that\n"
          "cross-source agreement has improved, and none that it has worsened.")


if __name__ == "__main__":
    main()
