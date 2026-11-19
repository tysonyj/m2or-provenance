"""Run the M2OR-Provenance three-task benchmark on ChEMBL, unchanged.

The point of this script is that it does not reimplement the benchmark. It
builds a records table with the column names the released harness expects and
then calls the harness's own fold construction, featurisation, reference models
and bootstrap. Whatever the method does to M2OR, it does here.

Design is frozen in PRESPECIFICATION.md; this script implements it and nothing
else. Run after collect.py:

    python3 external_benchmark.py [--pact 6.0] [--out out]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# The parent directory holds the benchmark harness this script adapts.
# Override with M2OR_HARNESS if you keep the two elsewhere.
HARNESS = os.environ.get(
    "M2OR_HARNESS",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if not os.path.isfile(os.path.join(HARNESS, "run_benchmark.py")):
    sys.exit(
        f"run_benchmark.py not found in {HARNESS}\n"
        "Run this from external_validation/ inside the repository, or set "
        "M2OR_HARNESS to the directory holding run_benchmark.py."
    )
sys.path.insert(0, HARNESS)
import run_benchmark as rb                                    # noqa: E402

STANDARDISED = {"KCNH2", "CYP3A4", "CYP2C9", "CYP2D6"}
KEEP_TYPES = {"IC50", "Ki", "Kd", "EC50"}


def load(pages="pages"):
    """Every collected record, with the frozen filters of section 3 applied."""
    rows, counts = [], {}
    for path in sorted(glob.glob(os.path.join(pages, "*.jsonl"))):
        gene = os.path.basename(path)[:-6]
        n = 0
        for line in open(path):
            rows.append(dict(json.loads(line), gene=gene))
            n += 1
        counts[gene] = n
    d = pd.DataFrame(rows)
    print("collected records per target:")
    for g, n in sorted(counts.items()):
        print("  %-8s %7d" % (g, n))
    print("  %-8s %7d" % ("TOTAL", len(d)))

    step = [("collected", len(d))]
    d = d[d.standard_type.isin(KEEP_TYPES)];            step.append(("type in {IC50,Ki,Kd,EC50}", len(d)))
    d = d[d.standard_units == "nM"];                    step.append(("units nM", len(d)))
    d["val"] = pd.to_numeric(d.standard_value, errors="coerce")
    d = d[d.val.notna() & (d.val > 0)];                 step.append(("value present and > 0", len(d)))
    d = d[d.standard_relation == "="];                  step.append(("relation '='", len(d)))
    d = d[d.data_validity_comment.isna()];              step.append(("no validity flag", len(d)))
    d = d[d.document_chembl_id.notna()];                step.append(("document present", len(d)))
    d = d[d.canonical_smiles.notna()];                  step.append(("SMILES present", len(d)))
    print("\nfilter cascade (frozen specification, section 3):")
    for name, n in step:
        print("  %-30s %7d" % (name, n))
    return d.reset_index(drop=True)


def build_records(d, pact_cut):
    """A records table in the column names the harness expects."""
    d = d.copy()
    d["pact"] = 9.0 - np.log10(d.val)
    d["y"] = (d.pact >= pact_cut).astype(int)
    R = pd.DataFrame({
        "rec": d.gene.values,                     # the "receptor" is the target
        "smi": d.canonical_smiles.values,
        "DOI": d.document_chembl_id.values,        # the source is the publication
        "y": d.y.values,
    })
    R["ikey"] = d.molecule_chembl_id.values      # the molecule identifier
    R["key"] = R.rec + "|" + R.ikey
    # Bemis-Murcko scaffolds, by exactly the harness's own code path, and
    # molecules RDKit cannot parse are dropped so featurisation cannot fail
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    sc = {}
    for smi in R.smi.unique():
        mol = Chem.MolFromSmiles(smi)
        sc[smi] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else None
    bad = len(R) - int(R.smi.map(sc).notna().sum())
    if bad:
        print("  dropped %d records whose SMILES RDKit could not parse" % bad)
    R = R[R.smi.map(sc).notna()].copy()
    R["scaf"] = R.smi.map(sc)
    R = R.sort_values(["key", "DOI"]).reset_index(drop=True)
    R["record_id"] = ["r%06d" % i for i in range(len(R))]
    return R


def target_eligibility(R):
    """Section 2: a target needs >= 200 records from >= 3 documents."""
    g = R.groupby("rec").agg(records=("record_id", "size"),
                             documents=("DOI", "nunique"),
                             positives=("y", "sum"))
    g["admitted"] = (g.records >= 200) & (g.documents >= 3)
    print("\ntarget eligibility (>= 200 records from >= 3 documents):")
    for rec, r in g.sort_values("records", ascending=False).iterrows():
        print("  %-8s %7d records  %4d documents  %6d positives  %s"
              % (rec, r.records, r.documents, r.positives,
                 "admitted" if r.admitted else "EXCLUDED"))
    return R[R.rec.isin(g.index[g.admitted])].reset_index(drop=True), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pact", type=float, default=6.0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--tasks", default="T1,T2,T3")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="process at most this many uncached batches, then exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Prefer an analysis set already built: the released results/records.csv.gz
    # lets the scoring be rerun without re-collecting from the ChEMBL API.
    # Delete out/records.csv (or pass a different --out) to rebuild from
    # the raw pages/ directory that collect.py writes.
    rp = os.path.join(a.out, "records.csv")
    if os.path.isfile(rp):
        R = pd.read_csv(rp)
        print("using the analysis set already in %s (%d records)" % (rp, len(R)))
        if a.pact != 6.0:
            sys.exit("--pact only applies when rebuilding; remove %s first" % rp)
        elig = None
    else:
        d = load()
        R = build_records(d, a.pact)
        R, elig = target_eligibility(R)
        elig.to_csv(os.path.join(a.out, "target_eligibility.csv"))

    print("\nrecords entering the benchmark: %d" % len(R))
    print("  targets %d, molecules %d, pairs %d, documents %d, positive rate %.4f"
          % (R.rec.nunique(), R.smi.nunique(), R.key.nunique(),
             R.DOI.nunique(), R.y.mean()))
    if elig is not None:
        R.to_csv(rp, index=False)

    # ---- the harness's own fold construction, unchanged
    folds = {}
    for task, fn in [("T1", rb.folds_T1), ("T2", rb.folds_T2), ("T3", rb.folds_T3)]:
        f = fn(R)
        folds[task] = f
        f.to_csv(os.path.join(a.out, "folds_%s.csv" % task), index=False)
        print("  %s: %d folds, %d rows" % (task, f.fold.nunique(), len(f)))

    # ---- the harness's own leakage audit, unchanged
    aud = []
    for task in folds:
        aud.append(rb.audit(R, folds[task], task))
    pd.concat(aud).to_csv(os.path.join(a.out, "leakage.csv"), index=False)

    # ---- the harness's own featurisation, models and bootstrap, unchanged
    xp = os.path.join(a.out, "X.npy")
    if os.path.exists(xp):
        X = np.load(xp)
        print("  design matrix %s (cached)" % (X.shape,))
    else:
        X = rb.featurise(R)
        np.save(xp, X)
        print("  design matrix %s (computed and cached)" % (X.shape,))

    # Results are written per batch as they finish. train_index for T2 and T3
    # selects the training set from R.DOI alone and never from the fold table,
    # so evaluating a subset of folds gives byte-identical training sets to
    # evaluating them all at once: batching is a checkpoint, not a change of
    # method.
    BATCH = a.batch
    budget = a.max_batches or 10**9
    T2_SAMPLE = 50          # see the addendum to PRESPECIFICATION.md
    for task in a.tasks.split(","):
        f = folds[task]
        names = sorted(f.fold.unique())
        if task == "T2" and len(names) > T2_SAMPLE:
            rng = np.random.default_rng(0)
            names = sorted(rng.choice(names, T2_SAMPLE, replace=False).tolist())
            pd.Series(names, name="fold").to_csv(
                os.path.join(a.out, "t2_sampled_folds.csv"), index=False)
            print("  T2 restricted to a seed-0 random sample of %d of %d documents"
                  % (len(names), f.fold.nunique()), flush=True)
            f = f[f.fold.isin(names)]
        chunks = [names[i:i + BATCH] for i in range(0, len(names), BATCH)]
        parts = []
        for bi, chunk in enumerate(chunks):
            out = os.path.join(a.out, "results_%s_b%02d.csv" % (task, bi))
            if os.path.exists(out):
                parts.append(pd.read_csv(out))
                continue
            if budget <= 0:
                print("  %s batch %d/%d PENDING (budget spent)"
                      % (task, bi + 1, len(chunks)), flush=True)
                continue
            print("  %s batch %d/%d (%d folds)"
                  % (task, bi + 1, len(chunks), len(chunk)), flush=True)
            res = rb.evaluate(R, X, f[f.fold.isin(chunk)], task,
                              use_consensus=(task == "T1"), outdir=a.out)
            res.to_csv(out, index=False)
            parts.append(res)
            budget -= 1
        if len(parts) == len(chunks):
            pd.concat(parts, ignore_index=True).to_csv(
                os.path.join(a.out, "results_%s.csv" % task), index=False)
            print("  %s COMPLETE (%d folds)" % (task, len(chunks) and
                  pd.concat(parts).fold.nunique()), flush=True)
        else:
            print("  %s partial: %d of %d batches done"
                  % (task, len(parts), len(chunks)), flush=True)

    have = [t for t in a.tasks.split(",")
            if os.path.exists(os.path.join(a.out, "results_%s.csv" % t))]
    if have:
        allres = pd.concat([pd.read_csv(os.path.join(a.out, "results_%s.csv" % t))
                            for t in have], ignore_index=True)
        allres.to_csv(os.path.join(a.out, "results.csv"), index=False)
        print("\nrolled up %s: %d rows" % ("+".join(have), len(allres)))


if __name__ == "__main__":
    main()
