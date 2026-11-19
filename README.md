# M2OR-Provenance

A three-task benchmark for olfactory receptor activity prediction, with fixed
folds, an enforced leakage audit and per-fold uncertainty.

[![DOI](https://zenodo.org/badge/DOI/REPLACE_WITH_ZENODO_DOI.svg)](https://doi.org/REPLACE_WITH_ZENODO_DOI)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

| Task | Split | Question |
|---|---|---|
| T1 new molecule | scaffold-disjoint, 5 repeated splits | Will this receptor respond to a chemotype not in training? |
| T2 new source | publication-disjoint | Will this laboratory's screen report a positive? |
| T3 replication | publication-disjoint, test restricted to pairs already called positive by a training source | Will an independent laboratory reproduce a reported positive? |

Olfactory receptor ligand prediction is usually evaluated by splitting a curated
database once and reporting one AUC. That number is being asked to answer three
different questions at once, and they do not give the same answer: no model
ranking survives across the three tasks, the width of the uncertainty on a
number differs more than five-fold between them, and a baseline that uses **no
chemical information at all** is not consistently beaten on two of the three.

Three rules are implemented in the harness rather than left to the user:

- **labels are rebuilt inside each fold** from training sources only, so a
  consensus label never carries information from the held-out source;
- **a four-way leakage audit** (pairs, molecules, scaffolds, sources) runs on
  every fold and is written out as a table;
- **the fold is the reporting unit**, each with a percentile bootstrap
  interval. Folds are never pooled into a single figure.

## What is in this repository

```
run_benchmark.py      the harness: folds -> leakage audit -> evaluate -> report
verify.py             independent re-implementation of the headline label statistics
dose_stats.py         label composition and dose dependence
controls.py           study-design split and Dawid-Skene relabelling controls
paper_figures.py      regenerates every figure in the paper and the supplement
agreement_trend.py    has cross-source agreement improved over two decades?
M2OR_20230428.csv     the M2OR release analysed (42 MB, redistributed unmodified)
ikey_smiles.csv       InChIKey -> SMILES, resolved from PubChem
results/              all per-fold results and result tables from our run
figures/              the figures as published
third_party/          the licence of the redistributed database
external_validation/  the same harness rebuilt on ChEMBL, with its frozen
                      specification, code and per-fold results
```

## Reference models

Four models run on every fold, all library defaults with no tuning: random
forest, L2 logistic regression and histogram gradient boosting on Morgan
fingerprints (radius 2, 1024 bits) with a receptor indicator, plus a **receptor
prior** that uses no chemistry and predicts each receptor's positive rate in the
training fold.

The prior is a diagnostic, not a recommended model. A result that does not clear
it has not been shown to use chemical structure. Running it costs nothing, and
we would ask anyone reporting on these data to report it.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. No GPU is used; everything is scikit-learn on CPU. `statsmodels`
is needed only for the odds-ratio models in `dose_stats.py` and `matplotlib`
only for `paper_figures.py`.

## Run, in order

Run the stages in this order. Each writes into `bench_out/` and the later
stages read what the earlier ones wrote.

### 1. Folds (about 2 minutes)

```bash
python run_benchmark.py --stage folds
```

Builds `bench_out/records.csv` (one row per pair-source observation) and
`folds_T1.csv`, `folds_T2.csv`, `folds_T3.csv`. Fold membership is written as
record ids **before any model is fitted**, so the splits can be inspected or
replaced without running anything else.

Expected: `records 37462  sha256 54c430b262c20361`, and
T1 25 folds / 187,310 rows, T2 17 folds / 33,053, T3 5 folds / 315.
**If the record count or hash differs, stop** - the input data is not the
release the manuscript analysed.

### 2. Leakage audit (seconds)

```bash
python run_benchmark.py --stage audit
```

Writes `bench_out/leakage.csv`. Expected maxima per task:

| task | pairs | molecules | scaffolds | sources |
|---|---|---|---|---|
| T1 | 0 | 0 | 0 | 28 |
| T2 | 1495 | 195 | 30 | 0 |
| T3 | 111 | 97 | 15 | 0 |

T1 shares no pair, molecule or scaffold; T2 and T3 share no source. The shared
pairs on T2 and T3 are intended - the question there is whether a new
laboratory's calls can be predicted, including for pairs measured elsewhere.

### 3. Evaluate (the long one)

One task at a time is easiest to monitor and restart:

```bash
python run_benchmark.py --stage evaluate --tasks T3   # ~10 min, 5 folds
python run_benchmark.py --stage evaluate --tasks T2   # ~45 min, 17 folds
python run_benchmark.py --stage evaluate --tasks T1   # ~90 min, 25 folds
```

**Note:** each invocation overwrites `bench_out/results.csv` with only the
tasks named in that run. To keep all three, either run them in one call

```bash
python run_benchmark.py --stage evaluate --tasks T3,T2,T1
```

or copy `results.csv` aside after each task and concatenate at the end.

Four reference models run on every fold: random forest, L2 logistic regression
and histogram gradient boosting on Morgan fingerprints (radius 2, 1024 bits)
with a receptor indicator, plus a **receptor prior** that uses no chemistry and
predicts each receptor's positive rate in the training fold. All are library
defaults with no tuning. Labels for T1 are pair consensus rebuilt inside each
fold from training sources only; `labels_<task>.csv` records this per fold.

### 4. Report

```bash
python run_benchmark.py --stage report
```

Prints and writes `bench_out/report.txt`: Table 1, and per task the pairwise
differences between the receptor prior and each learner, the median per-fold
bootstrap interval width pooled over all four reference models, the median
per-fold ratio of PR-AUC to prevalence for each model, and the AUC range.

Both summaries are medians over folds, because the fold is the reporting unit.
Expected: interval width 0.056 / 0.183 / 0.300 on T1 / T2 / T3, and for
gradient boosting a PR lift of 3.38 / 2.16 / 1.16. Earlier versions of this
script printed a ratio of means for gradient boosting alone (3.71 / 1.69 /
1.18); those numbers are not what the manuscript reports and should not be
compared against it.

### 5. Independent check of the label statistics (about 1 minute)

```bash
python verify.py
```

Re-implements the three headline label statistics without reference to the
analysis code, using only the standard library for parsing. Expected:
44,477 records, 35,440 pairs, 1,854 replicated, 2,230 comparisons,
discordance 0.2278, Cohen's kappa 0.2232.

### 6. Label composition and dose dependence (about 3 minutes)

```bash
pip install statsmodels        # only needed for the odds-ratio models
python dose_stats.py
```

Prints every count quoted in the manuscript's "Why provenance has to be kept"
section and in the supplement's label-composition and concentration sections:
the analysis set, how the screening-concentration fields resolve, the
model-free call-change counts, the three nested within-group odds ratios, and
the audited label set. Expected highlights: 42,493 records with a positive
screening concentration; 1,629 pair x source x assay groups tested at more than
one concentration, of which 152 (9.3%) change call and 114 of those in dose
order; odds ratios 3.89, 12.34 and 73.1 per log10 uM; 1,854 replicated pairs
splitting into 1,334 unanimously negative, 54 unanimously positive and 466
contested.

### 7. Figures

```bash
python paper_figures.py --results bench_out/results.csv \
    --leakage bench_out/leakage.csv --bench bench_out \
    --aux <dir with the released result tables> --out figs
```

Writes the four main-text figures and the four supplementary ones, and prints
every quantity it draws so the figures can be checked against the text without
opening them. `--bench` points at the harness output and supplies `records.csv`
and the fold tables, which Figure 1a needs to draw how each task cuts the data;
it defaults to `bench_out`. `--aux` is only needed for the receptor-certification
figure.

### 8. Controls on the new-source result (about 20 minutes)

```bash
python controls.py --out bench_out
```

Writes `controls_design.csv` and `controls_latent.csv`, the two tables panels c
and e of the controls figure are drawn from. Expected: broad screens mean AUC
0.671 over 11 sources against 0.615 over 6 targeted studies (Mann-Whitney
p = 0.59), and training on Dawid-Skene posteriors instead of the observed calls
moves mean AUC from 0.652 to 0.581 over 17 folds, better on 4 of them
(Wilcoxon p = 0.040).

## Expected results

| Task | RandomForest | Logistic | HistGB | ReceptorPrior |
|---|---|---|---|---|
| T1 new molecule (25 folds) | 0.664 | 0.742 | 0.739 | 0.751 |
| T2 new source (17 folds) | 0.553 | 0.630 | 0.652 | 0.654 |
| T3 replication (5 folds) | 0.517 | 0.489 | 0.591 | 0.423 |

These are the values in `results/results.csv`, which is what a fresh run should
match. The same table was reproduced on a second machine, from this package
alone, agreeing within the tolerance below (logistic regression moved in the
third decimal; the other three models did not move at all). Over
all 188 fold-model values, random forest, gradient boosting and the receptor
prior came out identical to five decimals; only logistic regression moved, by at
most 0.0042 on one fold, which shifts a task mean by at most 0.0016 (its lbfgs
solve is the one step sensitive to BLAS and thread count). No fold differed by
more than 0.01, and the model ordering was unchanged in all three tasks.

So treat a difference above about 0.005 in a task mean, a difference above 0.01
in any fold, or any change in model ordering as something to investigate rather
than accept -- and expect the three non-logistic models to match exactly.

Reading the table: on T1 and T2 the learners show no consistent gain over the
chemistry-free receptor prior; on T3 all three exceed it. The prior is a
diagnostic, not a recommended model - a result that does not clear it has not
been shown to use chemical structure.

## Varying the eligibility thresholds

The fold-eligibility thresholds are conventions, and the three flags that set
them are what the supplement's sensitivity table varies:

```bash
python run_benchmark.py --stage folds    --out bench_loose \
    --min-test 20 --min-pos 3 --min-source-records 20
python run_benchmark.py --stage evaluate --tasks T2 --out bench_loose \
    --min-test 20 --min-pos 3 --min-source-records 20
```

At that looser setting T2 admits 21 sources instead of 17.

## Replacing the splits

`folds_*.csv` are plain `record_id, fold` tables keyed to `records.csv`. To
evaluate a different splitting rule, write your own fold table with the same
columns and rerun stages 2 to 4; the audit and the per-fold reporting apply
unchanged.

## Our run, for comparison

`results/` holds everything our own run produced, so the tables above can be
checked without rerunning anything: `results.csv` (every fold, model and
interval), `report.txt`, the two control tables, and the result tables behind
the supplementary figures. `figures/` holds the figures as published.

Table 1 has been reproduced end to end on a second machine from this repository
alone. Of the 188 fold-model values, random forest, gradient boosting and the
receptor prior matched to five decimal places; only logistic regression moved,
by at most 0.0042 on one fold and 0.0016 in a task mean, and no model ordering
changed.

## Citing

Please cite both the paper and this record. `CITATION.cff` carries the metadata
in machine-readable form, and GitHub renders a "Cite this repository" button
from it.

The database itself is a separate work and should be cited too:

> Lalis, M.; Hladis, M.; Abi Khalil, S.; Briand, L.; Fiorucci, S.; Topin, J.
> M2OR: a database of olfactory receptor-odorant pairs for understanding the
> molecular mechanisms of olfaction. *Nucleic Acids Research* **2024**, 52 (D1),
> D1370-D1379. https://doi.org/10.1093/nar/gkad886

## External validation on ChEMBL

`external_validation/` rebuilds this harness, unmodified, on ChEMBL, under a
specification frozen before any ChEMBL record was collected. Two of its four
pre-specified outcomes transfer and two do not; the ranking result does not, and
the paper confines that claim to M2OR accordingly. See
`external_validation/README.md` and section 11 of the Supplementary Information.

## Licence

Apache License 2.0; see `LICENSE`.

`M2OR_20230428.csv` is third-party content, redistributed **unmodified** from
[chemosim-lab/M2OR](https://github.com/chemosim-lab/M2OR) under the same
licence. Every transformation applied to it happens at run time in the scripts
here and is visible in their source. See `NOTICE` for attribution and
`third_party/M2OR-LICENSE-Apache-2.0` for the licence text.

We redistribute the file rather than download it so that the exact bytes the
paper analysed stay available: the checkpoint in step 1 above is only meaningful
against a fixed input.
