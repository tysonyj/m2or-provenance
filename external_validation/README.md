# External validation on ChEMBL

The benchmark in the parent directory was designed on M2OR. No number computed
from M2OR can test whether the design was fitted to it, so the same harness was
rebuilt without modification on ChEMBL, a database whose provenance played no
part in designing it.

**Read `PRESPECIFICATION.md` first.** It was frozen on 2026-11-17, before any
ChEMBL activity record was collected, and it fixes the targets, the record
filters, the binarisation threshold, the task construction, the models, and four
outcomes together with the meaning of each possible answer. Its dated addendum
records the one deviation — T2 evaluated on a seed-0 random sample of 50 of the
260 eligible documents — declared before any fold was scored.

`results/O1_O4_REPORT.txt` is the scored outcome. Section 11 of the paper's
Supplementary Information discusses it.

## Result in one paragraph

Two of the four pre-specified outcomes transfer and two do not. Task-specific
uncertainty (O3) transfers and is larger on ChEMBL than on M2OR: pooled median
per-fold interval width 0.013 / 0.230 / 0.130 across the three tasks, a ratio of
17.4 against the factor of two set in advance. Cross-document label disagreement
(O4) transfers at lower magnitude: 9.2% discordance, kappa 0.534, against 22.8%
and 0.223 on M2OR. Model ranking (O1) does **not** transfer — one ordering holds
across all three ChEMBL tasks where M2OR gives three — so the ranking-instability
result is a property of sparse many-receptor data, and the paper confines that
claim to M2OR. The chemistry-free prior (O2) is beaten consistently on T1 and T2
and is not beaten on T3. The specification's conjunctive falsification condition
is not triggered; the generalisation is narrowed, not withdrawn.

One finding was not anticipated by the specification and is reported as such: in
a fold whose held-out document reports a single target, the target prior emits a
constant and scores AUC 0.500 with a zero-width interval by construction. This
affects 129 of 260 eligible T2 folds and 22 of 27 T3 folds on ChEMBL, against 1
of 17 and 0 of 5 on M2OR, because a ChEMBL document usually reports one target
while an M2OR publication is a panel screen. Those folds are excluded from the
O2 comparisons; including them would report the prior as beaten on T3 at
p <= 0.012, which would be an artefact.

## Reproducing

Collection hits the public ChEMBL REST API and takes several hours. The
collected analysis set is included so the scoring can be rerun without it.

```
pip install -r ../requirements.txt

# put the released analysis set where the scripts look for it
mkdir -p out && gunzip -c results/records.csv.gz > out/records.csv

# 1. label agreement, no model fitting (about a minute)
python3 o4_agreement.py

# 2. the three tasks. --batch and --max-batches checkpoint per batch so a run
#    can be resumed; omit --max-batches to run a task to completion.
#    T1 and T2 take hours on two cores; T3 is the quickest.
python3 external_benchmark.py --out out --tasks T3 --batch 5
python3 external_benchmark.py --out out --tasks T1 --batch 3
python3 external_benchmark.py --out out --tasks T2 --batch 4
```

Both scripts use `out/records.csv` when it is present. To rebuild the analysis
set from scratch instead, delete it and run `python3 collect.py --out out`
first; collection hits the public ChEMBL REST API, is resumable, and takes
several hours.

Verified on a clean checkout: step 1 reproduces every value in
`results/o4_agreement.csv` exactly, and step 2 reproduces the per-fold AUCs and
both bootstrap interval bounds in `results/results_T3.csv` exactly.

`external_benchmark.py` is an adapter, not a reimplementation: it builds a
records table carrying the parent harness's own column names and then calls
`run_benchmark.featurise`, `folds_T1/T2/T3`, `audit` and `evaluate` unchanged.
Any difference between the two analyses is therefore a difference in the data,
not in the scoring code.

## Files

| file | contents |
|---|---|
| `PRESPECIFICATION.md` | the frozen design and its dated addendum |
| `collect.py` | resumable ChEMBL REST collector |
| `external_benchmark.py` | adapter onto the parent harness |
| `o4_agreement.py` | cross-document agreement, clustered bootstrap |
| `results/records.csv.gz` | the analysis set, 57,799 records, 10 targets, 4,271 documents |
| `results/folds_*.csv*` | fold assignments for the three tasks |
| `results/t2_sampled_folds.csv` | the 50 documents drawn by seed 0 for T2 |
| `results/target_eligibility.csv` | record and document counts per target; all ten admitted |
| `results/leakage.csv` | four-way leakage audit, per fold |
| `results/results_T*.csv` | per-fold AUC and bootstrap interval, per model |
| `results/o4_agreement.csv` | kappa and discordance overall, by group and per target |
| `results/O1_O4_REPORT.txt` | the four outcomes scored against the specification |

ChEMBL data is used under CC BY-SA 3.0. Cite the ChEMBL release you collect.
