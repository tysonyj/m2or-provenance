# External validation on ChEMBL — frozen specification

**Status:** frozen before any ChEMBL activity record was collected or inspected.
**Written:** 2026-11-17.
**Purpose:** test whether the three-task separation reported for M2OR reproduces
on a database whose provenance played no part in designing that benchmark.

## 1. What this does and does not test

This is **not** an independent test set for the M2OR results. Every M2OR fold
informed those numbers and nothing can change that after the fact.

What it tests is the **claim**, which is about evaluation methodology rather than
about olfactory receptors: that scoring a curated bioassay database with one
split conflates distinct questions, that model ranking does not survive the
separation, and that a baseline using no chemical information is not
consistently beaten. If that holds on a database we never touched while
designing the benchmark, the design was not fitted to M2OR. If it does not hold,
the claim is narrower than the paper states and we will say so.

An earlier attempt to do this on PubChem BioAssay failed for a diagnosable
reason: PubChem's source field names the depositor, not the laboratory that
performed the measurement, so a publication-disjoint split was not
constructible. ChEMBL carries `document_chembl_id`, `document_journal` and
`document_year` on every activity record, which is the provenance M2OR has and
PubChem lacks. That is the reason to expect this attempt to be answerable at
all; it is not a reason to expect any particular answer.

## 2. Targets

The targets are those already classified in
`endpoint_classification_criteria.md`, which was frozen earlier and before any
of the data below was seen. Taking them unchanged avoids choosing targets to
suit a result.

**STANDARDISED** (a named regulatory or consensus document fixes both the assay
system and a quantitative decision rule): KCNH2 (hERG), CYP3A4, CYP2C9, CYP2D6.

**EXPLORATORY** (no guideline fixes both): AR, ESR1, ESR2, PPARG, NR3C1, THRB.

A target enters only if, after the filters in §3, it contributes at least 200
records from at least 3 documents. Targets failing that are reported as excluded
with their counts, not silently dropped.

## 3. Record filters

Applied in this order:

1. `target_chembl_id` is one of the targets in §2, single protein, *Homo sapiens*.
2. `standard_type` in {IC50, Ki, Kd, EC50}.
3. `standard_units` is nM and `standard_value` is present and strictly positive.
4. `standard_relation` is `=`. Censored values (`>`, `<`) are excluded, because a
   binary call derived from a censored measurement is not comparable with one
   derived from a point estimate.
5. `data_validity_comment` is null (drops ChEMBL's own flags for outside-typical-
   range and potential transcription errors).
6. `document_chembl_id` is present.
7. A molecule must have a canonical SMILES in ChEMBL.

## 4. Binarisation

`pActivity = 9 - log10(standard_value in nM)`.

**Active if pActivity >= 6**, that is, potency of 1 micromolar or better. This is
the primary threshold, fixed here. Sensitivity to it is reported at `>= 5` and
`>= 7`; the primary conclusion is the one at `>= 6` whatever the others show.

A document's call on a (target, molecule) pair is the **modal call** over its
records for that pair, ties broken to inactive — the same rule the M2OR analysis
uses for cross-source comparisons.

## 5. Tasks

Constructed to mirror the M2OR benchmark exactly.

| Task | Split |
|---|---|
| T1 new molecule | Bemis-Murcko scaffolds assigned to 5 folds, 5 repeated shuffles with seeds 1-5 |
| T2 new source | leave-one-document-out |
| T3 replication | leave-one-document-out, test records restricted to pairs called active by at least one document remaining in the training set |

Pair-level consensus labels for T1 are rebuilt inside each fold from training
documents only, as in the M2OR harness.

**Eligibility**, unchanged from the paper: a test fold needs at least 40 scored
records, both classes, and at least 5 positives. A document needs at least 40
records to form a fold.

## 6. Models

The same four, library defaults, no tuning:

1. Random forest on Morgan fingerprints (radius 2, 1024 bits) plus a target indicator
2. L2-regularised logistic regression on the same features
3. Histogram gradient boosting on the same features
4. **Target prior** — uses no chemical information; predicts each target's
   positive rate in the training fold

Reporting unit is the fold, with a 1000-sample percentile bootstrap interval.
Folds are never pooled into a single figure.

## 7. Primary outcomes, and what each would mean

Stated before the data exist. Each is reported whichever way it comes out.

**O1. Does model ranking transfer?** Rank the four models by mean AUC within each
task. *Supports the paper* if the ordering differs between at least two tasks.
*Contradicts it* if one ordering holds across all three.

**O2. Is the chemistry-free prior consistently beaten?** Pre-specified paired
comparisons, prior minus each learner on identical folds, with paired Wilcoxon.
No per-fold maximum over models is computed, at all. *Supports the paper* if on
T1 or T2 no learner shows a consistent gain over the prior. *Contradicts it* if
every learner beats the prior on every task.

**O3. Is uncertainty task-specific?** Median per-fold interval width by task,
pooled over the four models. *Supports the paper* if the widths differ by a
factor of two or more between the widest and narrowest task.

**O4. Cross-document label agreement.** Cohen's kappa over pairs measured by two
or more documents, with a receptor-clustered bootstrap interval. Reported as a
descriptive quantity, and separately for STANDARDISED and EXPLORATORY targets as
the one-sided confirmatory test that `endpoint_classification_criteria.md`
pre-registered and that we could not previously complete.

## 8. What would falsify the claim

If O1 shows one stable ranking across all three tasks **and** O2 shows every
learner beating the prior on every task, then the M2OR finding does not
generalise beyond M2OR. In that case the paper's claim is restricted to M2OR in
the abstract and the discussion, this analysis is reported in full as a failed
external validation, and the generalisation is withdrawn — in the same way the
PubChem attempt, the latent-class relabelling and the standardisation hypothesis
were withdrawn.

## 9. Analyses explicitly not permitted

- Adding or dropping a target after seeing its result.
- Changing the binarisation threshold to the one that best supports a conclusion;
  `>= 6` is primary regardless.
- Comparing the prior with a per-fold maximum over the learners.
- Treating the five T1 repeats as five independent datasets; the repeat is the
  unit for T1 intervals and tests.
- Reporting kappa for a subset of targets chosen after seeing the kappa values.

---

## Addendum, 2026-11-18T02:10Z — a compute limit on T2, declared before any result

The eligibility rule of section 5 admitted 260 documents as T2 folds, far more
than the 17 M2OR admits. Evaluating all of them requires refitting three
learners on roughly 57,700 records 260 times, which on the two cores available
here is about eight hours of continuous compute, and the execution environment
does not keep a background job alive that long.

**T2 is therefore restricted to a random sample of 50 of the 260 eligible
documents, drawn with seed 0 before any fold was scored.** T1 (25 folds) and T3
(27 folds) are evaluated in full.

Three things about this deviation, stated plainly:

1. **No T2 or T3 result had been computed when this was written.** The earlier
   run was stopped after five T1 folds and, because of a bug in the adapter, had
   written no scores at all. Nothing about any outcome was known.
2. **The sample is random, not chosen.** Fifty documents drawn by seed from the
   eligible set is an unbiased estimate of what all 260 would give, with a wider
   standard error. It is not a selection of favourable folds, and the seed and
   the drawn identifiers are released with the results so the draw can be
   checked.
3. **Fifty is still three times M2OR's seventeen.** The pre-specified outcomes
   O1 to O3 are task means and paired fold-level comparisons; fifty paired folds
   is ample for both. O4 needs no model fitting and uses every eligible
   comparison, unsampled.

If the sampled T2 result is close to a decision boundary for O1 or O2, that will
be reported as the limitation it is rather than resolved by drawing more folds.
