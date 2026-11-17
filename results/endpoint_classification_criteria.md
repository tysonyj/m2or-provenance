# Endpoint classification criteria — frozen specification

**Status:** frozen before any further targets are analysed.
**Purpose:** classify a target endpoint as STANDARDISED or EXPLORATORY, so that the
hypothesis "cross-source label agreement is higher where the calling criterion is
standardised" can be tested without post-hoc adjustment.

## Provenance of this document

The first pass at this split was made informally while inspecting results, and it
is not defensible as a pre-registered test. That analysis gave a Mann–Whitney
p = 0.021 over 14 targets and is reported below as **exploratory**. The criterion
defined here is written after seeing those results but before analysing any
additional target, and the confirmatory test is the one applied to targets added
after this document was frozen.

## Operational criterion

A target endpoint is **STANDARDISED** if, at the time of analysis, a named
regulatory or formal consensus document specifies *both*:

1. **the assay system** — the measurement to be performed, in enough detail that
   two laboratories following it would run the same experiment (cell system or
   preparation, readout, control compounds, key conditions); and
2. **a quantitative decision rule** — a threshold, margin, or model by which a
   compound is declared positive.

Both conditions must be met. A guideline that merely requires that an endpoint be
*assessed*, without fixing the assay and the threshold, does not qualify.
Otherwise the endpoint is **EXPLORATORY**.

The classification is a property of the endpoint, not of any individual assay in
PubChem, and is made from the guideline documents alone — never from the observed
agreement.

## Classification of targets analysed to date

### STANDARDISED

**KCNH2 (hERG).** ICH S7B specifies an in vitro IKr/hERG study as a component of
the integrated risk assessment for QT prolongation, and the ICH E14/S7B Q&A fixes
the protocol: voltage protocols approximating a ventricular action potential,
near-physiological temperature, limits on leak current and stability, use of
reference compounds, and reporting of percent inhibition and IC50. The decision
rule is a safety margin benchmarked against reference drugs of known TdP risk.
Both conditions met.

**CYP3A4, CYP2C9, CYP2D6.** The FDA *In Vitro Drug Interaction Studies* guidance
(2020) and ICH M12 specify the inhibition study design — human liver microsomes
or recombinant enzyme, named probe substrates and positive-control inhibitors per
isoform, IC50 determination, and a preincubation design for time-dependent
inhibition — together with basic-model cutoffs (R1 based on Imax,u/Ki,u; IC50
fold-shift) for declaring an inhibitor. Both conditions met.

### EXPLORATORY

**TP53, ATAD5.** These are Tox21/ToxCast reporter endpoints. No guideline fixes
either the assay system or an activity threshold for them; hit calls are set per
campaign. The regulatory genotoxicity battery (ICH S2, OECD TG 471/487) specifies
different assays entirely and does not govern these reporters. Classified
EXPLORATORY under the criterion above. **Note:** both were classified
"regulated" in the informal first pass. Reclassifying them here is a correction
of a mis-application, not a response to their agreement values, and it is flagged
because it moves the exploratory test's p value. Any reported confirmatory result
must come from targets added after this freeze.

**AR, ESR1, ESR2, PPARG, PPARA, RARA, RXRA, VDR, NR1I2, NR3C1, PGR, THRB, AHR,
OPRD1, DRD1, HTR1A, and the olfactory receptors of M2OR.** Screening for
agonism or antagonism at these targets is performed for discovery and hazard
characterisation with laboratory-chosen assay systems and thresholds. Some appear
in EPA endocrine screening programmes, but those programmes specify a battery to
be run rather than a fixed assay and activity threshold per target, so condition
(2) is not met.

## Confirmatory test

Targets added after this freeze are classified by the criterion above, with the
guideline citation recorded before their agreement statistics are computed. The
confirmatory hypothesis is one-sided: median Cohen's kappa for cross-source
comparisons is higher among STANDARDISED endpoints than among EXPLORATORY ones.
Comparisons are restricted as in the primary analysis — matched direction
(agonist/antagonist), matched readout modality, matched orthosteric/allosteric
mode, and different source publications — and a target enters only with at least
200 such comparisons from at least two sources.

## Known limitations of the criterion

Guideline coverage correlates with commercial and regulatory attention, which
also correlates with assay maturity, compound-set overlap and laboratory
expertise. The criterion therefore cannot separate "the calling rule is written
down" from "this endpoint is simply better developed". Establishing the causal
claim would require an endpoint that acquired a standardised calling rule during
the period covered by the data, so that agreement before and after could be
compared. We do not have such a case and do not claim causality.
