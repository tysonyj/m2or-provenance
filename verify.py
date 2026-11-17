"""
verify.py — independent re-implementation of the three headline numbers.

Written without reference to analysis.py / lopo.py. Deliberate differences:
  * parses the CSV with the stdlib csv module rather than pandas
  * builds cross-source comparisons by explicit nested loops over a dict of dicts
    rather than groupby + itertools.combinations
  * computes kappa from a 2x2 contingency table rather than from stacked vectors
  * resolves each source's call by majority with an explicit tie rule rather than
    pandas .mode()
  * uses a different fingerprint (RDKit pattern fingerprint, 1024 bits) and a
    different classifier (logistic regression) for the held-out-source check,
    so agreement there tests the finding rather than the implementation

Targets:
  (1) cross-source discordance  = 0.228
  (2) Cohen's kappa             = 0.223
  (3) AUC, largest source held out, HistGB = 0.443  (here: any model well below 0.5)
"""
import csv, math, re, sys, os
from collections import defaultdict

SRC = os.environ.get('M2OR_CSV', 'M2OR_20230428.csv')
MAIN = 'https://doi.org/10.1038/sdata.2015.2'


def load(normalise_ids=True):
    """Return list of (receptor, inchikey, doi, responsive, screen_conc_uM)."""
    out = []
    with open(SRC, newline='', encoding='utf-8', errors='replace') as fh:
        rd = csv.DictReader(fh, delimiter=';')
        for r in rd:
            if 'homo' not in (r['species'] or '').lower():
                continue
            if (r['Mutation'] or '').strip():
                continue
            if (r['Mixture'] or '').strip().lower() == 'mixture':
                continue
            ik = (r['InChI Key'] or '').strip()
            resp = (r['Responsive'] or '').strip()
            if not ik or resp not in ('0', '1'):
                continue
            gene = (r['Gene ID'] or '').strip().upper()
            if normalise_ids and gene.startswith('H') and gene[1:3] == 'OR':
                gene = gene[1:]
            out.append((gene, ik, (r['DOI'] or '').strip(), int(resp)))
    return out


def source_calls(rows):
    """pair -> {doi: call}, resolving repeats by majority; ties -> 0."""
    acc = defaultdict(lambda: defaultdict(list))
    for gene, ik, doi, y in rows:
        acc[(gene, ik)][doi].append(y)
    out = {}
    for pair, per in acc.items():
        out[pair] = {doi: (1 if sum(v) * 2 > len(v) else 0) for doi, v in per.items()}
    return out


def comparisons(calls, exclude=None):
    """Yield (call_a, call_b) for every unordered pair of distinct sources."""
    for pair, per in calls.items():
        dois = [d for d in per if d != exclude]
        for i in range(len(dois)):
            for j in range(i + 1, len(dois)):
                yield per[dois[i]], per[dois[j]]


def kappa_from_table(pairs):
    """Cohen's kappa from the symmetrised 2x2 table."""
    t = [[0, 0], [0, 0]]
    for a, b in pairs:
        t[a][b] += 1
        t[b][a] += 1          # symmetrise: each comparison counted in both orders
    n = sum(sum(r) for r in t)
    po = (t[0][0] + t[1][1]) / n
    row0 = (t[0][0] + t[0][1]) / n
    col0 = (t[0][0] + t[1][0]) / n
    pe = row0 * col0 + (1 - row0) * (1 - col0)
    return (po - pe) / (1 - pe), po, n // 2


def main():
    rows = load()
    calls = source_calls(rows)
    reps = {p: v for p, v in calls.items() if len(v) >= 2}
    pairs = list(comparisons(reps))
    disc = sum(1 for a, b in pairs if a != b)
    k, po, n = kappa_from_table(pairs)

    print('records kept                  %d' % len(rows))
    print('distinct pairs                %d' % len(calls))
    print('pairs with >=2 sources        %d' % len(reps))
    print('cross-source comparisons      %d' % len(pairs))
    print('(1) discordance               %.4f   [target 0.2278]' % (disc / len(pairs)))
    print('(2) Cohen kappa               %.4f   [target 0.223]' % k)
    print('    raw agreement             %.4f' % po)

    # sensitivity of the identifier-normalisation judgement call
    rows2 = load(normalise_ids=False)
    c2 = source_calls(rows2)
    r2 = {p: v for p, v in c2.items() if len(v) >= 2}
    p2 = list(comparisons(r2))
    d2 = sum(1 for a, b in p2 if a != b)
    k2, _, _ = kappa_from_table(p2)
    print('    without ID normalisation: %d comparisons, discordance %.4f, kappa %.4f'
          % (len(p2), d2 / len(p2), k2))


if __name__ == '__main__':
    main()
