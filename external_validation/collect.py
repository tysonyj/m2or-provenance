"""Resumable ChEMBL activity collector. Caches each page; safe to re-run."""
import json, os, sys, time, urllib.request, urllib.parse
B = "https://www.ebi.ac.uk/chembl/api/data"
CACHE = "pages"; os.makedirs(CACHE, exist_ok=True)
KEEP = ("target_chembl_id molecule_chembl_id document_chembl_id assay_chembl_id "
        "standard_type standard_value standard_units standard_relation "
        "data_validity_comment canonical_smiles document_year pchembl_value").split()

def fetch(url, tries=14):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={'User-Agent': 'm2or-provenance-external/1.0'}),
                    timeout=120) as r:
                return json.load(r)
        except Exception as e:
            if t == tries - 1:
                print("      give up: %s" % type(e).__name__, flush=True)
                return None
            time.sleep(min(2 ** t, 120))

def collect(gene, tid):
    out = os.path.join(CACHE, "%s.jsonl" % gene)
    done = os.path.join(CACHE, "%s.done" % gene)
    if os.path.exists(done):
        n = sum(1 for _ in open(out))
        print("  %-8s cached, %d records" % (gene, n), flush=True)
        return
    url = (B + "/activity.json?" + urllib.parse.urlencode({
        "target_chembl_id": tid,
        "standard_type__in": "IC50,Ki,Kd,EC50",
        "limit": 1000}))
    seen = 0
    f = open(out, "w")
    page = 0
    while url:
        d = fetch(url)
        if d is None:
            f.close()
            print("  %-8s INCOMPLETE at page %d (%d records)" % (gene, page, seen), flush=True)
            return
        acts = d.get("activities", [])
        for a in acts:
            f.write(json.dumps({k: a.get(k) for k in KEEP}) + "\n")
        seen += len(acts); page += 1
        total = d["page_meta"]["total_count"]
        nxt = d["page_meta"].get("next")
        if page % 5 == 0 or not nxt:
            print("  %-8s page %3d  %6d / %6d" % (gene, page, seen, total), flush=True)
        url = "https://www.ebi.ac.uk" + nxt if nxt else None
        time.sleep(0.2)
    f.close(); open(done, "w").write(str(seen))
    print("  %-8s DONE %d records" % (gene, seen), flush=True)

if __name__ == "__main__":
    T = json.load(open("targets.json"))
    for gene, tid in sorted(T.items()):
        collect(gene, tid)
    print("\ncollection pass complete", flush=True)
