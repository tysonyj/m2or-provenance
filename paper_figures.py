"""Regenerate every figure in the manuscript and the supplement.

Panels carry a letter only; everything else belongs in the caption. Each
function prints the quantities it draws so the figures can be checked against
the text without reading the image.

Run after `run_benchmark.py --stage evaluate`:
    python3 paper_figures.py [--results bench_out/results.csv]
                             [--leakage bench_out/leakage.csv]
                             [--aux DIR] [--out .]

--aux is the directory holding the released result tables
(M2OR_receptor_reliability.csv and friends); figures that need them are
skipped with a note if it is not given.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd

from dose_stats import SRC, analysis_set

GREY, LBLUE, DBLUE, RED = "#8c8c8c", "#7aa6d6", "#2f6ca8", "#c0504d"
MODELS = ["RandomForest", "Logistic", "HistGB", "ReceptorPrior"]
SHORT = ["Random forest", "Logistic", "Gradient boosting", "Receptor prior"]
COLS = [GREY, LBLUE, DBLUE, RED]
TASKS = ["T1", "T2", "T3"]
TASKNAME = ["T1  new molecule", "T2  new source", "T3  replication"]

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                     "legend.fontsize": 10.5, "axes.titlesize": 15,
                     "figure.dpi": 110})


def letter(ax, c):
    """A panel letter, and nothing else: the caption carries the rest."""
    ax.set_title(c, loc="left", fontweight="bold", fontsize=15, pad=8)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------- figure 1
def fig_design(R, leak, out, bench="bench_out"):
    fig = plt.figure(figsize=(20, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.28)

    # a: how the three tasks cut the same records. Rows are the 34 source
    # publications, columns the scaffold groups; a cell is a (source, scaffold)
    # combination that the database actually contains. Tint marks the unit the
    # task holds out as a whole -- whole scaffold columns for T1, a whole source
    # row for T2 and T3 -- and solid colour the records actually scored.
    rec = pd.read_csv(os.path.join(bench, "records.csv"))
    fds = {t: pd.read_csv(os.path.join(bench, "folds_%s.csv" % t))
           for t in ("T1", "T2", "T3")}
    # Acyclic molecules have an empty Bemis-Murcko scaffold. They are one group
    # in the split and never divide, so they get the last column.
    rec = rec.assign(scafg=rec.scaf.fillna("(acyclic)"))
    srcs = rec.groupby("DOI").size().sort_values(ascending=False).index.tolist()
    cnt = rec.groupby("scafg").size().sort_values(ascending=False)
    scfs = [c for c in cnt.index if c != "(acyclic)"] + ["(acyclic)"]
    si = {v: i for i, v in enumerate(srcs)}
    ci = {v: i for i, v in enumerate(scfs)}
    rr = rec.DOI.map(si).astype(int).values
    cc = rec.scafg.map(ci).astype(int).values
    nr, nc = len(srcs), len(scfs)
    has = np.zeros((nr, nc), bool)
    has[rr, cc] = True
    pos = {v: i for i, v in enumerate(rec.record_id)}

    def cells(task, fold):
        f = fds[task]
        ix = [pos[i] for i in f.record_id[f.fold == fold].values]
        m = np.zeros((nr, nc), bool)
        m[rr[ix], cc[ix]] = True
        return m

    f2 = fds["T2"].fold.iloc[0] if "sdata.2015.2" not in set(fds["T2"].fold) \
        else "sdata.2015.2"
    t1, t2, t3 = cells("T1", "seed1-fold0"), cells("T2", f2), cells("T3", f2)
    h1 = np.zeros((nr, nc), bool); h1[:, t1.any(0)] = True
    h2 = np.zeros((nr, nc), bool); h2[t2.any(1), :] = True
    tint = np.array(matplotlib.colors.to_rgb("#e0eaf5"))
    cell = np.array(matplotlib.colors.to_rgb("#cfcfcf"))
    mark = np.array(matplotlib.colors.to_rgb(DBLUE))

    sub = gs[0].subgridspec(3, 1, hspace=0.95)
    rows = [("T1", "new molecule", "whole chemotypes held out", t1, h1),
            ("T2", "new source", "one laboratory's whole screen held out", t2, h2),
            ("T3", "replication", "that screen, already-reported positives only",
             t3, h2)]
    for k, (tag, name, note, M, H) in enumerate(rows):
        ax = fig.add_subplot(sub[k])
        if k == 0:
            ax.text(-0.035, 1.72, "a", transform=ax.transAxes, fontsize=15,
                    fontweight="bold", va="top", ha="left")
        img = np.ones((nr, nc, 3))
        img[H] = tint
        img[has] = cell
        img[M] = mark
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.axvline(nc - 1.5, color="#888888", lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#aaaaaa"); sp.set_linewidth(0.6)
        ax.text(0.0, 1.30, tag, transform=ax.transAxes, fontsize=13.5,
                fontweight="bold", va="bottom")
        ax.text(0.055, 1.30, name, transform=ax.transAxes, fontsize=13.5,
                va="bottom")
        ax.text(0.0, 1.05, note, transform=ax.transAxes, fontsize=11,
                color="#555555", va="bottom")
        if k == 1:
            ax.set_ylabel("34 source publications", fontsize=11.5, labelpad=2)
        if k == 2:
            ax.set_xticks([nc - 1])
            ax.set_xticklabels(["acyclic"], fontsize=10.5, color="#444444")
            ax.tick_params(axis="x", length=2.5, pad=1.5, color="#888888")
            ax.set_xlabel("111 scaffold groups, by frequency",
                          fontsize=11.5, labelpad=2)

    # b: leakage audit. Colour says whether sharing is permitted for that task,
    # not how large the count is, so a reader does not read red as a fault.
    ax = fig.add_subplot(gs[1]); letter(ax, "b")
    cols = ["pairs_shared", "mols_shared", "scaffolds_shared", "sources_shared"]
    names = ["pairs", "molecules", "scaffolds", "sources"]
    M = leak.groupby("task")[cols].max()
    idx = [t for t in ["T1", "T2", "T3"] if t in M.index]
    V = M.loc[idx].values
    # permitted: T1 may share sources; T2/T3 may share pairs, molecules, scaffolds
    permitted = np.array([[0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 0]], dtype=bool)[:len(idx)]
    ax.imshow(np.where(permitted, 1, 0), cmap=matplotlib.colors.ListedColormap(
        ["#ffffff", "#dce6f1"]), vmin=0, vmax=1, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, f"{int(V[i, j]):,}", ha="center", va="center",
                    fontsize=13, fontweight="bold" if V[i, j] == 0 else "normal",
                    color="#1a1a1a")
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=11.5)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([TASKNAME[["T1", "T2", "T3"].index(t)] for t in idx], fontsize=11.5)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(idx), 1), minor=True)
    ax.grid(which="minor", color="#b7b7b7", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.legend(handles=[Patch(facecolor="#dce6f1", edgecolor="#b7b7b7",
                             label="sharing intended for this task"),
                       Patch(facecolor="#ffffff", edgecolor="#b7b7b7",
                             label="must be zero")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, handlelength=1.4)
    print("\nleakage maxima")
    for i, t in enumerate(idx):
        print("  %-3s " % t + "  ".join("%s %d" % (n, v) for n, v in zip(names, V[i])))

    # c: fold size and prevalence, with a size key
    ax = fig.add_subplot(gs[2]); letter(ax, "c")
    f = R.groupby(["task", "fold"]).agg(n=("n_test", "first"),
                                        p=("prev", "first")).reset_index()
    rng = np.random.default_rng(0)
    for i, t in enumerate(TASKS):
        s = f[f.task == t]
        ax.scatter(np.full(len(s), i) + rng.uniform(-.17, .17, len(s)), s.p,
                   s=28 + 170 * (s.n / f.n.max()) ** 0.5, alpha=0.55,
                   color=[GREY, DBLUE, RED][i], linewidth=0, zorder=3)
        print("  %s folds %d, n %d-%d, prevalence %.3f-%.3f"
              % (t, len(s), s.n.min(), s.n.max(), s.p.min(), s.p.max()))
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["T1\n%d folds" % f[f.task == "T1"].shape[0],
                        "T2\n%d folds" % f[f.task == "T2"].shape[0],
                        "T3\n%d folds" % f[f.task == "T3"].shape[0]], fontsize=11.5)
    ax.set_ylabel("positive rate in test fold")
    ax.set_xlim(-0.55, 2.55)
    for n in [100, 1000, 10000]:
        ax.scatter([], [], s=28 + 170 * (n / f.n.max()) ** 0.5, color="#bbbbbb",
                   linewidth=0, label="%s records" % f"{n:,}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, labelspacing=1.1, handletextpad=1.2)
    tidy(ax)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.93, bottom=0.2)
    fig.savefig(os.path.join(out, "b1.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 2
def fig_results(R, out):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.0),
                           gridspec_kw={"width_ratios": [1.15, 0.9, 1.3],
                                        "wspace": 0.26})
    # a: task means with every fold behind them
    W = 0.2
    print("\nmean AUC by task and model")
    for j, (m, lab, c) in enumerate(zip(MODELS, SHORT, COLS)):
        mu = [R[(R.task == t) & (R.model == m)].auc.mean() for t in TASKS]
        ax[0].bar(np.arange(3) + (j - 1.5) * W, mu, W * 0.9, color=c, label=lab, zorder=2)
        for i, t in enumerate(TASKS):
            v = R[(R.task == t) & (R.model == m)].auc.values
            ax[0].scatter(np.full(len(v), i + (j - 1.5) * W), v, s=9,
                          color="#333333", alpha=0.45, zorder=3, linewidth=0)
        print("  %-18s " % lab + "  ".join("%s %.3f" % (t, v) for t, v in zip(TASKS, mu)))
    ax[0].axhline(0.5, color="k", ls=":", linewidth=1.1, zorder=1)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(TASKNAME, fontsize=11)
    ax[0].set_ylabel("AUC"); ax[0].set_ylim(0.2, 1.0)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    letter(ax[0], "a"); tidy(ax[0])

    # b: rank within task
    print("\nrank within task (1 = highest mean AUC)")
    rk = {}
    for t in TASKS:
        mu = {m: R[(R.task == t) & (R.model == m)].auc.mean() for m in MODELS}
        order = sorted(mu, key=mu.get, reverse=True)
        for m in MODELS:
            rk.setdefault(m, []).append(order.index(m) + 1)
    for m, lab, c in zip(MODELS, SHORT, COLS):
        ax[1].plot(range(3), rk[m], "-o", color=c, markersize=11, linewidth=2.6,
                   label=lab, zorder=3, markeredgecolor="white", markeredgewidth=1.3)
        print("  %-18s %s" % (lab, rk[m]))
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(TASKS, fontsize=12)
    ax[1].set_yticks([1, 2, 3, 4]); ax[1].invert_yaxis()
    ax[1].set_ylabel("rank within task")
    ax[1].set_xlim(-0.25, 2.25); ax[1].set_ylim(4.6, 0.55)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    letter(ax[1], "b"); tidy(ax[1])

    # c: prior minus each learner, one point per fold, tasks separated
    print("\nreceptor prior minus learner, per fold")
    pos, labs, bounds = [], [], []
    x = 0
    for t in TASKS:
        w = R[R.task == t].pivot(index="fold", columns="model", values="auc")
        for m, c in zip(["Logistic", "HistGB", "RandomForest"], [LBLUE, DBLUE, GREY]):
            d = (w.ReceptorPrior - w[m]).values
            ax[2].scatter(np.full(len(d), x) + np.random.default_rng(1).uniform(
                -.1, .1, len(d)), d, s=34, color=c, alpha=0.7, linewidth=0, zorder=3)
            ax[2].hlines(d.mean(), x - .3, x + .3, color="k", linewidth=2.6, zorder=4)
            pos.append(x); labs.append({"Logistic": "Log", "HistGB": "GB",
                                        "RandomForest": "RF"}[m])
            print("  %s prior-%-13s mean %+.3f" % (t, m, d.mean()))
            x += 1
        bounds.append(x - 0.5)
        x += 0.6
    ax[2].axhline(0, color="k", ls="--", linewidth=1.2, zorder=2)
    for b in bounds[:-1]:
        ax[2].axvline(b + 0.3, color="#cccccc", linewidth=1)
    ax[2].set_xticks(pos); ax[2].set_xticklabels(labs, fontsize=11)
    ax[2].set_ylabel("AUC: receptor prior $-$ learner")
    lo, hi = ax[2].get_ylim()
    ax[2].set_ylim(lo, hi)
    for t, c in zip(TASKS, [0, 1, 2]):
        ax[2].annotate(t, xy=((pos[c * 3] + pos[c * 3 + 2]) / 2, lo),
                       xytext=(0, -34), textcoords="offset points",
                       ha="center", fontsize=12.5, fontweight="bold",
                       annotation_clip=False)
    ax[2].legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, label=l)
                          for c, l in zip([LBLUE, DBLUE, GREY],
                                          ["Logistic", "Gradient boosting",
                                           "Random forest"])],
                 loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    letter(ax[2], "c"); tidy(ax[2])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.22)
    fig.savefig(os.path.join(out, "b2.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 3
def fig_uncertainty(R, out):
    R = R.copy()
    R["w"] = R.auc_hi - R.auc_lo
    R["lift"] = R.pr / R.prev
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.3),
                           gridspec_kw={"width_ratios": [2.0, 1.0, 1.15],
                                        "wspace": 0.22})
    T2 = R[R.task == "T2"]
    order = T2.groupby("fold").n_test.first().sort_values(ascending=False).index
    y = np.arange(len(order))
    for m, lab, c, k in zip(MODELS, SHORT, COLS, ["s", "^", "o", "D"]):
        s = T2[T2.model == m].set_index("fold").loc[order]
        ax[0].scatter(s.auc, y, marker=k, s=55, color=c, label=lab, zorder=3,
                      edgecolor="white", linewidth=0.5)
        if m == "HistGB":
            ax[0].hlines(y, s.auc_lo, s.auc_hi, color=c, alpha=0.45,
                         linewidth=1.6, zorder=2)
    ax[0].axvline(0.5, color="k", ls=":", linewidth=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f[:26] for f in order], fontsize=10)
    ax[0].invert_yaxis(); ax[0].set_xlabel("AUC"); ax[0].set_xlim(0, 1)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
                 frameon=False)
    letter(ax[0], "a"); tidy(ax[0])

    print("\nmedian per-fold 95% interval width")
    for i, t in enumerate(TASKS):
        for j, (m, c) in enumerate(zip(MODELS, COLS)):
            v = R[(R.task == t) & (R.model == m)].w.values
            ax[1].scatter(np.full(len(v), i) + (j - 1.5) * 0.14, v, s=26,
                          color=c, alpha=0.75, zorder=3, linewidth=0)
        a = R[R.task == t].w
        ax[1].hlines(a.median(), i - .34, i + .34, color="k", linewidth=2.4, zorder=4)
        print("  %-18s %.3f" % (TASKNAME[i], a.median()))
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(TASKS, fontsize=12)
    ax[1].set_ylabel("width of 95% CI on AUC")
    letter(ax[1], "b"); tidy(ax[1])

    print("\nmedian per-fold PR-AUC / prevalence")
    W = 0.2
    for j, (m, lab, c) in enumerate(zip(MODELS, SHORT, COLS)):
        v = [R[(R.task == t) & (R.model == m)].lift.median() for t in TASKS]
        ax[2].bar(np.arange(3) + (j - 1.5) * W, v, W * 0.9, color=c, label=lab, zorder=3)
        print("  %-18s " % lab + "  ".join("%s %.2f" % (t, x) for t, x in zip(TASKS, v)))
    ax[2].axhline(1.0, color="k", ls="--", linewidth=1.2, zorder=4)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(TASKNAME, fontsize=11)
    ax[2].set_ylabel("PR-AUC / prevalence (median over folds)")
    ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    letter(ax[2], "c"); tidy(ax[2])
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.2)
    fig.savefig(os.path.join(out, "b3.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- figure 4
def fig_dose(out):
    d, _ = analysis_set(SRC)
    ok = d.unit.notna() & d.val.notna() & (d.val > 0)
    dd = d[ok].copy()
    dd["uM"] = np.where(dd.unit == "mM", dd.val * 1000.0, dd.val)
    dd["ctx"] = (dd.Cell_line.astype(str) + "|" + dd.Assay.astype(str)
                 + "|" + dd.Gprotein.astype(str))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"wspace": 0.26})

    edges = [0, 1, 3, 10, 30, 100, 300, np.inf]
    names = ["<1", "1-3", "3-10", "10-30", "30-100", "100-300", ">300"]
    xs, ps, lo_, hi_ = [], [], [], []
    print("\npositive rate by screening concentration (uM)")
    for i in range(len(edges) - 1):
        s = dd[(dd.uM > edges[i]) & (dd.uM <= edges[i + 1])] if i else dd[dd.uM <= edges[1]]
        if len(s) < 30:
            continue
        k = int(s.y.sum()); a, b = wilson(k, len(s))
        xs.append(names[i]); ps.append(k / len(s)); lo_.append(a); hi_.append(b)
        print("  %-9s n=%6d  %.4f [%.4f, %.4f]" % (names[i], len(s), k / len(s), a, b))
    x = np.arange(len(xs))
    ax[0].errorbar(x, ps, yerr=[np.array(ps) - lo_, np.array(hi_) - np.array(ps)],
                   fmt="o", color=DBLUE, capsize=4, markersize=8, linewidth=1.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(xs)
    ax[0].set_xlabel("screening concentration (μM)")
    ax[0].set_ylabel("positive call rate")
    letter(ax[0], "a"); tidy(ax[0])

    G = ["key", "DOI", "ctx"]
    try:
        import statsmodels.api as sm
        dd["lc"] = np.log10(dd.uM)
        specs = [("within\npair", ["key"]), ("within pair\nand source", ["key", "DOI"]),
                 ("within pair, source\nand assay context", G)]
        est = []
        print("\nodds ratio per log10 uM")
        for lab, keys in specs:
            g = dd.groupby(keys)
            sel = g.y.transform("nunique").gt(1) & g.lc.transform("nunique").gt(1)
            s = dd[sel]
            gid = s.groupby(keys).ngroup()
            X = pd.get_dummies(gid, prefix="g", drop_first=True, dtype=float)
            X.index = s.index; X.insert(0, "lc", s.lc.values)
            mm = sm.Logit(s.y.values, sm.add_constant(X)).fit(
                disp=0, method="lbfgs", maxiter=800,
                cov_type="cluster", cov_kwds={"groups": s.key.values})
            b, se = mm.params["lc"], mm.bse["lc"]
            est.append((lab, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)))
            print("  %-34s OR %6.2f [%.2f, %.2f]  n=%d"
                  % (lab.replace("\n", " "), est[-1][1], est[-1][2], est[-1][3], len(s)))
        yy = np.arange(len(est))
        ax[1].errorbar([e[1] for e in est], yy,
                       xerr=[[e[1] - e[2] for e in est], [e[3] - e[1] for e in est]],
                       fmt="D", color=RED, capsize=4, markersize=9, linewidth=1.8)
        ax[1].axvline(1.0, color="k", ls="--", linewidth=1.2)
        ax[1].set_yticks(yy); ax[1].set_yticklabels([e[0] for e in est], fontsize=11)
        ax[1].invert_yaxis(); ax[1].set_xscale("log")
        ax[1].set_xlabel("odds ratio per log$_{10}$ μM (95% CI)")
    except ImportError:
        ax[1].text(.5, .5, "statsmodels not installed", ha="center")
    letter(ax[1], "b"); tidy(ax[1])

    nc = dd.groupby(G).uM.nunique(); multi = nc[nc > 1].index
    sub = dd.set_index(G).loc[multi]
    nun = sub.groupby(level=[0, 1, 2]).y.nunique(); ch = nun[nun > 1].index
    ordered = 0
    for _, s in sub[sub.index.isin(ch)].groupby(level=[0, 1, 2]):
        p_, n_ = s.uM[s.y == 1], s.uM[s.y == 0]
        if len(p_) and len(n_) and p_.min() > n_.max():
            ordered += 1
    vals = [len(multi) - len(ch), ordered, len(ch) - ordered]
    labs = ["call unchanged", "changed,\nin dose order", "changed,\nout of order"]
    print("\ngroups tested at more than one concentration: %d" % len(multi))
    for l, v in zip(labs, vals):
        print("  %-24s %5d (%.1f%%)" % (l.replace("\n", " "), v, 100 * v / len(multi)))
    bars = ax[2].bar(range(3), vals, color=[GREY, DBLUE, RED], zorder=3, width=0.6)
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v,
                   "%d\n(%.1f%%)" % (v, 100 * v / len(multi)),
                   ha="center", va="bottom", fontsize=11)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(labs, fontsize=11)
    ax[2].set_ylabel("pair × source × assay groups")
    ax[2].set_ylim(0, len(multi) * 1.2)
    letter(ax[2], "c"); tidy(ax[2])
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.17)
    fig.savefig(os.path.join(out, "f2.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- supplement S1
def fig_s1(out):
    """Composition of the audited label set, and the selection effect."""
    d, _ = analysis_set(SRC)
    nsrc = d.groupby("key").DOI.nunique()
    rep = nsrc[nsrc >= 2].index
    by = d[d.key.isin(rep)].groupby("key").y
    neg, pos = int((by.max() == 0).sum()), int((by.min() == 1).sum())
    vals = [int((nsrc < 2).sum()), neg, len(rep) - neg - pos, pos]
    labs = ["unreplicated", "unanimously\nnegative", "contested", "unanimously\npositive"]
    cols = ["#cccccc", DBLUE, RED, LBLUE]
    all_rate, rep_rate = d.y.mean(), d.y[d.key.isin(rep)].mean()
    print("\nS1  label composition: " + ", ".join("%s %d" % (l.replace("\n", " "), v)
                                                  for l, v in zip(labs, vals)))
    print("S1  record-level positive rate: all %.4f, replicated pairs %.4f"
          % (all_rate, rep_rate))

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.0), gridspec_kw={"wspace": 0.22})
    b = ax[0].bar(range(4), vals, color=cols, zorder=3, width=0.62)
    ax[0].set_yscale("log")
    ax[0].set_ylim(30, max(vals) * 4.5)          # headroom so labels clear the frame
    for r, v in zip(b, vals):
        ax[0].text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:,}",
                   ha="center", va="bottom", fontsize=12)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labs, fontsize=11)
    ax[0].set_ylabel("receptor--molecule pairs")
    letter(ax[0], "a"); tidy(ax[0])

    b = ax[1].bar([0, 1], [100 * all_rate, 100 * rep_rate],
                  color=[GREY, DBLUE], zorder=3, width=0.45)
    for r, v in zip(b, [100 * all_rate, 100 * rep_rate]):
        ax[1].text(r.get_x() + r.get_width() / 2, v, "%.1f%%" % v,
                   ha="center", va="bottom", fontsize=12)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["all records", "records of\nreplicated pairs"], fontsize=11.5)
    ax[1].set_xlim(-0.55, 1.55)
    ax[1].set_ylim(0, 100 * rep_rate * 1.3)
    ax[1].set_ylabel("positive call rate (%)")
    letter(ax[1], "b"); tidy(ax[1])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.14)
    fig.savefig(os.path.join(out, "figS1.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- supplement S2
# Counts as reported in the inclusion-criteria sensitivity table of the
# supplement: (label, comparisons, discordant, kappa).
SENS = [("primary: human, WT, no mixtures", 2230, 508, 0.223),
        ("mono ligands only", 1479, 290, 0.325),
        ("include mixtures", 2230, 508, 0.223),
        ("include mutants", 2243, 517, 0.219),
        ("all species", 2500, 543, 0.246),
        ("require $\\geq$3 sources", 515, 140, 0.389)]


def fig_s2(out):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.2),
                           gridspec_kw={"wspace": 0.06, "width_ratios": [1.25, 1.0]})
    y = np.arange(len(SENS))[::-1]
    base = 100 * SENS[0][2] / SENS[0][1]
    print("\nS2  discordance by inclusion criterion")
    for (lab, n, k, _), yy in zip(SENS, y):
        r = 100 * k / n
        lo, hi = [100 * v for v in wilson(k, n)]
        ax[0].plot([lo, hi], [yy, yy], color=DBLUE, linewidth=1.8, zorder=3)
        ax[0].plot([lo, lo, hi, hi], [yy - .12, yy + .12, yy + .12, yy - .12],
                   " ", zorder=0)
        for e in (lo, hi):
            ax[0].plot([e, e], [yy - .13, yy + .13], color=DBLUE, linewidth=1.8, zorder=3)
        ax[0].scatter([r], [yy], s=70, color=DBLUE, zorder=4)
        print("  %-34s %.1f%% [%.1f, %.1f]" % (lab.replace("$\\geq$", ">="), r, lo, hi))
    ax[0].axvline(base, color=RED, ls="--", linewidth=1.3, zorder=2)
    ax[0].set_yticks(y); ax[0].set_yticklabels([s[0] for s in SENS], fontsize=11.5)
    ax[0].set_xlabel("discordance (%)")
    ax[0].set_ylim(-0.7, len(SENS) - 0.3)
    letter(ax[0], "a"); tidy(ax[0])

    kv = [s[3] for s in SENS]
    ax[1].barh(y, kv, height=0.52, color=GREY, zorder=3)
    for v, yy in zip(kv, y):
        ax[1].text(v + 0.008, yy, "%.3f" % v, va="center", fontsize=11.5)
    ax[1].axvline(SENS[0][3], color=RED, ls="--", linewidth=1.3, zorder=4)
    ax[1].set_yticks(y); ax[1].set_yticklabels([])
    ax[1].set_xlim(0, max(kv) * 1.22)            # headroom so no label is clipped
    ax[1].set_ylim(-0.7, len(SENS) - 0.3)
    ax[1].set_xlabel("Cohen's $\\kappa$")
    letter(ax[1], "b"); tidy(ax[1])
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.13)
    fig.savefig(os.path.join(out, "figS2.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- supplement S3
def ec50_log10uM(d):
    """log10 EC50 in uM, over all three unit conventions M2OR uses.

    Records are stored as 'uM', 'M' or 'Log(M)'; the last of these is a base-10
    logarithm of a molar concentration and is the most common of the three, so
    dropping it would discard most of the available potencies.
    """
    e = d[d["Parameter"].astype(str).str.upper() == "EC50"].copy()
    u = e["Unit"].astype(str).str.strip()
    v = pd.to_numeric(e["Value"], errors="coerce")
    l = pd.Series(np.nan, index=e.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        l[u == "uM"] = np.log10(v[u == "uM"])
        l[u == "M"] = np.log10(v[u == "M"]) + 6
    l[u == "Log(M)"] = v[u == "Log(M)"] + 6
    e["l10"] = l.replace([np.inf, -np.inf], np.nan)
    return e[e.l10.notna()]


def fig_s3(out):
    d, _ = analysis_set(SRC)
    E = ec50_log10uM(d)
    ns = E.groupby("key").DOI.nunique()
    multi = ns[ns >= 2].index
    p = E[E.key.isin(multi)].groupby(["key", "DOI"]).l10.median().reset_index()
    ds = []
    for _, g in p.groupby("key"):
        x = g.l10.values
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ds.append(abs(x[i] - x[j]))
    ds = np.sort(np.array(ds))
    print("\nS3  EC50 records %d from %d sources, %d pairs, %d pairs with >=2 sources"
          % (len(E), E.DOI.nunique(), E.key.nunique(), len(multi)))
    print("S3  %d cross-source comparisons: median |delta| %.2f, %.1f%% exceed 0.5, "
          "%.1f%% exceed 1.0" % (len(ds), np.median(ds), 100 * (ds > 0.5).mean(),
                                 100 * (ds > 1.0).mean()))

    fig, ax = plt.subplots(1, 2, figsize=(17, 6.0), gridspec_kw={"wspace": 0.22})
    lo, hi = -4.0, 4.5
    out_lo, out_hi = int((E.l10 < lo).sum()), int((E.l10 > hi).sum())
    ax[0].hist(E.l10, bins=np.arange(lo, hi + 0.25, 0.25), color=GREY, zorder=3)
    ax[0].set_xlabel("log$_{10}$ EC$_{50}$ (μM)")
    ax[0].set_ylabel("records")
    ax[0].set_xlim(lo, hi)
    if out_lo or out_hi:
        ax[0].annotate("%d records below %g and %d above %g, not shown"
                       % (out_lo, lo, out_hi, hi),
                       xy=(0.02, 0.96), xycoords="axes fraction", fontsize=10,
                       color="#666666", va="top")
    print("S3  histogram range %g to %g; %d below, %d above"
          % (lo, hi, out_lo, out_hi))
    letter(ax[0], "a"); tidy(ax[0])

    ax[1].step(np.concatenate([[0], ds]),
               np.arange(len(ds) + 1) / len(ds), where="post",
               color=DBLUE, linewidth=2.2, zorder=3)
    for v, c, yy in [(0.5, LBLUE, 0.30), (1.0, RED, 0.16)]:
        ax[1].axvline(v, color=c, ls="--", linewidth=1.3, zorder=2)
        ax[1].annotate("%.0f%% exceed %.1f" % (100 * (ds > v).mean(), v),
                       xy=(v, yy), xytext=(7, 0), textcoords="offset points",
                       color=c, ha="left", va="center", fontsize=11.5)
    ax[1].set_xlabel("|$\\Delta$ log$_{10}$ EC$_{50}$| between sources")
    ax[1].set_ylabel("cumulative fraction")
    ax[1].set_ylim(0, 1.0)
    letter(ax[1], "b"); tidy(ax[1])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14)
    fig.savefig(os.path.join(out, "figS3.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- supplement S4
def fig_s4(aux, out):
    rel = pd.read_csv(os.path.join(aux, "M2OR_receptor_reliability.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.0), gridspec_kw={"wspace": 0.22})
    n = np.arange(3, 121)
    ub = np.array([wilson(0, i)[1] for i in n])
    ax[0].plot(n, ub, color=DBLUE, linewidth=2.2, zorder=3)
    need = {}
    for thr, c in [(0.15, RED), (0.10, LBLUE)]:
        k = int(n[np.argmax(ub <= thr)]); need[thr] = k
        ax[0].axhline(thr, color=c, ls="--", linewidth=1.2, zorder=2)
        ax[0].scatter([k], [thr], s=70, color=c, zorder=4)
        ax[0].annotate("n = %d" % k, (k, thr), textcoords="offset points",
                       xytext=(10, 9), color=c, fontsize=11.5)
    ax[0].set_xlabel("cross-source comparisons with zero discordance")
    ax[0].set_ylabel("upper 95% Wilson bound")
    ax[0].set_ylim(0, 0.62)
    letter(ax[0], "a"); tidy(ax[0])

    z = rel[rel.n_disc == 0].sort_values("n_cmp", ascending=False).head(10)
    thr = need[0.15]
    y = np.arange(len(z))[::-1]
    short = np.maximum(thr - z.n_cmp.values, 0)      # never a negative shortfall
    done = z.n_cmp.values >= thr
    ax[1].barh(y, z.n_cmp, height=0.6, zorder=3,
               color=[LBLUE if d else DBLUE for d in done])
    ax[1].barh(y[~done], short[~done], left=z.n_cmp.values[~done], height=0.6,
               color=RED, zorder=3)
    ax[1].legend(handles=[Patch(facecolor=LBLUE, label="already certified"),
                          Patch(facecolor=DBLUE, label="observed"),
                          Patch(facecolor=RED, label="still needed")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
                 frameon=False)
    ax[1].axvline(thr, color="k", ls="--", linewidth=1.3, zorder=4)
    ax[1].set_yticks(y); ax[1].set_yticklabels(z.rec, fontsize=11.5)
    ax[1].set_xlabel("cross-source comparisons, all concordant")
    ax[1].set_ylim(-0.7, len(z) - 0.3)
    letter(ax[1], "b"); tidy(ax[1])
    print("\nS4  zero-discordance receptors: %d; %d comparisons certify at 0.15, "
          "%d at 0.10" % (int((rel.n_disc == 0).sum()), need[0.15], need[0.10]))
    print("S4  closest: " + ", ".join("%s %d" % (r, c) for r, c in zip(z.rec, z.n_cmp)))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.17)
    fig.savefig(os.path.join(out, "figS4.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- figure 5: controls
def fig_controls(aux, bench, out):
    """Five checks on the new-source result. a, b and d come from the released
    result tables; c and e from controls.py."""
    ts = pd.read_csv(os.path.join(aux, "M2OR_trainsize_control.csv"), index_col=0)
    lo = pd.read_csv(os.path.join(aux, "M2OR_lopo_results.csv"))
    sp = pd.read_csv(os.path.join(aux, "M2OR_sourcepair_transfer.csv"))
    dg = pd.read_csv(os.path.join(bench, "controls_design.csv"))
    lt = pd.read_csv(os.path.join(bench, "controls_latent.csv"))
    from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

    fig, ax = plt.subplots(1, 5, figsize=(24, 5.4), gridspec_kw={"wspace": 0.34})

    # a: full against size-matched training sets
    m = ts.dropna(subset=["matched"])
    ax[0].plot([0.3, 1.0], [0.3, 1.0], "--", color="k", linewidth=1.1, zorder=2)
    ax[0].scatter(m.full_train, m.matched, s=62, color=GREY, zorder=3,
                  edgecolor="white", linewidth=0.6)
    d = (m.matched - m.full_train).dropna()
    p = wilcoxon(d).pvalue if len(d) > 1 else float("nan")
    ax[0].annotate("$p$ = %.2f" % p, xy=(0.04, 0.93), xycoords="axes fraction",
                   fontsize=11.5, va="top")
    ax[0].set_xlabel("AUC, full training set")
    ax[0].set_ylabel("AUC, size-matched")
    ax[0].set_xlim(0.3, 1.0); ax[0].set_ylim(0.3, 1.0)
    print("\nS5a training size: n=%d, mean change %+.3f, Wilcoxon p=%.2f"
          % (len(d), d.mean(), p))
    letter(ax[0], "a"); tidy(ax[0])

    # b: is the test pair already in training?
    cols = [("auc_novel", "unseen"), ("auc_seen", "seen"), ("auc_conflict", "contested")]
    names = {"RF": "Random forest", "Logistic": "Logistic", "HistGB": "Gradient boosting"}
    W = 0.26
    print("S5b record overlap (mean AUC)")
    for j, (c, lab) in enumerate(cols):
        v = [lo[lo.model == k][c].mean() for k in names]
        ax[1].bar(np.arange(len(names)) + (j - 1) * W, v, W * 0.9,
                  color=[DBLUE, LBLUE, RED][j], label=lab, zorder=3)
        print("  %-10s " % lab + "  ".join("%s %.3f" % (k, x) for k, x in zip(names, v)))
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(["RF", "Log", "GB"], fontsize=11.5)
    ax[1].set_ylabel("AUC, held-out source")
    ax[1].set_ylim(0.4, None)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
                 frameon=False, columnspacing=1.0, handlelength=1.2)
    letter(ax[1], "b"); tidy(ax[1])

    # c: what kind of study is the held-out source?
    groups = ["broad screen", "targeted study"]
    data = [dg.auc[dg.group == g].values for g in groups]
    bp = ax[2].boxplot(data, widths=0.45, patch_artist=True,
                       medianprops=dict(color="k", linewidth=1.8))
    for b, c in zip(bp["boxes"], [DBLUE, GREY]):
        b.set_facecolor(c); b.set_alpha(0.55); b.set_edgecolor("#555555")
    for i, v in enumerate(data):
        ax[2].scatter(np.full(len(v), i + 1)
                      + np.random.default_rng(3).uniform(-.09, .09, len(v)),
                      v, s=30, color="#333333", alpha=0.6, zorder=4, linewidth=0)
    pm = mannwhitneyu(data[0], data[1]).pvalue
    ax[2].axhline(0.5, color="k", ls=":", linewidth=1.1)
    ax[2].annotate("$p$ = %.2f" % pm, xy=(0.04, 0.93), xycoords="axes fraction",
                   fontsize=11.5, va="top")
    ax[2].set_xticks([1, 2])
    ax[2].set_xticklabels(["broad\nscreens\n(n=%d)" % len(data[0]),
                           "targeted\nstudies\n(n=%d)" % len(data[1])], fontsize=11)
    ax[2].set_ylabel("AUC, held-out source")
    print("S5c study design: broad %.3f (n=%d), targeted %.3f (n=%d), p=%.2f"
          % (data[0].mean(), len(data[0]), data[1].mean(), len(data[1]), pm))
    letter(ax[2], "c"); tidy(ax[2])

    # d: does label discordance predict transfer loss?
    rho, pr = spearmanr(sp.disc, sp.loss)
    ax[3].scatter(sp.disc, sp.loss, s=52, color=GREY, zorder=3,
                  edgecolor="white", linewidth=0.6)
    ax[3].axhline(0, color="k", ls=":", linewidth=1.1)
    ax[3].annotate("Spearman $\\rho$ = %+.2f\n$p$ = %.2f  ($n$ = %d)"
                   % (rho, pr, len(sp)), xy=(0.04, 0.95),
                   xycoords="axes fraction", fontsize=11.5, va="top")
    ax[3].set_xlabel("label discordance on shared pairs")
    ax[3].set_ylabel("transfer loss (AUC)")
    print("S5d discordance vs loss: rho %+.2f, p %.2f, n %d" % (rho, pr, len(sp)))
    letter(ax[3], "d"); tidy(ax[3])

    # e: training on latent-truth posteriors
    d2 = lt.auc_latent - lt.auc_observed
    pw = wilcoxon(d2).pvalue
    lim = [min(lt.auc_observed.min(), lt.auc_latent.min()) - 0.03,
           max(lt.auc_observed.max(), lt.auc_latent.max()) + 0.03]
    ax[4].plot(lim, lim, "--", color="k", linewidth=1.1, zorder=2)
    ax[4].scatter(lt.auc_observed, lt.auc_latent, s=62, color=RED, zorder=3,
                  edgecolor="white", linewidth=0.6)
    ax[4].annotate("mean %+.3f\n$p$ = %.3f  ($n$ = %d)" % (d2.mean(), pw, len(d2)),
                   xy=(0.04, 0.95), xycoords="axes fraction", fontsize=11.5, va="top")
    ax[4].set_xlabel("AUC, observed labels")
    ax[4].set_ylabel("AUC, latent-truth labels")
    ax[4].set_xlim(lim); ax[4].set_ylim(lim)
    print("S5e latent relabelling: observed %.3f, latent %.3f, mean %+.3f, "
          "better in %d/%d, Wilcoxon p=%.3f"
          % (lt.auc_observed.mean(), lt.auc_latent.mean(), d2.mean(),
             int((d2 > 0).sum()), len(d2), pw))
    letter(ax[4], "e"); tidy(ax[4])

    fig.subplots_adjust(left=0.04, right=0.995, top=0.91, bottom=0.21)
    fig.savefig(os.path.join(out, "f5.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench_out/results.csv")
    ap.add_argument("--leakage", default="bench_out/leakage.csv")
    ap.add_argument("--bench", default="bench_out",
                    help="harness output: records.csv and folds_T*.csv")
    ap.add_argument("--aux", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = pd.read_csv(a.results)
    fig_design(R, pd.read_csv(a.leakage), a.out, a.bench)
    fig_results(R, a.out)
    fig_uncertainty(R, a.out)
    fig_dose(a.out)
    fig_s1(a.out)
    fig_s2(a.out)
    fig_s3(a.out)
    made = "b1.png, b2.png, b3.png, f2.png, figS1.png, figS2.png, figS3.png"
    if a.aux:
        fig_s4(a.aux, a.out)
        fig_controls(a.aux, os.path.dirname(a.results) or ".", a.out)
        made += ", figS4.png, f5.png"
    else:
        print("\n[--aux not given: figS4 needs M2OR_receptor_reliability.csv, skipped]")
    print("\nwrote %s to %s" % (made, a.out))
