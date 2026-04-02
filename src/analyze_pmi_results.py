"""
Analysis and Visualization for PMI Bigram Prediction Experiments.

Generates 11 figures covering all four experiments:

    Fig 1:  PMI vs Priming Ratio scatter (Task 1)
    Fig 2:  Top-1/5/10 accuracy by PMI bucket (Task 1)
    Fig 3:  Logit lens commitment layer by PMI tier (Task 1, novel)
    Fig 4:  POS pattern accuracy breakdown (Task 1)
    Fig 5:  Majority vs Minority probability paired comparison (Task 2)
    Fig 6:  Over-Memorization Index distribution (Task 2, novel)
    Fig 7:  Logit lens competition curves — majority vs minority (Task 2, novel)
    Fig 8:  PMI-gap threshold accuracy curve (Task 2)
    Fig 9:  General vs Biomedical probability comparison (Exp 3)
    Fig 10: Three-way PMI alignment scatter (Exp 3, novel — if PMI_bio available)
    Fig 11: Four-quadrant Task 3 scatter with Hidden Knowledge highlighted (Exp 4, novel)

Usage:
    python src/analyze_pmi_results.py

Prerequisites:
    Run pmi_prediction_exp.py first.

Colab (prior cell if needed):
    !pip install -q matplotlib seaborn scipy numpy
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Must match general-domain PMI source in pmi_dataset_builder.py
GENERAL_CORPUS_SHORT = "OpenWebText"

try:
    ROOT = Path(__file__).parent.parent        # local: .py script
except NameError:
    ROOT = Path("/content/llm-washing-machine-claude")   # Colab: notebook cell
RESULTS_DIR  = ROOT / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})

# ── Load results ─────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "pmi_prediction_results.json")  as f: exp1_data = json.load(f)
with open(RESULTS_DIR / "minority_class_results.json")  as f: exp2_data = json.load(f)
with open(RESULTS_DIR / "cross_domain_results.json")    as f: exp3_data = json.load(f)
with open(RESULTS_DIR / "hidden_knowledge_results.json") as f: exp4_data = json.load(f)

exp1 = exp1_data["results"]
exp2 = exp2_data["results"]
exp3 = exp3_data["results"]
exp4 = exp4_data["results"]

s1, s2, s3, s4 = exp1_data["summary"], exp2_data["summary"], exp3_data["summary"], exp4_data["summary"]

# ── Print full statistical summary ───────────────────────────────────────────
print("=" * 65)
print("Full Statistical Summary")
print("=" * 65)

print(f"\n[Exp 1 — Task 1]  n={s1['n']}, {s1['n_templates']}-template avg")
print(f"  Top-1:  {s1['top1_accuracy']:.1%}   Top-5:  {s1['top5_accuracy']:.1%}   Top-10: {s1['top10_accuracy']:.1%}")
print(f"  Median priming ratio: {s1['median_priming_ratio']:.1f}x")
print(f"  Spearman(PMI, priming): r={s1['spearman_pmi_priming_r']:.3f}, p={s1['spearman_pmi_priming_p']:.4f}")
print(f"  Median commitment layer: {s1['median_commitment_layer']}  "
      f"({s1['pct_committed']:.1%} committed within 12 layers)")
print("  POS breakdown (top-1 acc):")
for pos, d in sorted(s1["pos_breakdown"].items(), key=lambda x: -x[1]["n"])[:8]:
    print(f"    {pos:12s}: n={d['n']:3d}, acc={d['top1_acc']:.1%}")

print(f"\n[Exp 2 — Task 2]  n={s2['n_groups']} groups, {s2['n_templates']}-template avg")
print(f"  Median majority advantage: {s2['median_majority_advantage']:.1f}x")
if s2["wilcoxon_p"] is not None:
    print(f"  Wilcoxon: stat={s2['wilcoxon_stat']:.2f}, p={s2['wilcoxon_p']:.4f}")
if s2["omi_median"] is not None:
    print(f"  OMI median={s2['omi_median']:.2f},  % > 1 = {s2['omi_pct_above_1']:.1%}")
if s2["spearman_pmi_gap_correct_r"] is not None:
    print(f"  Spearman(PMI_gap, correct): r={s2['spearman_pmi_gap_correct_r']:.3f}, "
          f"p={s2['spearman_pmi_gap_correct_p']:.4f}")

print(f"\n[Exp 3 — Biomedical]  n={s3['n_groups']} groups")
print(f"  General favoured: {s3['n_general_favoured']}/{s3['n_groups']}")
if s3["general_to_bio_ratio"]:
    print(f"  General/Bio ratio: {s3['general_to_bio_ratio']:.1f}x")
if s3["mann_whitney_p"] is not None:
    print(f"  Mann-Whitney: stat={s3['mann_whitney_stat']:.2f}, p={s3['mann_whitney_p']:.4f}")
if s3["domain_alignment_general_r"] is not None:
    print(f"  align_general: r={s3['domain_alignment_general_r']:.3f}, "
          f"p={s3['domain_alignment_general_p']:.4f}")
if s3["domain_alignment_bio_r"] is not None:
    print(f"  align_bio:     r={s3['domain_alignment_bio_r']:.3f}, "
          f"p={s3['domain_alignment_bio_p']:.4f}")

print(f"\n[Exp 4 — Task 3]  n={s4['n_total']} bigrams")
for quad, cnt in sorted(s4["quadrant_counts"].items(), key=lambda x: -x[1]):
    print(f"  {quad:25s}: {cnt} ({cnt/s4['n_total']:.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — PMI vs Priming Ratio
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

pmi_arr  = np.array([r["pmi_general"]       for r in exp1])
prim_arr = np.array([r["avg_priming_ratio"]  for r in exp1])
cols = ["#27ae60" if r["top_1_correct"] else "#e74c3c" for r in exp1]

ax.scatter(pmi_arr, prim_arr, c=cols, alpha=0.55, s=45, edgecolors="none", zorder=3)

slope, intercept, r_val, p_val, _ = stats.linregress(pmi_arr, prim_arr)
x_line = np.linspace(pmi_arr.min(), pmi_arr.max(), 200)
ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5, lw=1.5,
        label=f"Linear fit  r={r_val:.2f}, p={p_val:.3f}")

for r in sorted(exp1, key=lambda x: -x["avg_priming_ratio"])[:5]:
    ax.annotate(f"{r['word1']} {r['word2']}", (r["pmi_general"], r["avg_priming_ratio"]),
                fontsize=7.5, alpha=0.85, xytext=(6, 4), textcoords="offset points")

ax.legend(handles=[
    ax.get_lines()[0],
    mpatches.Patch(color="#27ae60", alpha=0.8, label="Top-1 correct"),
    mpatches.Patch(color="#e74c3c", alpha=0.8, label="Top-1 incorrect"),
], fontsize=9, loc="upper left")

ax.set_xlabel(f"Corpus PMI ({GENERAL_CORPUS_SHORT})", fontsize=12)
ax.set_ylabel("Priming Ratio (log scale)", fontsize=12)
ax.set_title(f"PMI vs Priming Ratio  (n={len(exp1)}, "
             f"Spearman r={s1['spearman_pmi_priming_r']:.2f})", fontsize=12)
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pmi_vs_priming.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 1 — pmi_vs_priming.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Top-1/5/10 accuracy by PMI bucket
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

edges  = [float("-inf"), 4, 6, 8, 10, float("inf")]
labels = ["< 4", "4–6", "6–8", "8–10", "> 10"]
t1r, t5r, t10r, ns = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    bkt = [r for r in exp1 if lo <= r["pmi_general"] < hi]
    ns.append(len(bkt))
    t1r.append(sum(r["top_1_correct"]  for r in bkt) / max(len(bkt), 1))
    t5r.append(sum(r["top_5_correct"]  for r in bkt) / max(len(bkt), 1))
    t10r.append(sum(r["top_10_correct"] for r in bkt) / max(len(bkt), 1))

x, w = np.arange(len(labels)), 0.25
ax.bar(x - w, t1r,  w, label="Top-1",  color="#3498db", alpha=0.85)
ax.bar(x,     t5r,  w, label="Top-5",  color="#e67e22", alpha=0.85)
ax.bar(x + w, t10r, w, label="Top-10", color="#2ecc71", alpha=0.85)
for i, (n, t1) in enumerate(zip(ns, t1r)):
    ax.text(x[i] - w, (t1 or 0.01) + 0.01, f"n={n}", ha="center", fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.18)
ax.set_xlabel("PMI Bucket", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Prediction Accuracy by PMI Bucket", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "accuracy_by_pmi_bucket.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 2 — accuracy_by_pmi_bucket.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Logit lens: commitment layer by PMI tier
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: commitment layer distribution by PMI tier
tiers = {"High PMI (≥ 8)": [], "Med PMI (6–8)": [], "Low PMI (< 6)": []}
for r in exp1:
    cl = r.get("commitment_layer")
    if cl is None:
        continue
    p = r["pmi_general"]
    if p >= 8:
        tiers["High PMI (≥ 8)"].append(cl)
    elif p >= 6:
        tiers["Med PMI (6–8)"].append(cl)
    else:
        tiers["Low PMI (< 6)"].append(cl)

ax = axes[0]
colors_t = ["#3498db", "#e67e22", "#e74c3c"]
for (label, vals), col in zip(tiers.items(), colors_t):
    if vals:
        ax.hist(vals, bins=range(13), alpha=0.6, color=col, label=f"{label} (n={len(vals)})", density=True)
ax.set_xlabel("Commitment Layer", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Logit Lens: Commitment Layer Distribution\nby PMI Tier", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Right: mean layer-wise P(word2) averaged over high vs low PMI bigrams
high_pmi_items = [r for r in exp1 if r["pmi_general"] >= 8 and r["logit_lens_probs"]]
low_pmi_items  = [r for r in exp1 if r["pmi_general"] < 6  and r["logit_lens_probs"]]

ax2 = axes[1]
if high_pmi_items and low_pmi_items:
    n_layers = len(high_pmi_items[0]["logit_lens_probs"])
    layers   = list(range(n_layers))
    high_mean = np.array([[p for p in r["logit_lens_probs"]] for r in high_pmi_items]).mean(axis=0)
    low_mean  = np.array([[p for p in r["logit_lens_probs"]] for r in low_pmi_items]).mean(axis=0)
    ax2.plot(layers, high_mean, "b-o", markersize=4, label=f"High PMI (n={len(high_pmi_items)})")
    ax2.plot(layers, low_mean,  "r-s", markersize=4, label=f"Low PMI  (n={len(low_pmi_items)})")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Mean P(word2)  via logit lens", fontsize=12)
    ax2.set_title("Logit Lens: Mean P(word2) Across Layers\nHigh vs Low PMI Bigrams", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
else:
    ax2.text(0.5, 0.5, "Insufficient data for layer comparison",
             transform=ax2.transAxes, ha="center", va="center")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "logit_lens_commitment.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 3 — logit_lens_commitment.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — POS pattern accuracy breakdown
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

pos_bd = s1["pos_breakdown"]
pos_sorted = sorted(pos_bd.items(), key=lambda x: -x[1]["n"])[:12]
pos_labels  = [p for p, _ in pos_sorted]
pos_accs    = [d["top1_acc"] for _, d in pos_sorted]
pos_ns      = [d["n"]       for _, d in pos_sorted]

bars = ax.barh(range(len(pos_labels)), pos_accs,
               color=["#3498db" if a >= 0.5 else "#e74c3c" for a in pos_accs], alpha=0.8)
ax.set_yticks(range(len(pos_labels)))
ax.set_yticklabels([f"{l} (n={n})" for l, n in zip(pos_labels, pos_ns)], fontsize=9)
ax.set_xlabel("Top-1 Accuracy", fontsize=12)
ax.set_title("Top-1 Prediction Accuracy by POS Pattern\n(NN-NN, JJ-NN, VBG-NN, etc.)", fontsize=12)
ax.axvline(0.5, color="k", linestyle="--", alpha=0.4, lw=1.2, label="50% baseline")
ax.set_xlim(0, 1.05); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pos_pattern_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 4 — pos_pattern_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Majority vs Minority probability paired comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

prefixes_l, maj_ps, min_ps = [], [], []
for prefix, result in exp2.items():
    if result["majority"] and result["minority"]:
        prefixes_l.append(prefix)
        maj_ps.append(result["majority"]["avg_p_model"])
        min_ps.append(float(np.mean([r["avg_p_model"] for r in result["minority"]])))

ax = axes[0]
ax.scatter(maj_ps, [1.0] * len(maj_ps), c="#3498db", s=70, alpha=0.75, zorder=3, label="Majority")
ax.scatter(min_ps, [0.0] * len(min_ps), c="#e74c3c", s=70, alpha=0.75, zorder=3, label="Minority (mean)")
for m, n in zip(maj_ps, min_ps):
    ax.plot([m, n], [1.0, 0.0], color="gray", alpha=0.25, lw=0.9)
for prefix, m in zip(prefixes_l, maj_ps):
    ax.annotate(prefix, (m, 1.0), fontsize=6.5, alpha=0.7, xytext=(4, 3), textcoords="offset points")
ax.set_yticks([0.0, 1.0]); ax.set_yticklabels(["Minority", "Majority"], fontsize=11)
ax.set_xscale("log"); ax.set_xlabel("Model Probability", fontsize=11)
ax.set_title("Paired Majority vs Minority Probabilities", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")

ax2 = axes[1]
bp = ax2.boxplot([maj_ps, min_ps], labels=["Majority", "Minority"], patch_artist=True,
                  medianprops=dict(color="#e74c3c", lw=2.5))
bp["boxes"][0].set_facecolor("#aed6f1"); bp["boxes"][1].set_facecolor("#f1948a")
if s2["wilcoxon_p"] is not None:
    sig = "***" if s2["wilcoxon_p"] < 0.001 else ("**" if s2["wilcoxon_p"] < 0.01 else "*")
    ax2.text(0.5, 0.93, f"Wilcoxon p={s2['wilcoxon_p']:.4f} {sig}",
             transform=ax2.transAxes, ha="center", fontsize=10, color="#922b21")
ax2.text(0.5, 0.86, f"Median advantage: {s2['median_majority_advantage']:.0f}x",
         transform=ax2.transAxes, ha="center", fontsize=10, color="#1a5276")
ax2.set_yscale("log"); ax2.set_ylabel("Model Probability (log)", fontsize=11)
ax2.set_title("Distribution: Majority vs Minority", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "majority_vs_minority.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 5 — majority_vs_minority.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Over-Memorization Index distribution
# ─────────────────────────────────────────────────────────────────────────────
all_omi_vals = [
    r["over_memorization_index"]
    for result in exp2.values()
    for r in result["minority"]
    if r.get("over_memorization_index") is not None
]

if all_omi_vals:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_omi_vals, bins=30, color="#8e44ad", alpha=0.75, edgecolor="white")
    ax.axvline(1.0, color="k", linestyle="--", lw=2, label="OMI = 1 (no extra bias)")
    ax.axvline(float(np.median(all_omi_vals)), color="#e74c3c", linestyle="-", lw=2,
               label=f"Median = {np.median(all_omi_vals):.2f}")
    pct_gt1 = np.mean([o > 1 for o in all_omi_vals])
    ax.text(0.98, 0.92, f"{pct_gt1:.1%} of minority pairs\nhave OMI > 1",
            transform=ax.transAxes, ha="right", fontsize=10, color="#6c3483",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5eef8"))
    ax.set_xlabel("Over-Memorization Index (OMI)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Over-Memorization Index Distribution\n"
                 "OMI > 1: model majority bias exceeds corpus frequency prediction", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "over_memorization_index.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: Fig 6 — over_memorization_index.png")
else:
    print("Skip: Fig 6 — no OMI values (missing corpus freq data)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Logit lens competition curves (majority vs minority)
# ─────────────────────────────────────────────────────────────────────────────
groups_with_lens = [(prefix, res) for prefix, res in exp2.items()
                    if res.get("logit_lens_competition")]

if groups_with_lens:
    n_show = min(6, len(groups_with_lens))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    layers = list(range(12))

    for idx, (prefix, res) in enumerate(groups_with_lens[:n_show]):
        lc  = res["logit_lens_competition"]
        ax  = axes[idx]
        ax.plot(layers, lc["majority_lens"],    "b-o", markersize=3, lw=1.5,
                label=f"majority ({res['majority']['word']})")
        ax.plot(layers, lc["minority_lens_mean"], "r-s", markersize=3, lw=1.5,
                label="minority (mean)")
        cross = lc.get("majority_dominates_from_layer")
        if cross is not None:
            ax.axvline(cross, color="gray", linestyle=":", lw=1.5, alpha=0.8,
                       label=f"dominates @ layer {cross}")
        ax.set_title(f"'{prefix}'", fontsize=10)
        ax.set_xlabel("Layer", fontsize=9); ax.set_ylabel("P (logit lens)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    for idx in range(n_show, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Logit Lens: Majority vs Minority Competition Across Layers",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "logit_lens_competition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: Fig 7 — logit_lens_competition.png")
else:
    print("Skip: Fig 7 — no logit lens competition data")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — PMI-gap threshold accuracy curve
# ─────────────────────────────────────────────────────────────────────────────
gap_correct_pairs = [
    (r["pmi_gap_vs_majority"],
     int(res["majority"]["avg_p_model"] > r["avg_p_model"]))
    for res in exp2.values()
    for r in res["minority"]
    if r.get("pmi_gap_vs_majority") is not None
]

if len(gap_correct_pairs) >= 10:
    gap_correct_pairs.sort(key=lambda x: x[0])
    gaps    = np.array([g for g, _ in gap_correct_pairs])
    correct = np.array([c for _, c in gap_correct_pairs])

    # Rolling accuracy with window
    window = max(5, len(gap_correct_pairs) // 10)
    roll_gaps, roll_accs = [], []
    for i in range(len(gaps) - window + 1):
        roll_gaps.append(float(np.mean(gaps[i:i+window])))
        roll_accs.append(float(np.mean(correct[i:i+window])))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(gaps, correct + np.random.uniform(-0.02, 0.02, len(correct)),
               alpha=0.3, s=25, c="#3498db", edgecolors="none")
    ax.plot(roll_gaps, roll_accs, "k-", lw=2.5, label=f"Rolling mean (window={window})")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Chance (50%)")
    ax.axhline(0.75, color="#27ae60", linestyle=":", alpha=0.6, label="75% threshold")
    ax.set_xlabel("PMI Gap (majority − minority)", fontsize=12)
    ax.set_ylabel("Model Correct (majority > minority)", fontsize=12)
    ax.set_title("PMI Gap vs. Model Accuracy for Majority/Minority Distinction", fontsize=12)
    ax.set_ylim(-0.1, 1.15); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pmi_gap_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: Fig 8 — pmi_gap_threshold.png")
else:
    print("Skip: Fig 8 — insufficient PMI-gap data")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — General vs Biomedical probability comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cd_prefixes = [r["prefix"]              for r in exp3]
gen_max_l   = [r["summary"]["gen_max_p"] for r in exp3]
bio_max_l   = [r["summary"]["bio_max_p"] for r in exp3]

ax = axes[0]
ax.scatter(gen_max_l, [1.0]*len(gen_max_l), c="#3498db", s=70, alpha=0.75, zorder=3, label="General")
ax.scatter(bio_max_l, [0.0]*len(bio_max_l), c="#8e44ad", s=70, alpha=0.75, zorder=3, label="Biomedical")
for g, b in zip(gen_max_l, bio_max_l):
    ax.plot([g, b], [1.0, 0.0], color="gray", alpha=0.25, lw=0.9)
for prefix, g, b in zip(cd_prefixes, gen_max_l, bio_max_l):
    ax.annotate(prefix, (max(g, b), 1.0 if g > b else 0.0),
                fontsize=7, alpha=0.7, xytext=(4, 3), textcoords="offset points")
ax.set_yticks([0.0, 1.0]); ax.set_yticklabels(["Biomedical", "General"], fontsize=11)
ax.set_xscale("log"); ax.set_xlabel("Max Completion Probability", fontsize=11)
ax.set_title(f"General vs Biomedical Completions\n"
             f"(General favoured in {s3['n_general_favoured']}/{s3['n_groups']} groups)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="x")

ax2 = axes[1]
bp = ax2.boxplot([gen_max_l, bio_max_l], labels=["General", "Biomedical"], patch_artist=True,
                  medianprops=dict(color="#e74c3c", lw=2.5))
bp["boxes"][0].set_facecolor("#aed6f1"); bp["boxes"][1].set_facecolor("#d7bde2")
if s3["mann_whitney_p"] is not None:
    sig = "***" if s3["mann_whitney_p"] < 0.001 else ("**" if s3["mann_whitney_p"] < 0.01 else "*")
    ax2.text(0.5, 0.93, f"Mann-Whitney p={s3['mann_whitney_p']:.4f} {sig}",
             transform=ax2.transAxes, ha="center", fontsize=10, color="#922b21")
if s3["general_to_bio_ratio"]:
    ax2.text(0.5, 0.86, f"Gen/Bio ratio: {s3['general_to_bio_ratio']:.1f}x",
             transform=ax2.transAxes, ha="center", fontsize=10, color="#1a5276")
ax2.set_yscale("log"); ax2.set_ylabel("Max Completion Probability (log)", fontsize=11)
ax2.set_title("Distribution: General vs Biomedical", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cross_domain_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 9 — cross_domain_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Three-way PMI alignment (if PMI_bio available)
# ─────────────────────────────────────────────────────────────────────────────
all_words_bio = [w for r in exp3 for w in r["general"] + r["biomedical"]]
has_pmi_gen = [w for w in all_words_bio if w.get("pmi_general") is not None]
has_pmi_bio = [w for w in all_words_bio if w.get("pmi_bio")     is not None]

if len(has_pmi_gen) >= 5 or len(has_pmi_bio) >= 5:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(has_pmi_gen) >= 5:
        ax = axes[0]
        x = np.array([w["pmi_general"]  for w in has_pmi_gen])
        y = np.array([w["avg_p_model"]  for w in has_pmi_gen])
        cols_d = ["#3498db" if w["domain"] == "general" else "#8e44ad" for w in has_pmi_gen]
        ax.scatter(x, y, c=cols_d, alpha=0.6, s=50)
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, slope * xl + intercept, "k--", alpha=0.5,
                label=f"r={r_val:.2f}, p={p_val:.3f}")
        ax.set_xlabel(f"PMI_general ({GENERAL_CORPUS_SHORT})", fontsize=11)
        ax.set_ylabel("P_model (GPT-2)", fontsize=11)
        ax.set_title("P_model vs PMI_general\n(domain alignment: general corpus)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        for w in [mpatches.Patch(color="#3498db", alpha=0.8, label="General"),
                  mpatches.Patch(color="#8e44ad", alpha=0.8, label="Biomedical")]:
            pass
        ax.legend(handles=[mpatches.Patch(color="#3498db", label="General domain"),
                            mpatches.Patch(color="#8e44ad", label="Biomedical domain"),
                            ax.get_lines()[0]], fontsize=9)

    if len(has_pmi_bio) >= 5:
        ax2 = axes[1]
        x2 = np.array([w["pmi_bio"]      for w in has_pmi_bio])
        y2 = np.array([w["avg_p_model"]  for w in has_pmi_bio])
        cols_d2 = ["#3498db" if w["domain"] == "general" else "#8e44ad" for w in has_pmi_bio]
        ax2.scatter(x2, y2, c=cols_d2, alpha=0.6, s=50)
        slope2, intercept2, r_val2, p_val2, _ = stats.linregress(x2, y2)
        xl2 = np.linspace(x2.min(), x2.max(), 100)
        ax2.plot(xl2, slope2 * xl2 + intercept2, "k--", alpha=0.5,
                 label=f"r={r_val2:.2f}, p={p_val2:.3f}")
        ax2.set_xlabel("PMI_bio (PubMed abstracts)", fontsize=11)
        ax2.set_ylabel("P_model (GPT-2)", fontsize=11)
        ax2.set_title("P_model vs PMI_bio\n(domain alignment: biomedical corpus)", fontsize=11)
        ax2.legend(handles=[mpatches.Patch(color="#3498db", label="General domain"),
                             mpatches.Patch(color="#8e44ad", label="Biomedical domain"),
                             ax2.get_lines()[0]], fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "PMI_bio unavailable\n(PubMed not loaded)",
                     transform=axes[1].transAxes, ha="center", va="center", fontsize=12)
        axes[1].set_title("P_model vs PMI_bio", fontsize=11)

    plt.suptitle("Three-Way Alignment: Does GPT-2 Follow General or Biomedical PMI?",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "three_way_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: Fig 10 — three_way_alignment.png")
else:
    print("Skip: Fig 10 — insufficient PMI data for alignment scatter")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — Four-quadrant Task 3 scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))

quad_color = {
    "HIDDEN_KNOWLEDGE": "#e74c3c",
    "EXPECTED_SUCCESS": "#27ae60",
    "EXPECTED_FAILURE": "#e67e22",
    "EXPECTED_UNKNOWN": "#95a5a6",
}
quad_label = {
    "HIDDEN_KNOWLEDGE": "Hidden Knowledge (Task 3)",
    "EXPECTED_SUCCESS": "Expected Success",
    "EXPECTED_FAILURE": "Expected Failure (PR 1 focus)",
    "EXPECTED_UNKNOWN": "Expected Unknown",
}
quad_size  = {"HIDDEN_KNOWLEDGE": 80, "EXPECTED_FAILURE": 60}

plotted_labels: set[str] = set()
for r in exp4:
    q   = r["quadrant"]
    col = quad_color.get(q, "#bdc3c7")
    lbl = quad_label.get(q, q) if q not in plotted_labels else None
    ax.scatter(r["pmi_general"], r["avg_priming_ratio"],
               c=col, alpha=0.65, s=quad_size.get(q, 40),
               edgecolors="white" if q == "HIDDEN_KNOWLEDGE" else "none",
               linewidths=0.8 if q == "HIDDEN_KNOWLEDGE" else 0,
               label=lbl, zorder=3 if q == "HIDDEN_KNOWLEDGE" else 2)
    if lbl:
        plotted_labels.add(q)

# Annotate top hidden knowledge items
hk_items = sorted([r for r in exp4 if r["quadrant"] == "HIDDEN_KNOWLEDGE"],
                   key=lambda x: -x["avg_priming_ratio"])
for r in hk_items[:8]:
    ax.annotate(f"{r['word1']} {r['word2']}",
                (r["pmi_general"], r["avg_priming_ratio"]),
                fontsize=7.5, color="#922b21", alpha=0.9,
                xytext=(6, 4), textcoords="offset points")

# Quadrant dividers
ax.axvline(exp4_data["metadata"]["thresholds"]["pmi_low"],  color="gray", lw=1, linestyle=":")
ax.axvline(exp4_data["metadata"]["thresholds"]["pmi_high"], color="gray", lw=1, linestyle=":")
ax.axhline(exp4_data["metadata"]["thresholds"]["priming_low"],  color="gray", lw=1, linestyle=":")
ax.axhline(exp4_data["metadata"]["thresholds"]["priming_high"], color="gray", lw=1, linestyle=":")

ax.set_yscale("log")
ax.set_xlabel(f"Corpus PMI ({GENERAL_CORPUS_SHORT})", fontsize=12)
ax.set_ylabel("Model Priming Ratio (log scale)", fontsize=12)
ax.set_title("Task 3 — Four-Quadrant Analysis\n"
             "Hidden Knowledge: low corpus PMI, high model priming", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "four_quadrant_task3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: Fig 11 — four_quadrant_task3.png")

print("\n" + "=" * 65)
print("Analysis complete. All plots saved to results/plots/")
print("=" * 65)
