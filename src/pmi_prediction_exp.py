"""
PMI Bigram Prediction Experiments — four experiments covering Roadmap Tasks 1-3.

Experiment 1 (Task 1): Systematic PMI prediction with logit lens.
    Multi-template priming, per-layer commitment tracking, POS breakdown.

Experiment 2 (Task 2): Minority class bias + over-memorization quantification.
    Multi-template majority/minority probabilities, Over-Memorization Index (OMI),
    logit lens competition curves, PMI-gap threshold analysis.

Experiment 3 (AI4Science): Biomedical three-way PMI alignment.
    Compare P_model against PMI_general and PMI_bio to quantify domain bias
    beyond what corpus frequency predicts.

Experiment 4 (Task 3): Hidden knowledge analysis.
    Four-quadrant (corpus PMI × model priming) classification; characterise
    bigrams the model "knows" that corpus statistics don't predict.

Usage:
    python src/pmi_prediction_exp.py

Prerequisites:
    Run pmi_dataset_builder.py first.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    ROOT = Path(__file__).parent.parent        # local: .py script
except NameError:
    ROOT = Path("/content/llm-washing-machine-claude")   # Colab: notebook cell
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR  = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Python:  {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Device:  {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Load model ───────────────────────────────────────────────────────────────
print("\n=== Loading GPT-2 Small (TransformerLens) ===")
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
tokenizer = model.tokenizer
N_LAYERS = model.cfg.n_layers
print(f"Model: {model.cfg.model_name}, {N_LAYERS} layers, d_model={model.cfg.d_model}")

# ── Multi-template priming setup ─────────────────────────────────────────────
# Six short templates; all put word1 at the final position so word2 is the
# natural next token.  Priming ratios are averaged across templates for
# robustness against prompt-wording artefacts (motivated by PR 1's 8-template approach).
PROMPT_TEMPLATES = [
    "The {w1}",
    "A {w1}",
    "This {w1}",
    "One {w1}",
    "Her {w1}",
    "My {w1}",
]
BASELINE_CONTEXTS = ["The", "A", "This", "One", "Her", "My"]


# ── Helper functions ─────────────────────────────────────────────────────────

def _probs(prompt: str) -> torch.Tensor:
    """Next-token probability distribution after prompt (CPU, vocab-dim)."""
    with torch.no_grad():
        logits = model(prompt)          # [1, seq, vocab]
    return torch.softmax(logits[0, -1, :], dim=-1).cpu()


def _token_id(word: str) -> int | None:
    """First GPT-2 token id for ' word' (with leading space)."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if ids else None


def _prob_rank(probs: torch.Tensor, tid: int) -> tuple[float, int]:
    """(probability, 1-indexed rank) of token id in probs."""
    p = probs[tid].item()
    rank = int((probs > p).sum().item()) + 1
    return p, rank


def multi_template_priming(
    w1: str,
    w2: str,
    return_all: bool = False,
) -> dict:
    """
    Compute priming ratio and rank for (w1, w2) averaged across all templates.

    Returns a dict with keys:
        avg_priming_ratio, avg_rank, avg_p_w2,
        top_1_correct, top_5_correct, top_10_correct,
        template_results (list, only when return_all=True)
    """
    tid = _token_id(w2)
    if tid is None:
        return {"avg_priming_ratio": 0, "avg_rank": 9999,
                "avg_p_w2": 0, "top_1_correct": False,
                "top_5_correct": False, "top_10_correct": False}

    priming_ratios, ranks, p_w2s = [], [], []
    template_results = []

    for tmpl, base_probs in zip(PROMPT_TEMPLATES, _BASELINE_PROBS):
        prompt   = tmpl.format(w1=w1)
        probs    = _probs(prompt)

        p_w2, rank = _prob_rank(probs, tid)
        p_base, _  = _prob_rank(base_probs, tid)
        ratio = p_w2 / max(p_base, 1e-10)

        priming_ratios.append(ratio)
        ranks.append(rank)
        p_w2s.append(p_w2)

        if return_all:
            template_results.append({
                "template": tmpl,
                "p_w2": round(p_w2, 6),
                "p_base": round(p_base, 6),
                "priming_ratio": round(ratio, 2),
                "rank": rank,
            })

    avg_ratio = float(np.mean(priming_ratios))
    avg_rank  = float(np.mean(ranks))
    avg_p     = float(np.mean(p_w2s))
    out = {
        "avg_priming_ratio": round(avg_ratio, 2),
        "avg_rank": round(avg_rank, 1),
        "avg_p_w2": round(avg_p, 6),
        "top_1_correct": avg_rank <= 1.5,      # correct in majority of templates
        "top_5_correct": avg_rank <= 5.0,
        "top_10_correct": avg_rank <= 10.0,
    }
    if return_all:
        out["template_results"] = template_results
    return out


# Pre-compute all 6 baseline probability tensors once (reused across all experiments)
_BASELINE_PROBS = [_probs(ctx) for ctx in BASELINE_CONTEXTS]
_BASELINE_PROBS_FIRST = _BASELINE_PROBS[0]   # "The" — used for logit lens commitment
print(f"Pre-computed {len(_BASELINE_PROBS)} baseline probability tensors.")


def logit_lens_probs(prompt: str, tids: list[int]) -> list[list[float]]:
    """
    Track probability of each token id in tids across all layers via logit lens.

    Returns list of length len(tids), each containing n_layers floats.

    Logit lens:  residual_post[layer]  →  ln_final  →  unembed  →  softmax
    This reuses the final layer norm and unembedding, giving the model's
    "prediction so far" at each intermediate layer.
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)

    per_token: list[list[float]] = [[] for _ in tids]

    for layer in range(N_LAYERS):
        resid = cache[f"blocks.{layer}.hook_resid_post"]   # [1, seq, d_model]
        normed = model.ln_final(resid)                      # [1, seq, d_model]
        logits = model.unembed(normed)                      # [1, seq, vocab]
        probs  = torch.softmax(logits[0, -1, :], dim=-1).cpu()
        for i, tid in enumerate(tids):
            per_token[i].append(probs[tid].item())

    return per_token


def commitment_layer(layer_probs: list[float], baseline_p: float, threshold: float = 10.0) -> int | None:
    """Earliest layer where P(word2) > threshold × baseline.  None if never."""
    for l, p in enumerate(layer_probs):
        if p > baseline_p * threshold:
            return l
    return None


def over_memorization_index(
    p_maj_model: float,
    p_min_model: float,
    freq_maj_corpus: int | None,
    freq_min_corpus: int | None,
) -> float | None:
    """
    OMI = (P_model(majority) / P_model(minority))
           / (freq_corpus(majority) / freq_corpus(minority))

    OMI > 1 → model is MORE biased toward majority than corpus predicts.
    Returns None if any input is zero/missing.
    """
    if p_min_model <= 0:
        return None
    if not freq_maj_corpus or not freq_min_corpus:
        return None
    model_ratio  = p_maj_model / p_min_model
    corpus_ratio = freq_maj_corpus / freq_min_corpus
    return round(model_ratio / corpus_ratio, 3)


def top_k_preds(prompt: str, k: int = 10) -> list[tuple[str, float]]:
    """Top-k (decoded_string, probability) from the first template only."""
    probs = _probs(prompt)
    top = torch.topk(probs, k=k)
    return [(tokenizer.decode([top.indices[i].item()]), round(top.values[i].item(), 6))
            for i in range(k)]


def multi_template_word_stats(word: str, prompt_probs: list) -> dict:
    """
    Multi-template word stats from pre-computed prompt probability tensors.

    prompt_probs: list of 6 torch.Tensors, one per template, already computed
                  for the prefix of this group.  Compute once per group with
                  [_probs(tmpl.format(w1=prefix)) for tmpl in PROMPT_TEMPLATES],
                  then reuse for majority + all minorities in that group.
    Baselines come from global _BASELINE_PROBS (computed once at startup).
    """
    tid = _token_id(word)
    if tid is None:
        return {"avg_p": 0.0, "avg_rank": 9999, "avg_priming_ratio": 0.0}
    priming_rs, rnks, ps = [], [], []
    for pr, bpr in zip(prompt_probs, _BASELINE_PROBS):
        p, rk = _prob_rank(pr, tid)
        pb, _ = _prob_rank(bpr, tid)
        priming_rs.append(p / max(pb, 1e-10))
        rnks.append(rk)
        ps.append(p)
    return {
        "avg_p": round(float(np.mean(ps)), 6),
        "avg_rank": round(float(np.mean(rnks)), 1),
        "avg_priming_ratio": round(float(np.mean(priming_rs)), 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Task 1: Systematic PMI Prediction + Logit Lens
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 1 — Task 1: PMI Prediction + Logit Lens")
print("=" * 60)

with open(DATASETS_DIR / "high_pmi_bigrams.json") as f:
    pmi_data = json.load(f)
bigrams = pmi_data["bigrams"]
print(f"Loaded {len(bigrams)} bigrams")

exp1_results: list[dict] = []

for i, bg in enumerate(bigrams):
    w1, w2 = bg["word1"], bg["word2"]
    tid = _token_id(w2)

    # Multi-template priming
    mt = multi_template_priming(w1, w2)

    # Logit lens (using "The {w1}" — first template)
    lens_probs: list[float] | None = None
    commit_layer: int | None = None
    if tid is not None:
        lp = logit_lens_probs(f"The {w1}", [tid])
        lens_probs = [round(p, 6) for p in lp[0]]
        base_p = _BASELINE_PROBS_FIRST[tid].item()
        commit_layer = commitment_layer(lens_probs, base_p)

    exp1_results.append({
        "word1": w1,
        "word2": w2,
        "pmi_general": bg["pmi_general"],
        "corpus_freq": bg["freq"],
        "pos_pattern": bg.get("pos_pattern", "?"),
        "avg_priming_ratio": mt["avg_priming_ratio"],
        "avg_rank": mt["avg_rank"],
        "avg_p_w2": mt["avg_p_w2"],
        "top_1_correct": mt["top_1_correct"],
        "top_5_correct": mt["top_5_correct"],
        "top_10_correct": mt["top_10_correct"],
        "logit_lens_probs": lens_probs,
        "commitment_layer": commit_layer,
        "top_10_predictions": top_k_preds(f"The {w1}"),
    })

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(bigrams)} done ...")

# Summary statistics
n = len(exp1_results)
top1 = sum(r["top_1_correct"] for r in exp1_results) / n
top5 = sum(r["top_5_correct"] for r in exp1_results) / n
top10 = sum(r["top_10_correct"] for r in exp1_results) / n

pmi_arr     = np.array([r["pmi_general"] for r in exp1_results])
priming_arr = np.array([r["avg_priming_ratio"] for r in exp1_results])
sp_r, sp_p  = stats.spearmanr(pmi_arr, priming_arr)

commit_valid = [r["commitment_layer"] for r in exp1_results if r["commitment_layer"] is not None]
median_commit = int(np.median(commit_valid)) if commit_valid else None

# POS breakdown
from collections import defaultdict as _dd
pos_stats: dict[str, dict] = _dd(lambda: {"n": 0, "top1": 0})
for r in exp1_results:
    p = r["pos_pattern"]
    pos_stats[p]["n"] += 1
    pos_stats[p]["top1"] += int(r["top_1_correct"])

print(f"\nTask 1 Results (n={n}, {len(PROMPT_TEMPLATES)}-template avg):")
print(f"  Top-1:  {top1:.1%}   Top-5:  {top5:.1%}   Top-10: {top10:.1%}")
print(f"  Median priming:  {np.median(priming_arr):.1f}x")
print(f"  Spearman(PMI, priming): r={sp_r:.3f}, p={sp_p:.4f}")
print(f"  Median commitment layer: {median_commit}  ({len(commit_valid)}/{n} bigrams committed)")
print("  POS breakdown (top-1 acc):")
for pos, s in sorted(pos_stats.items(), key=lambda x: -x[1]["n"])[:8]:
    acc = s["top1"] / s["n"]
    print(f"    {pos:12s}: n={s['n']:3d}, acc={acc:.1%}")

exp1_out = {
    "summary": {
        "n": n,
        "n_templates": len(PROMPT_TEMPLATES),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "top10_accuracy": round(top10, 4),
        "median_priming_ratio": round(float(np.median(priming_arr)), 2),
        "mean_priming_ratio": round(float(np.mean(priming_arr)), 2),
        "spearman_pmi_priming_r": round(sp_r, 4),
        "spearman_pmi_priming_p": round(sp_p, 6),
        "median_commitment_layer": median_commit,
        "pct_committed": round(len(commit_valid) / n, 4),
        "pos_breakdown": {
            pos: {"n": s["n"], "top1_acc": round(s["top1"] / s["n"], 4)}
            for pos, s in pos_stats.items()
        },
    },
    "results": exp1_results,
}
with open(RESULTS_DIR / "pmi_prediction_results.json", "w") as f:
    json.dump(exp1_out, f, indent=2)
print("Saved: results/pmi_prediction_results.json")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Task 2: Minority Class + Over-Memorization Index
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2 — Task 2: Minority Class + Over-Memorization Index")
print("=" * 60)

with open(DATASETS_DIR / "minority_class_pairs.json") as f:
    min_data = json.load(f)
minority_pairs = min_data["pairs"]
print(f"Loaded {len(minority_pairs)} minority class groups")

exp2_results: dict[str, dict] = {}

for i, (prefix, group) in enumerate(minority_pairs.items()):
    # Pre-compute 6 prompt probs once for this prefix — shared by majority + all minorities
    prompt_probs = [_probs(tmpl.format(w1=prefix)) for tmpl in PROMPT_TEMPLATES]

    maj_entry = group["majority"][0]
    maj_word  = maj_entry["word"]
    maj_mt    = multi_template_word_stats(maj_word, prompt_probs)

    minority_rows = []
    for min_entry in group["minority"]:
        min_word = min_entry["word"]
        min_mt   = multi_template_word_stats(min_word, prompt_probs)
        omi = over_memorization_index(
            maj_mt["avg_p"], min_mt["avg_p"],
            maj_entry.get("corpus_freq"), min_entry.get("corpus_freq"),
        )
        # PMI gap
        pmi_gap = None
        if maj_entry.get("pmi_general") and min_entry.get("pmi_general"):
            pmi_gap = round(maj_entry["pmi_general"] - min_entry["pmi_general"], 4)
        minority_rows.append({
            "word":             min_word,
            "avg_p_model":      min_mt["avg_p"],
            "avg_rank":         min_mt["avg_rank"],
            "avg_priming_ratio": min_mt["avg_priming_ratio"],
            "corpus_freq":      min_entry.get("corpus_freq"),
            "pmi_general":      min_entry.get("pmi_general"),
            "over_memorization_index": omi,
            "pmi_gap_vs_majority":     pmi_gap,
        })

    # ── Logit lens competition curve (one pass, first template) ──────────
    maj_tid = _token_id(maj_word)
    min_tids = [_token_id(r["word"]) for r in minority_rows]
    valid_min_tids = [t for t in min_tids if t is not None]

    logit_competition: dict | None = None
    if maj_tid and valid_min_tids:
        all_probs = logit_lens_probs(f"The {prefix}", [maj_tid] + valid_min_tids)
        logit_competition = {
            "majority_lens": [round(p, 6) for p in all_probs[0]],
            "minority_lens_mean": [round(float(np.mean([all_probs[j][l]
                                   for j in range(1, len(all_probs))])), 6)
                                   for l in range(N_LAYERS)],
        }
        # Find crossover layer: first layer where majority > all minorities
        crossover = None
        for l in range(N_LAYERS):
            maj_p = all_probs[0][l]
            if all(all_probs[j][l] < maj_p for j in range(1, len(all_probs))):
                crossover = l
                break
        logit_competition["majority_dominates_from_layer"] = crossover

    exp2_results[prefix] = {
        "majority": {
            "word":             maj_word,
            "avg_p_model":      maj_mt["avg_p"],
            "avg_rank":         maj_mt["avg_rank"],
            "avg_priming_ratio": maj_mt["avg_priming_ratio"],
            "corpus_freq":      maj_entry.get("corpus_freq"),
            "pmi_general":      maj_entry.get("pmi_general"),
        },
        "minority": minority_rows,
        "logit_lens_competition": logit_competition,
        "top_10_predictions": top_k_preds(f"The {prefix}"),
        "summary": {
            "maj_avg_p": maj_mt["avg_p"],
            "min_mean_avg_p": round(float(np.mean([r["avg_p_model"] for r in minority_rows])), 6),
            "min_max_avg_p":  round(float(max(r["avg_p_model"] for r in minority_rows)), 6),
            "prob_ratio":     round(maj_mt["avg_p"] / max(
                float(np.mean([r["avg_p_model"] for r in minority_rows])), 1e-10), 2),
            "mean_omi": round(float(np.mean([r["over_memorization_index"]
                              for r in minority_rows
                              if r["over_memorization_index"] is not None])), 3)
                        if any(r["over_memorization_index"] for r in minority_rows) else None,
        },
    }
    ratio = exp2_results[prefix]["summary"]["prob_ratio"]
    print(f"  [{i+1:3d}/{len(minority_pairs)}] {prefix:<15s} maj={maj_word:<12s} "
          f"{len(minority_rows)} minority  ratio={ratio:.1f}x")

# ── Statistical tests ──────────────────────────────────────────────────────
all_maj_p, all_min_mean_p, all_omi, all_pmi_gap, all_correct = [], [], [], [], []
for prefix, result in exp2_results.items():
    maj_p = result["majority"]["avg_p_model"]
    min_p_list = [r["avg_p_model"] for r in result["minority"]]
    min_mean   = float(np.mean(min_p_list))
    all_maj_p.append(maj_p)
    all_min_mean_p.append(min_mean)
    all_correct.append(int(maj_p > min_mean))
    for r in result["minority"]:
        if r["over_memorization_index"] is not None:
            all_omi.append(r["over_memorization_index"])
        if r["pmi_gap_vs_majority"] is not None:
            all_pmi_gap.append(r["pmi_gap_vs_majority"])
            all_correct.append(int(maj_p > r["avg_p_model"]))

w_stat, w_p = (stats.wilcoxon(all_maj_p, all_min_mean_p)
               if len(all_maj_p) >= 5 else (None, None))
pmi_gap_arr = np.array([r["pmi_gap_vs_majority"] for rr in exp2_results.values()
                         for r in rr["minority"] if r["pmi_gap_vs_majority"] is not None])
correct_arr = np.array([int(rr["majority"]["avg_p_model"] > r["avg_p_model"])
                         for rr in exp2_results.values()
                         for r in rr["minority"] if r["pmi_gap_vs_majority"] is not None])
gap_sp_r, gap_sp_p = (stats.spearmanr(pmi_gap_arr, correct_arr)
                       if len(pmi_gap_arr) >= 5 else (None, None))

median_adv = float(np.median([m / max(n, 1e-10) for m, n in zip(all_maj_p, all_min_mean_p)]))
omi_median = float(np.median(all_omi)) if all_omi else None
omi_pct_gt1 = float(np.mean([o > 1 for o in all_omi])) if all_omi else None

print(f"\nTask 2 Results (n={len(exp2_results)} groups):")
print(f"  Median majority advantage: {median_adv:.1f}x")
if w_p is not None:
    print(f"  Wilcoxon (maj vs min): stat={w_stat:.2f}, p={w_p:.4f}")
print(f"  Over-Memorization Index: median={omi_median:.2f}, % > 1 = {omi_pct_gt1:.1%}" if omi_median else "  OMI: N/A")
if gap_sp_r is not None:
    print(f"  Spearman(PMI_gap, correct): r={gap_sp_r:.3f}, p={gap_sp_p:.4f}")

exp2_out = {
    "summary": {
        "n_groups": len(exp2_results),
        "n_templates": len(PROMPT_TEMPLATES),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "median_majority_advantage": round(median_adv, 2),
        "omi_median": round(omi_median, 3) if omi_median else None,
        "omi_pct_above_1": round(omi_pct_gt1, 4) if omi_pct_gt1 is not None else None,
        "spearman_pmi_gap_correct_r": round(gap_sp_r, 4) if gap_sp_r else None,
        "spearman_pmi_gap_correct_p": round(gap_sp_p, 6) if gap_sp_p else None,
    },
    "results": exp2_results,
}
with open(RESULTS_DIR / "minority_class_results.json", "w") as f:
    json.dump(exp2_out, f, indent=2)
print("Saved: results/minority_class_results.json")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — AI4Science: Biomedical Three-Way PMI Alignment
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 3 — AI4Science: Biomedical Three-Way PMI Alignment")
print("=" * 60)

with open(DATASETS_DIR / "cross_domain_pairs.json") as f:
    cd_data = json.load(f)
cd_pairs = cd_data["pairs"]
print(f"Loaded {len(cd_pairs)} cross-domain groups")
pubmed_available = cd_data["metadata"].get("source_bio_pmi", "unavailable") != "unavailable"
print(f"PubMed PMI_bio available: {pubmed_available}")

exp3_results: list[dict] = []

for i, group in enumerate(cd_pairs):
    prefix = group["prefix"]
    # Pre-compute 6 prompt probs once for this prefix (shared by all general + biomedical words)
    prompt_probs = [_probs(tmpl.format(w1=prefix)) for tmpl in PROMPT_TEMPLATES]

    def _process_words(entries: list[dict]) -> list[dict]:
        rows = []
        for e in entries:
            word = e["word"]
            tid  = _token_id(word)
            if tid is None:
                continue
            mt = multi_template_word_stats(word, prompt_probs)
            rows.append({
                "word":           word,
                "domain":         e["domain"],
                "avg_p_model":    mt["avg_p"],
                "avg_rank":       mt["avg_rank"],
                "avg_priming_ratio": mt["avg_priming_ratio"],
                "pmi_general":    e.get("pmi_general"),
                "pmi_bio":        e.get("pmi_bio"),
                "corpus_freq_general": e.get("corpus_freq_general"),
                "corpus_freq_bio": e.get("corpus_freq_bio"),
            })
        return rows

    gen_rows = _process_words(group["general"])
    bio_rows = _process_words(group["biomedical"])

    gen_p_list = [r["avg_p_model"] for r in gen_rows]
    bio_p_list = [r["avg_p_model"] for r in bio_rows]
    gen_max = float(max(gen_p_list)) if gen_p_list else 0.0
    bio_max = float(max(bio_p_list)) if bio_p_list else 0.0

    exp3_results.append({
        "prefix":    prefix,
        "ambiguity": group["ambiguity"],
        "note":      group["note"],
        "general":   gen_rows,
        "biomedical": bio_rows,
        "top_10_predictions": top_k_preds(f"The {prefix}"),
        "summary": {
            "gen_mean_p": round(float(np.mean(gen_p_list)) if gen_p_list else 0, 6),
            "bio_mean_p": round(float(np.mean(bio_p_list)) if bio_p_list else 0, 6),
            "gen_max_p":  round(gen_max, 6),
            "bio_max_p":  round(bio_max, 6),
            "general_favoured": gen_max > bio_max,
        },
    })
    print(f"  [{i+1}/{len(cd_pairs)}] {prefix:12s}: gen_max={gen_max:.4f}  bio_max={bio_max:.4f}  "
          f"({'general' if gen_max > bio_max else 'biomedical'} favoured)")

# ── Domain alignment: Spearman(P_model, PMI_general) vs Spearman(P_model, PMI_bio) ──
all_gen_bio_words: list[dict] = []
for r in exp3_results:
    for row in r["general"] + r["biomedical"]:
        all_gen_bio_words.append(row)

p_model_arr   = np.array([w["avg_p_model"]  for w in all_gen_bio_words])
pmi_gen_arr   = np.array([w["pmi_general"]  if w["pmi_general"]  is not None else 0.0 for w in all_gen_bio_words])
pmi_bio_arr   = np.array([w["pmi_bio"]      if w["pmi_bio"]      is not None else 0.0 for w in all_gen_bio_words])

has_pmi_gen = np.array([w["pmi_general"] is not None for w in all_gen_bio_words])
has_pmi_bio = np.array([w["pmi_bio"]     is not None for w in all_gen_bio_words])

align_gen = stats.spearmanr(p_model_arr[has_pmi_gen], pmi_gen_arr[has_pmi_gen]) \
            if has_pmi_gen.sum() >= 5 else (None, None)
align_bio = stats.spearmanr(p_model_arr[has_pmi_bio], pmi_bio_arr[has_pmi_bio]) \
            if has_pmi_bio.sum() >= 5 else (None, None)

gen_max_list = [r["summary"]["gen_max_p"] for r in exp3_results if r["general"]]
bio_max_list = [r["summary"]["bio_max_p"] for r in exp3_results if r["biomedical"]]
n_gen_fav = sum(r["summary"]["general_favoured"] for r in exp3_results)

mwu = stats.mannwhitneyu(gen_max_list, bio_max_list, alternative="two-sided") \
      if len(gen_max_list) >= 5 else type("R", (), {"statistic": None, "pvalue": None})()

gen_bio_ratio = (float(np.mean(gen_max_list)) / max(float(np.mean(bio_max_list)), 1e-10)
                 if gen_max_list and bio_max_list else None)

print(f"\nTask 3 / Biomedical Results (n={len(exp3_results)} groups):")
print(f"  General favoured: {n_gen_fav}/{len(exp3_results)}")
print(f"  Mean gen/bio ratio: {gen_bio_ratio:.1f}x" if gen_bio_ratio else "")
if align_gen[0] is not None:
    print(f"  align_general: Spearman r={align_gen[0]:.3f}, p={align_gen[1]:.4f}")
if align_bio[0] is not None:
    print(f"  align_bio:     Spearman r={align_bio[0]:.3f}, p={align_bio[1]:.4f}")
if mwu.pvalue is not None:
    print(f"  Mann-Whitney U: stat={mwu.statistic:.2f}, p={mwu.pvalue:.4f}")

exp3_out = {
    "summary": {
        "n_groups": len(exp3_results),
        "n_templates": len(PROMPT_TEMPLATES),
        "pubmed_pmi_available": pubmed_available,
        "n_general_favoured":   n_gen_fav,
        "general_to_bio_ratio": round(gen_bio_ratio, 2) if gen_bio_ratio else None,
        "mean_gen_max_p": round(float(np.mean(gen_max_list)), 6) if gen_max_list else None,
        "mean_bio_max_p": round(float(np.mean(bio_max_list)), 6) if bio_max_list else None,
        "mann_whitney_stat": mwu.statistic,
        "mann_whitney_p":    mwu.pvalue,
        "domain_alignment_general_r": round(align_gen[0], 4) if align_gen[0] is not None else None,
        "domain_alignment_general_p": round(align_gen[1], 6) if align_gen[1] is not None else None,
        "domain_alignment_bio_r":     round(align_bio[0], 4) if align_bio[0] is not None else None,
        "domain_alignment_bio_p":     round(align_bio[1], 6) if align_bio[1] is not None else None,
    },
    "results": exp3_results,
}
with open(RESULTS_DIR / "cross_domain_results.json", "w") as f:
    json.dump(exp3_out, f, indent=2)
print("Saved: results/cross_domain_results.json")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Task 3: Hidden Knowledge Analysis
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 4 — Task 3: Hidden Knowledge Analysis")
print("=" * 60)

# Four-quadrant classification
# Axis 1: corpus PMI  — Low < 6, High >= 8   (medium 6-8 excluded from quadrants)
# Axis 2: model priming — Low < 10x, High >= 50x
PMI_LOW_THRESH     = 6.0
PMI_HIGH_THRESH    = 8.0
PRIMING_LOW_THRESH = 10.0
PRIMING_HIGH_THRESH = 50.0

quadrant_labels = {
    ("low_pmi",  "high_priming"): "HIDDEN_KNOWLEDGE",   # Task 3 focus
    ("high_pmi", "high_priming"): "EXPECTED_SUCCESS",
    ("high_pmi", "low_priming"):  "EXPECTED_FAILURE",    # PR 1 focus
    ("low_pmi",  "low_priming"):  "EXPECTED_UNKNOWN",
}

exp4_results: list[dict] = []
for r in exp1_results:
    pmi_val    = r["pmi_general"]
    priming    = r["avg_priming_ratio"]

    if pmi_val >= PMI_HIGH_THRESH:
        pmi_cat = "high_pmi"
    elif pmi_val < PMI_LOW_THRESH:
        pmi_cat = "low_pmi"
    else:
        pmi_cat = "medium_pmi"  # transition zone

    priming_cat = "high_priming" if priming >= PRIMING_HIGH_THRESH else "low_priming"

    quadrant = quadrant_labels.get((pmi_cat, priming_cat), f"{pmi_cat}/{priming_cat}")

    exp4_results.append({
        "word1": r["word1"],
        "word2": r["word2"],
        "pmi_general": pmi_val,
        "avg_priming_ratio": priming,
        "avg_rank": r["avg_rank"],
        "top_1_correct": r["top_1_correct"],
        "pos_pattern": r["pos_pattern"],
        "commitment_layer": r["commitment_layer"],
        "pmi_category":    pmi_cat,
        "priming_category": priming_cat,
        "quadrant": quadrant,
    })

# Summarise each quadrant
from collections import Counter as _Counter
quad_counts  = _Counter(r["quadrant"] for r in exp4_results)
hidden_items = [r for r in exp4_results if r["quadrant"] == "HIDDEN_KNOWLEDGE"]
hk_pos_dist  = _Counter(r["pos_pattern"] for r in hidden_items)

print(f"Four-quadrant distribution (n={len(exp4_results)}):")
for quad, cnt in sorted(quad_counts.items(), key=lambda x: -x[1]):
    pct = cnt / len(exp4_results)
    print(f"  {quad:25s}: {cnt:3d} ({pct:.1%})")

print(f"\nHidden Knowledge bigrams (n={len(hidden_items)}):")
for r in sorted(hidden_items, key=lambda x: -x["avg_priming_ratio"])[:15]:
    print(f"  '{r['word1']} {r['word2']}': PMI={r['pmi_general']:.1f}, "
          f"priming={r['avg_priming_ratio']:.0f}x, POS={r['pos_pattern']}, "
          f"commit=layer{r['commitment_layer']}")

print(f"\nPOS pattern distribution in HIDDEN_KNOWLEDGE:")
for pos, cnt in hk_pos_dist.most_common():
    print(f"  {pos:12s}: {cnt}")

exp4_out = {
    "metadata": {
        "description": "Four-quadrant Task 3 analysis. HIDDEN_KNOWLEDGE = low corpus PMI but high model priming.",
        "thresholds": {
            "pmi_low":     PMI_LOW_THRESH,
            "pmi_high":    PMI_HIGH_THRESH,
            "priming_low": PRIMING_LOW_THRESH,
            "priming_high": PRIMING_HIGH_THRESH,
        },
    },
    "summary": {
        "n_total": len(exp4_results),
        "quadrant_counts": dict(quad_counts),
        "hidden_knowledge_n": len(hidden_items),
        "hidden_knowledge_pct": round(len(hidden_items) / len(exp4_results), 4),
        "hidden_knowledge_pos_distribution": dict(hk_pos_dist),
    },
    "results": exp4_results,
}
with open(RESULTS_DIR / "hidden_knowledge_results.json", "w") as f:
    json.dump(exp4_out, f, indent=2)
print("Saved: results/hidden_knowledge_results.json")

print("\n" + "=" * 60)
print("All experiments complete.")
print("Next: python src/analyze_pmi_results.py")
print("=" * 60)
