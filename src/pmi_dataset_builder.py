"""
PMI Dataset Builder.

Builds three datasets:

    Part A — high_pmi_bigrams.json (150-200 items)
        OpenWebText (first 100k docs, English-only); freq >= 10; single GPT-2
        token second word; NLTK POS tags added; tier-sampled across PMI range.

    Part B — minority_class_pairs.json (target: 60-80 groups)
        Fully automatic: for each Part-A prefix, find all single-token
        completions; keep groups with a clear majority (top completion >=
        3x any minority) and at least 2 minority completions (freq >= 5).

    Part C — cross_domain_pairs.json (target: 30-35 groups)
        Reads datasets/cross_domain_raw.json (manually curated).
        Verifies single-token status, attaches PMI_general (from corpus)
        and, optionally, PMI_bio (from PubMed abstracts via HuggingFace).

Usage:
    python src/pmi_dataset_builder.py

Dependencies (beyond stdlib):
    nltk, transformers, datasets (HuggingFace: OpenWebText + optional PubMed PMI)

Import order: load transformers (torch) before ``datasets`` to avoid WinError 1114
when PyArrow loads first on some Windows setups.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# ── Colab Cell 0 (run BEFORE this cell) ─────────────────────────────────────
# !pip install -q transformers datasets nltk
# !git clone https://github.com/<your-username>/llm-washing-machine-claude.git
# (pmi_prediction_exp needs: transformer-lens / transformer_lens, scipy, numpy — see that cell)
# ─────────────────────────────────────────────────────────────────────────────

try:
    ROOT = Path(__file__).parent.parent        # local: .py script
except NameError:
    ROOT = Path("/content/llm-washing-machine-claude")   # Colab: notebook cell

# HuggingFace dataset for general English PMI (sentence-internal bigrams)
# OpenWebText: English-only web text in the same domain as GPT-2's training data.
# Using a capped stream (OWT_MAX_DOCS docs) for practical runtime on CPU.
OWT_DATASET   = "Skylion007/openwebtext"
OWT_SPLIT     = "train"
OWT_MAX_DOCS  = 100_000   # ~50M words; sufficient for reliable PMI statistics
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Python: {sys.version}")
print(f"NLTK:   {nltk.__version__}")

# ── NLTK downloads ──────────────────────────────────────────────────────────
print("\nDownloading NLTK data (skipped if already present)...")
for pkg in (
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",  # NLTK 3.9+ English POS (PerceptronTagger)
    "punkt_tab",
):
    nltk.download(pkg, quiet=True)

# ── GPT-2 tokenizer ─────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer...")
from transformers import GPT2Tokenizer as _GPT2Tok

_tok = _GPT2Tok.from_pretrained("gpt2")

from datasets import load_dataset


def is_single_gpt2_token(word: str) -> bool:
    """True iff ' word' (with leading space) is exactly one GPT-2 token."""
    return len(_tok.encode(" " + word, add_special_tokens=False)) == 1


# ── General corpus statistics (OpenWebText, English-only) ───────────────────
print("\n" + "=" * 60)
print(f"Computing corpus statistics ({OWT_DATASET}, first {OWT_MAX_DOCS:,} docs)...")
print("=" * 60)


def _build_corpus_stats() -> tuple[Counter, Counter, int, int]:
    """Stream OpenWebText (English-only); sentence tokenize; count unigrams + ASCII alpha bigrams.

    Stops after OWT_MAX_DOCS non-empty documents for practical CPU runtime.
    ASCII filter (isascii()) removes non-English tokens that survive isalpha().
    """
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    ds = load_dataset(
        OWT_DATASET,
        split=OWT_SPLIT,
        streaming=True,
    )
    n_blocks = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        for sent in sent_tokenize(text):
            words = [w.lower() for w in word_tokenize(sent)]
            if not words:
                continue
            # isascii() guards against non-English tokens that pass isalpha()
            ascii_words = [w for w in words if w.isalpha() and w.isascii()]
            unigram_counts.update(ascii_words)
            for i in range(len(ascii_words) - 1):
                bigram_counts[(ascii_words[i], ascii_words[i + 1])] += 1
        n_blocks += 1
        if n_blocks % 10_000 == 0:
            print(f"  ... {n_blocks:,}/{OWT_MAX_DOCS:,} docs, |V|≈{len(unigram_counts):,} unigrams")
        if n_blocks >= OWT_MAX_DOCS:
            break
    total_w = sum(unigram_counts.values())
    total_b = sum(bigram_counts.values())
    return unigram_counts, bigram_counts, total_w, total_b


unigram_counts, bigram_counts, _total_words, _total_bigrams = _build_corpus_stats()

print(f"  Total words:          {_total_words:,}")
print(f"  Unique alpha bigrams: {len(bigram_counts):,}")
print(f"  Total alpha bigrams:  {_total_bigrams:,}")


def pmi_general(w1: str, w2: str) -> float | None:
    """PMI from OpenWebText (sentence-internal ASCII-alpha bigrams). None if count is zero."""
    cnt = bigram_counts[(w1, w2)]
    if cnt == 0:
        return None
    p12 = cnt / _total_bigrams
    p1  = unigram_counts[w1] / _total_words
    p2  = unigram_counts[w2] / _total_words
    return math.log2(p12 / (p1 * p2))


# ── Per-prefix completion lookup (shared by Parts B and C) ──────────────────
print("\nBuilding per-prefix completion lookup (freq >= 5, single-token)...")
prefix_completions: dict[str, list[dict]] = defaultdict(list)
for (w1, w2), freq in bigram_counts.items():
    if freq >= 5 and w1.isalpha() and w2.isalpha() and is_single_gpt2_token(w2):
        v = pmi_general(w1, w2)
        prefix_completions[w1].append(
            {"word": w2, "freq": freq, "pmi_general": round(v, 4) if v else None}
        )
for v in prefix_completions.values():
    v.sort(key=lambda x: x["freq"], reverse=True)
print(f"  Lookup covers {len(prefix_completions):,} prefix words")


# ── PART A — High-PMI Bigrams ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Part A: High-PMI Bigrams")
print("=" * 60)

MIN_FREQ = 10
PMI_HIGH, PMI_MED = 10.0, 6.0

print(f"Filtering bigrams (freq >= {MIN_FREQ}, single-token second word)...")
valid_bigrams: list[dict] = []
for (w1, w2), freq in bigram_counts.items():
    if freq < MIN_FREQ:
        continue
    if not is_single_gpt2_token(w2):
        continue
    v = pmi_general(w1, w2)
    if v is None:
        continue
    valid_bigrams.append(
        {
            "word1": w1,
            "word2": w2,
            "pmi_general": round(v, 4),
            "freq": freq,
            "word1_freq": unigram_counts[w1],
            "word2_freq": unigram_counts[w2],
        }
    )

valid_bigrams.sort(key=lambda x: x["pmi_general"], reverse=True)
print(f"  Valid (after single-token filter): {len(valid_bigrams):,}")

# Tier-sample
tier_h = [b for b in valid_bigrams if b["pmi_general"] >= PMI_HIGH]
tier_m = [b for b in valid_bigrams if PMI_MED <= b["pmi_general"] < PMI_HIGH]
tier_l = [b for b in valid_bigrams if b["pmi_general"] < PMI_MED]
selected = (tier_h[:100] + tier_m[:60] + tier_l[:40])[:200]

# Add POS pattern via NLTK
print("Adding POS patterns...")
for b in selected:
    tag1 = nltk.pos_tag([b["word1"]])[0][1]
    tag2 = nltk.pos_tag([b["word2"]])[0][1]
    b["pos_pattern"] = f"{tag1}-{tag2}"

print(f"\nSelected {len(selected)} bigrams:")
print(f"  High PMI (>= {PMI_HIGH}):       {sum(1 for b in selected if b['pmi_general'] >= PMI_HIGH)}")
print(f"  Med  PMI ({PMI_MED}–{PMI_HIGH}): {sum(1 for b in selected if PMI_MED <= b['pmi_general'] < PMI_HIGH)}")
print(f"  Low  PMI (< {PMI_MED}):         {sum(1 for b in selected if b['pmi_general'] < PMI_MED)}")
print("\nTop 20 by PMI:")
for b in selected[:20]:
    print(f"  '{b['word1']} {b['word2']}': PMI={b['pmi_general']:.2f}  freq={b['freq']}  POS={b['pos_pattern']}")

partA = {
    "metadata": {
        "description": (
            "High-PMI bigrams from HuggingFace OpenWebText (English-only, first 100k docs). "
            "Sentence tokenization (NLTK); word tokenization (NLTK); ASCII-alpha bigrams within sentences. "
            "Second word is a single GPT-2 token. Tier-sampled across PMI range. NLTK POS pattern included."
        ),
        "source": f"HuggingFace {OWT_DATASET} split={OWT_SPLIT} (streaming, first {OWT_MAX_DOCS:,} docs)",
        "min_freq": MIN_FREQ,
        "total_corpus_words": _total_words,
        "valid_bigrams_found": len(valid_bigrams),
        "pmi_tiers": {"high": f">={PMI_HIGH}", "medium": f"{PMI_MED}-{PMI_HIGH}", "low": f"<{PMI_MED}"},
    },
    "bigrams": selected,
}
out = DATASETS_DIR / "high_pmi_bigrams.json"
with open(out, "w") as f:
    json.dump(partA, f, indent=2)
print(f"\nSaved Part A → {out}")


# ── PART B — Minority Class Pairs (automatic) ────────────────────────────────
print("\n" + "=" * 60)
print("Part B: Minority Class Pairs (automatic extraction)")
print("=" * 60)

MAJORITY_RATIO = 3      # majority must be >= 3x any minority freq
MIN_MINORITY   = 2      # need at least 2 minority completions
MIN_MIN_FREQ   = 5      # each minority completion: freq >= 5

part_a_prefixes = {b["word1"] for b in selected}


def extract_minority_group(prefix: str) -> dict | None:
    completions = prefix_completions.get(prefix, [])
    if len(completions) < 1 + MIN_MINORITY:
        return None
    maj = completions[0]
    threshold = maj["freq"] / MAJORITY_RATIO
    mins = [c for c in completions[1:] if MIN_MIN_FREQ <= c["freq"] < threshold]
    if len(mins) < MIN_MINORITY:
        return None
    return {
        "majority": [{"word": maj["word"], "corpus_freq": maj["freq"], "pmi_general": maj["pmi_general"]}],
        "minority": [{"word": c["word"],  "corpus_freq": c["freq"],  "pmi_general": c["pmi_general"]} for c in mins],
        "top_corpus_completions": completions[:10],
    }


minority_groups: dict[str, dict] = {}
for prefix in sorted(part_a_prefixes):
    grp = extract_minority_group(prefix)
    if grp:
        minority_groups[prefix] = grp

print(f"  From Part A prefixes: {len(minority_groups)} groups")

if len(minority_groups) < 30:
    print("  Expanding to full corpus prefix list...")
    sorted_prefixes = sorted(
        prefix_completions.keys(),
        key=lambda p: prefix_completions[p][0]["freq"] if prefix_completions[p] else 0,
        reverse=True,
    )
    for prefix in sorted_prefixes:
        if prefix in minority_groups:
            continue
        grp = extract_minority_group(prefix)
        if grp:
            minority_groups[prefix] = grp
        if len(minority_groups) >= 80:
            break

minority_groups_capped = dict(list(minority_groups.items())[:80])
print(f"  Final: {len(minority_groups_capped)} groups")
print("\nExample groups:")
for prefix, grp in list(minority_groups_capped.items())[:6]:
    maj = grp["majority"][0]["word"]
    mins = [m["word"] for m in grp["minority"][:3]]
    print(f"  '{prefix}': majority='{maj}', minority={mins}")

partB = {
    "metadata": {
        "description": (
            "Minority class pairs automatically extracted from OpenWebText bigram statistics. "
            "Majority = most frequent single-token completion; minority = other "
            "completions with freq in [5, majority/3). Addresses Roadmap Task 2."
        ),
        "source": f"HuggingFace {OWT_DATASET} (same counts as Part A)",
        "params": {
            "majority_ratio": MAJORITY_RATIO,
            "min_minority_count": MIN_MINORITY,
            "min_minority_freq": MIN_MIN_FREQ,
        },
    },
    "pairs": minority_groups_capped,
}
out = DATASETS_DIR / "minority_class_pairs.json"
with open(out, "w") as f:
    json.dump(partB, f, indent=2)
print(f"\nSaved Part B → {out}")


# ── PubMed PMI (optional) ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PubMed PMI (optional — requires internet + datasets library)")
print("=" * 60)

PUBMED_SAMPLES = 50_000

bio_bigram_counts: Counter = Counter()
bio_unigram_counts: Counter = Counter()
bio_total_bigrams = 0
bio_total_words = 0
PUBMED_AVAILABLE = False

try:
    print(f"Loading up to {PUBMED_SAMPLES:,} PubMed abstracts (streaming)...")
    ds = load_dataset("ccdv/pubmed-summarization", split="train", streaming=True)
    n_loaded = 0
    for example in ds:
        abstract = example.get("abstract", "")
        if not abstract:
            continue
        # Sentence tokenise with simple newline/period split for speed
        sentences = [s.strip() for s in abstract.replace("\n", ". ").split(".") if s.strip()]
        for sent in sentences:
            words = [w.lower() for w in sent.split() if w.isalpha()]
            bio_unigram_counts.update(words)
            bio_total_words += len(words)
            for i in range(len(words) - 1):
                bio_bigram_counts[(words[i], words[i + 1])] += 1
                bio_total_bigrams += 1
        n_loaded += 1
        if n_loaded >= PUBMED_SAMPLES:
            break

    PUBMED_AVAILABLE = True
    print(f"  Loaded {n_loaded:,} abstracts")
    print(f"  Bio unique bigrams: {len(bio_bigram_counts):,}")

except Exception as exc:
    print(f"  PubMed unavailable ({exc.__class__.__name__}: {exc})")
    print("  → Continuing without PMI_bio. Cross-domain analysis will use PMI_general only.")


def pmi_bio(w1: str, w2: str) -> float | None:
    """PMI from PubMed. Returns None if not available or count is zero."""
    if not PUBMED_AVAILABLE:
        return None
    cnt = bio_bigram_counts[(w1, w2)]
    if cnt == 0:
        return None
    p12 = cnt / bio_total_bigrams
    p1  = bio_unigram_counts[w1] / bio_total_words
    p2  = bio_unigram_counts[w2] / bio_total_words
    if p1 == 0 or p2 == 0:
        return None
    return math.log2(p12 / (p1 * p2))


# ── PART C — Cross-Domain Biomedical Pairs ────────────────────────────────────
print("\n" + "=" * 60)
print("Part C: Cross-Domain Biomedical Pairs (from cross_domain_raw.json)")
print("=" * 60)

raw_path = DATASETS_DIR / "cross_domain_raw.json"
if not raw_path.exists():
    raise FileNotFoundError(
        f"Missing {raw_path}\n"
        "Create datasets/cross_domain_raw.json with manually curated pairs first."
    )

with open(raw_path) as f:
    raw = json.load(f)

print(f"Loaded {len(raw['pairs'])} raw groups from {raw_path.name}")
print("Verifying single-token status and attaching corpus PMI...")

cross_domain: list[dict] = []
n_skipped_total = 0

for entry in raw["pairs"]:
    prefix = entry["prefix"]
    gen_lookup  = {c["word"]: c for c in prefix_completions.get(prefix, [])}

    def _build_entries(words: list[str], domain: str) -> tuple[list[dict], int]:
        entries, skipped = [], 0
        for word in words:
            if not is_single_gpt2_token(word):
                print(f"  SKIP {domain:10s} '{prefix} {word}' — '{word}' not single token")
                skipped += 1
                continue
            e: dict = {"word": word, "domain": domain}
            # PMI_general
            if word in gen_lookup:
                e["corpus_freq_general"] = gen_lookup[word]["freq"]
                e["pmi_general"] = gen_lookup[word]["pmi_general"]
            else:
                e["corpus_freq_general"] = 0
                e["pmi_general"] = None
            # PMI_bio
            v = pmi_bio(prefix, word)
            e["pmi_bio"] = round(v, 4) if v is not None else None
            e["corpus_freq_bio"] = int(bio_bigram_counts[(prefix, word)]) if PUBMED_AVAILABLE else None
            entries.append(e)
        return entries, skipped

    gen_entries, sk1 = _build_entries(entry["general"], "general")
    bio_entries, sk2 = _build_entries(entry["biomedical"], "biomedical")
    n_skipped_total += sk1 + sk2

    cross_domain.append(
        {
            "prefix":    prefix,
            "ambiguity": entry["ambiguity"],
            "note":      entry["note"],
            "general":   gen_entries,
            "biomedical": bio_entries,
        }
    )
    print(
        f"  {prefix:12s}: {len(gen_entries)} general, {len(bio_entries)} biomedical"
        + (f"  (PMI_bio available)" if PUBMED_AVAILABLE else "  (PMI_bio=None)")
    )

print(f"\n{len(cross_domain)} groups built  ({n_skipped_total} words skipped — not single token)")

partC = {
    "metadata": {
        "description": (
            "Cross-domain pairs: ambiguous prefix words with general vs biomedical completions. "
            "PMI_general from OpenWebText; PMI_bio from PubMed (if available). "
            "Enables three-way comparison: PMI_general vs PMI_bio vs P_model."
        ),
        "source_raw":     "datasets/cross_domain_raw.json (manual curation)",
        "source_general_pmi": f"HuggingFace {OWT_DATASET} (first {OWT_MAX_DOCS:,} docs)",
        "source_bio_pmi": "ccdv/pubmed-summarization (HuggingFace)" if PUBMED_AVAILABLE else "unavailable",
        "pubmed_samples": PUBMED_SAMPLES if PUBMED_AVAILABLE else None,
        "rationale": (
            "GPT-2 is trained on general web text. Biomedical completions are valid "
            "but underrepresented. This dataset enables quantifying domain bias "
            "separately from corpus frequency effects — directly relevant to AI4Health "
            "applications (clinical NLP, drug discovery, literature mining)."
        ),
    },
    "pairs": cross_domain,
}
out = DATASETS_DIR / "cross_domain_pairs.json"
with open(out, "w") as f:
    json.dump(partC, f, indent=2)
print(f"Saved Part C → {out}")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Dataset Building Complete")
print("=" * 60)
print(f"  Part A: high_pmi_bigrams.json       — {len(selected)} bigrams")
print(f"  Part B: minority_class_pairs.json   — {len(minority_groups_capped)} groups")
print(f"  Part C: cross_domain_pairs.json     — {len(cross_domain)} groups")
print(f"  PubMed PMI_bio:  {'available' if PUBMED_AVAILABLE else 'NOT available (see warning above)'}")
print("\nNext: python src/pmi_prediction_exp.py")
