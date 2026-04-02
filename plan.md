# PR Implementation Plan: "Where is Washing Machine Stored in LLMs?"

## Context & Strategic Goal

This PR is part of an application to work with Prof. Chenhao Tan and Prof. Ari Holtzman at UChicago's CHAI Lab, submitted via fossick.ai. The application requires choosing an Active Project, completing a Roadmap item, submitting a PR, and writing a 300-word summary of learnings and proposed next steps.

The applicant (Mingtao Zhang) is a Zhejiang University undergraduate with dual majors in Pharmaceutical Science & AI, with strong PyTorch experience but no prior TransformerLens experience. Available hardware: 2x Titan XP (12GB VRAM each), CPU-only fallback fully supported. GPT-2 (124M params) requires <2GB VRAM.

**Competing PRs already submitted (2 total):**
- PR 1: Extends original 21 compounds with minority candidates, adds layerwise cosine similarity × association analysis (R²=0.39 at layer 0, p<0.001) and SAE feature analysis, plus curated sentence validation. Targets Roadmap Task 2. High quality.
- PR 2: Scales original experiments to Wikipedia dataset, adds PMI gap analysis (finding: near-chance accuracy). Partially targets Task 1. Medium quality.

**Our differentiation:** (a) logit lens cross-layer tracking — neither PR did this; (b) over-memorization quantification — novel metric; (c) biomedical three-way PMI comparison (PMI_general vs PMI_bio vs P_model) — AI4Science framing, completely novel; (d) Task 3 hidden knowledge analysis — unoccupied territory.

---

## Roadmap Tasks Targeted

From the GitHub issue:
1. **Task 1** ← our primary target: "Try to see if, in general, models know the 2nd token in high PMI 2-token phrases."
2. **Task 2** ← our primary target: "Specifically look and see if the minority class (e.g. 'washing method' instead of 'washing machine') is where LLMs fail to predict."
3. **Task 3** ← secondary target (unoccupied): "See if there are cases where the LLM consistently correctly predicts a 2-token phrase, but where we can't predict this ahead of time."
4. **Task 4** ← partial coverage: "See how many 2-token phrases we can find and then regroup."

---

## Dataset Construction

### Part A — High-PMI Bigrams (150–200 items)

**Source**: HuggingFace **WikiText-103** (`wikitext-103-raw-v1`, train split, streamed). NLTK `sent_tokenize` + `word_tokenize`; **sentence-internal** alpha bigrams only (no cross-sentence pairs).

**Pipeline**:
1. Compute PMI for all alpha bigrams with freq ≥ 10
2. Filter: second word must tokenise as a single GPT-2 token
3. Add NLTK POS tags to every bigram (NN-NN, JJ-NN, VBG-NN, etc.)
4. Tier-sample across PMI range: top-100 high (≥10), top-60 medium (6–10), top-40 low (<6)
5. Cap at 200 items

Output: `datasets/high_pmi_bigrams.json`

### Part B — Minority Class Pairs (target: 60–80 groups)

**Source**: Automatically derived from corpus + Part A prefixes.

**Pipeline** (no hand-coded annotations):
1. For each prefix word in Part A, collect all single-token completions (freq ≥ 5)
2. A group qualifies when: top completion freq ≥ 3× any minority completion freq AND ≥2 minority completions exist
3. If Part A prefixes yield < 30 groups, expand search to full corpus (cap at 80)

Output: `datasets/minority_class_pairs.json`

### Part C — Cross-Domain Biomedical Pairs (target: 30–35 groups)

**Source**: `datasets/cross_domain_raw.json` (manually curated 35 groups) + PMI from two corpora.

**Why this matters (AI4Science / AI4Health framing)**:
GPT-2-family models are increasingly used in biomedical NLP (clinical note generation, drug discovery, literature mining). If next-token prediction systematically favours general-domain completions over biomedical ones, this introduces hidden errors in all such applications — directly relevant to Prof. Tan's work at Abridge (AI clinical notes) and radiology AI.

**Pipeline**:
1. Load `cross_domain_raw.json` (35 groups, each with general + biomedical candidate second words)
2. Verify single-token status for all candidate words
3. Look up PMI_general from WikiText-103 statistics (already computed in Part A pipeline)
4. Compute PMI_bio from PubMed abstracts:
   - Load via HuggingFace `datasets`: `ccdv/pubmed-summarization` (streaming, ~50k abstracts)
   - Compute sentence-level bigram counts and PMI on PubMed text
   - If PubMed unavailable: skip PMI_bio, log warning, continue with remaining analyses
5. Each candidate word stored with: {word, domain, corpus_freq_general, pmi_general, corpus_freq_bio, pmi_bio}

Output: `datasets/cross_domain_pairs.json`

---

## Experiments

### Experiment 1 — Task 1: Systematic PMI Prediction with Logit Lens

**Purpose**: Answer "do models know the 2nd token in high-PMI bigrams?" with mechanistic depth.

**Multi-template priming** (6 templates, average for robustness):
```
"The {w1}"  |  "A {w1}"  |  "This {w1}"
"One {w1}"  |  "Her {w1}"  |  "My {w1}"
```
Priming ratio and rank averaged across all 6 templates per bigram.

**Logit lens** (novel — neither competing PR did this):
- Run `model.run_with_cache("The {w1}")` once per bigram
- At each layer l: apply `model.ln_final` + `model.unembed` to `cache["blocks.{l}.hook_resid_post"]`
- Track P(word2_token) across all 12 layers
- **Commitment layer**: earliest layer where P(word2) > 10× baseline P(word2)
- Expected finding: high-PMI compounds commit early; low-PMI compounds never commit

**Key metrics**:
- Top-1/5/10 accuracy overall and by PMI tier
- Spearman(PMI, priming_ratio)
- Median commitment layer by PMI tier
- Top-1 accuracy broken down by POS pattern (NN-NN vs JJ-NN vs VBG-NN)

Output: `results/pmi_prediction_results.json`

---

### Experiment 2 — Task 2: Minority Class + Over-Memorization Index

**Purpose**: Quantify minority prediction failure and determine whether it exceeds what corpus statistics predict.

**Multi-template priming** (same 6 templates) applied to each (prefix, majority_word) and (prefix, minority_word) pair.

**Over-Memorization Index (OMI)** — novel metric:
```
OMI = (P_model(majority|prefix) / P_model(minority|prefix))
       / (corpus_freq(prefix, majority) / corpus_freq(prefix, minority))
```
- OMI > 1: model's majority bias exceeds what corpus frequency predicts → over-memorisation
- OMI ≈ 1: model simply mirrors corpus statistics
- OMI < 1: model is less biased than corpus (surprising)

**Logit lens competition curves** (novel):
- For each group: track P(majority_token) and P(minority_token) at every layer simultaneously
- Find crossover layer (if any) where curves diverge
- Aggregate: is majority consistently dominant from layer 0, or does it pull ahead mid-network?

**PMI-gap threshold analysis** (extends PR 2's finding):
- For each minority group: PMI_gap = PMI(majority) - PMI(minority)
- Binary outcome: model_correct = P_model(majority) > P_model(minority)
- Plot accuracy vs PMI_gap; find threshold where accuracy > 75%

**Statistics**:
- Wilcoxon signed-rank: majority vs minority model probabilities
- OMI distribution (mean, median, % with OMI > 1)
- Spearman(PMI_gap, model_correct)

Output: `results/minority_class_results.json`

---

### Experiment 3 — AI4Science: Biomedical Three-Way PMI Analysis

**Purpose**: Quantify whether GPT-2's domain bias is explained by corpus frequency alone or reflects additional learned bias.

**Three-way comparison** for each (prefix, word2) pair:
- PMI_general: from WikiText-103 (what general-domain Wikipedia-style corpus predicts)
- PMI_bio: from PubMed (what biomedical corpus predicts)
- P_model: GPT-2's predicted probability

**Domain alignment score**:
- Spearman(P_model, PMI_general) across all words in the dataset → align_general
- Spearman(P_model, PMI_bio) across all words in the dataset → align_bio
- Interpretation: if align_general >> align_bio → model is general-domain aligned

**Minority comparison** (connects Experiments 2 and 3):
- Compute OMI for biomedical pairs (using PMI_bio as corpus reference)
- Compare OMI_biomedical vs OMI_general_minority
- Question: Is biomedical failure larger than ordinary minority failure, and by how much?

**Statistics**:
- Mann-Whitney U: general vs biomedical P_model (max per group)
- Domain alignment ratio: align_general / align_bio
- % groups where biomedical completions are never top-50 predictions

Output: `results/cross_domain_results.json`

---

### Experiment 4 — Task 3: Hidden Knowledge Analysis

**Purpose**: Answer "are there cases where the model correctly predicts a bigram that we can't predict ahead of time?"

**Operationalisation** via four-quadrant scatter:
```
              PMI_corpus
              Low (<6)   |  High (≥8)
             ────────────┼────────────
priming  High | HIDDEN   |  EXPECTED
(≥50x)       | KNOWLEDGE |  SUCCESS
             ────────────┼────────────
priming  Low  | EXPECTED  |  PR1 FOCUS
(<10x)        | UNKNOWN   |  (FAILURE)
```

**"Hidden knowledge" bigrams** (low PMI, high priming):
- Hypothesis: these are bigrams common in web text (GPT-2's training data) but rare in WikiText-103 (or otherwise underrepresented in that PMI estimate)
- Evidence: their top-10 predictions from GPT-2 will be more specific/modern vocabulary
- Analysis: list top-10 hidden knowledge cases, characterise by POS pattern and semantic category

**Statistics**:
- Count and % of bigrams in each quadrant
- POS pattern distribution within hidden knowledge quadrant vs expected success

Output: `results/hidden_knowledge_results.json`

---

## File Structure

```
src/
├── experiments.py              # EXISTING — do not modify
├── manual_sae.py               # EXISTING — do not modify
├── pmi_dataset_builder.py      # NEW: builds Parts A, B, C (PubMed + POS + auto-minority)
├── pmi_prediction_exp.py       # NEW: all 4 experiments
└── analyze_pmi_results.py      # NEW: all visualisations + statistical output

datasets/
├── compound_concepts/          # EXISTING
├── cross_domain_raw.json       # NEW: curated 35 biomedical groups (source for Part C)
├── high_pmi_bigrams.json       # NEW: Part A
├── minority_class_pairs.json   # NEW: Part B
└── cross_domain_pairs.json     # NEW: Part C (with PMI_general + PMI_bio)

results/
├── pmi_prediction_results.json     # Exp 1
├── minority_class_results.json     # Exp 2
├── cross_domain_results.json       # Exp 3
├── hidden_knowledge_results.json   # Exp 4
└── plots/
    ├── pmi_vs_priming.png
    ├── accuracy_by_pmi_bucket.png
    ├── logit_lens_commitment.png
    ├── majority_vs_minority.png
    ├── over_memorization_index.png
    ├── logit_lens_competition.png
    ├── pmi_gap_threshold.png
    ├── cross_domain_comparison.png
    ├── three_way_alignment.png
    ├── four_quadrant_task3.png
    └── pos_pattern_accuracy.png
```

---

## Running Order

```bash
# Step 1: Build all datasets (~10-20 min, CPU only, requires internet for PubMed)
python src/pmi_dataset_builder.py

# Step 2: Run all 4 experiments (~20-60 min on CPU, ~5 min on GPU)
python src/pmi_prediction_exp.py

# Step 3: Generate all plots and statistical summary
python src/analyze_pmi_results.py
```

---

## Key Principles

### Must Do
1. Follow existing code style (match experiments.py conventions)
2. All results reproducible: SEED=42, recorded parameters
3. Honest reporting — if results are not interesting, analyse why (this is itself research value)
4. Every piece of work traces back to a Roadmap Task
5. Understand every line — professor may ask in interview

### Must Not Do
1. Modify existing `experiments.py` or `manual_sae.py`
2. Fabricate results
3. Let biomedical angle dominate — it is ~15% of dataset, a compelling extension, not the main story
4. Over-engineer beyond what the task requires

### Differentiation vs Competing PRs
- PR 1 used the original 21 hand-picked compounds; we use 200 corpus-derived bigrams → scale + reproducibility
- PR 1 did cosine similarity × association; we add logit lens and OMI → additional mechanistic depth
- PR 2 scaled original experiments; we add biomedical three-way and Task 3 → new territory
- Logit lens, OMI, three-way PMI, Task 3 hidden knowledge: all unoccupied

---

## 300-Word Summary Framework

**What you did**: Built a 200-item corpus-derived high-PMI bigram dataset (PMI from WikiText-103), automatically extracted 60–80 minority class groups, and ran four experiments on GPT-2 Small using logit lens, over-memorization quantification, and biomedical domain alignment analysis.

**What you learned** (fill with actual results):
- Task 1: overall Top-1 accuracy for high-PMI bigrams is X%; Spearman(PMI, priming) = r; commitment layer is earlier for high-PMI bigrams (median layer Y vs Z for low-PMI).
- Task 2: majority completion probability exceeds minority by Nx (Wilcoxon p=); OMI median = M, indicating model's majority bias is M× larger than corpus frequency alone predicts. Logit lens shows majority pulls ahead at layer L on average.
- Biomedical: model alignment with PMI_general (r=A) is X× higher than with PMI_bio (r=B), confirming systematic domain bias not explained by corpus frequency.
- Task 3: K% of bigrams fall in the "hidden knowledge" quadrant (low corpus PMI, high model priming) — these are predominantly VBG-NN patterns, suggesting the model has learned modern compound conventions not well-represented in WikiText-103 (relative to web-scale training data).

**Concrete next step**: The OMI finding suggests a causal intervention: ablating the prefix token's attention influence on the head-noun position should reduce OMI for high-OMI groups more than low-OMI groups. Testing this with TransformerLens activation patching would directly test whether majority priming is mediated by specific attention heads.
