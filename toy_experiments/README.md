# Toy Experiments: Low-PMI Bigram Minority Class Analysis

This directory contains reproducible experiments investigating whether LLMs fail specifically on low-PMI compositional variants (Roadmap Item #2). We use GPT-2 models to study how high-frequency biases (e.g., "washing machine") interfere with minority-class predictions (e.g., "washing method").

## Overview

Three complementary experiments:
1. **Dataset construction**: Extract low-PMI bigrams from OpenWebText
2. **Context bias correction**: Test GPT-2 models on adversarial low-PMI contexts
3. **Internal analysis**: Inspect layer-wise probability evolution in GPT-2-XL

## Scripts

### 1. `openwebtext_bigram_extract.py`
**Purpose**: Identify low-PMI minority classes by scanning OpenWebText for "washing *" bigrams.

**Input**: 
- OpenWebText train split (first 1M documents)
- Loaded via Hugging Face `datasets` library

**Output**: 
- `washing_only_bigrams.json`: Frequency-ranked list of all tokens following "washing"

**Usage**:
```bash
python openwebtext_bigram_extract.py
```

**Key insight**: "washing machine" dominates with extremely high frequency; alternatives like "washing method", "washing liquid", "washing line" are long-tail (low-PMI).

---

### 2. `gpt2.py`
**Purpose**: Quantify how GPT-2 models default to high-PMI predictions ("machine") even when given strong contradictory context.

**Input**: 
- `test_cases.json`: 15 manually crafted low-PMI test cases
- GPT-2 models: small (124M), medium (355M), large (774M), xl (1.5B)

**Output**: 
- `gpt2_washing_detailed_results.csv`: Per-case metrics for all 4 models (60 rows)
- `gpt2_washing_summary.csv`: Aggregate statistics by model size (4 rows)
- `washing_experiment_analysis.png`: 4-panel visualization (bias rate, confidence gap, JSD, accuracy)

**Usage**:
```bash
python gpt2.py
```

**Metrics**:
| Metric | Interpretation |
|--------|-----------------|
| `Is_Machine` | Frequency of greedy decoder choosing "machine" (bias indicator) |
| `P_Machine` | Average probability assigned to " machine" across test cases |
| `JSD` | Jensen-Shannon divergence from baseline "The washing" distribution; higher = more context-sensitive |
| `Confidence_Gap` | p_target - p_machine; positive = target wins over machine |
| `Correct` | Fraction of test cases where target token is correct |

**Key findings**:
- Small model (gpt2): 46.7% bias rate, -0.031 confidence gap
- Large models (gpt2-medium/large/xl): 13-26% bias rate, +0.16 to +0.22 confidence gap
- JSD increases with model size (0.583 → 0.676), indicating larger models are more context-sensitive
- Best accuracy is 53.3%, showing persistent compositional challenges even for large models

---

### 3. `logit_lens.py`
**Purpose**: Inspect internal layer-wise probability evolution to understand how GPT-2-XL corrects or fails to correct high-PMI biases.

**Input**: 
- `test_cases.json`: Same 15 low-PMI test cases
- GPT-2-XL model (1.5B parameters, 12 layers)

**Output**: 
- `logit_lens_results/case_1.png` through `case_15.png`: Individual plots for each test case
- `logit_lens_results/summary_5x3_all_cases.png`: Grid visualization of all 15 cases

**Usage**:
```bash
python logit_lens.py
```

**Method**:
1. Feed each test case through GPT-2-XL
2. Extract hidden state from every transformer layer (0-11)
3. Apply layer norm and unembedding to each hidden state
4. Track probability evolution for: target token, " machine" (high-PMI distractor), and competing prediction artifacts
5. Plot layer-wise curves for visual inspection

**Key findings**:
- **Selective suppression**: ~50% of cases show "machine" spiking in early layers (0-4), then actively suppressed by later layers (6-11)
- **Tokenization artifacts**: Early layers often show spurious predictions (e.g., "ton" from Washington contexts), suppressed by later layers
- **Reinforcement pattern**: When suppression succeeds, target token probability increases monotonically through layers (not sudden jumps)
- **Variable success**: Some cases (e.g., "rituals") dominate from layer 0; others show intense battles between machine and target

**Interpretation**: 
The model lacks a unified error-correction mechanism. Instead, it uses conditional suppression of high-PMI distractors combined with layer-wise reinforcement of target tokens. Smaller models lack sufficient depth to resolve conflicts; larger models have more layers to accumulate evidence.

---

## Input Files

### `test_cases.json`
Contains 15 manually crafted low-PMI test cases with ground-truth target tokens:
```json
{
  "test_cases": [
    {
      "id": 1,
      "input_text": "I hate using washing powder, so instead I bought a huge jug of washing",
      "target_token": "liquid"
    },
    ...
  ]
}
```

Each case is designed so context strongly implies a non-machine target, directly contradicting high-PMI baseline.

---

## Data Flow

```
openwebtext_bigram_extract.py
    ↓
washing_only_bigrams.json
    ↓
[Manual construction of test_cases.json with low-PMI contexts]
    ↓
gpt2.py                                logit_lens.py
    ↓ (both use test_cases.json)              ↓
gpt2_washing_detailed_results.csv       logit_lens_results/ (15 individual plots)
gpt2_washing_summary.csv                summary_5x3_all_cases.png
washing_experiment_analysis.png
```

---

## Results Summary

### Context Bias Results (Experiment 2)

| Model | Bias Rate | JSD | Accuracy |
|-------|-----------|-----|----------|
| gpt2 | 46.7% | 0.583 | 33.3% |
| gpt2-medium | 26.7% | 0.653 | 53.3% |
| gpt2-large | 13.3% | 0.672 | 46.7% |
| gpt2-xl | 20.0% | 0.676 | 53.3% |

### Layer-wise Insights (Experiment 3)

- **50% of cases**: Machine prediction peaks in layers 0-4, suppressed by layers 6-11
- **Artifact pollution**: Early layers dominated by tokenization artifacts (e.g., "ton")
- **Target wins**: Layer-wise monotonic increase when successful (reinforcement)