# Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs
**Feucht et al., 2024 (arXiv:2406.20086v3)**

## Paper Overview
This paper directly investigates how LLMs convert arbitrary groups of subword tokens into meaningful representations - a critical question for understanding compound concept storage like "washing machine". The authors discover a "token erasure" phenomenon where token-level information is rapidly forgotten in early layers for multi-token words and entities.

---

## 1. What is "Token Erasure"? How is it measured?

### Definition
**Token erasure** is a phenomenon where hidden states at the last token position of multi-token words and entities "forget" information about previous and current tokens in early layers of the model.

### Key Characteristics
- Occurs specifically at the **last token position** of multi-token sequences
- Information about preceding tokens (i=-1, i=-2) and current token (i=0) is rapidly lost
- Happens in **early layers** (most pronounced by layer 9)
- Does NOT occur for:
  - First or middle tokens of multi-token entities
  - Other non-entity tokens
  - Single-token words

### Measurement Method: Linear Probes
The authors trained linear probes to predict token values from hidden states:

**Probe notation:** p_i^(ℓ) : R^d → R^|V|
- Takes hidden state h_t^(ℓ) at layer ℓ and position t
- Predicts nearby token at position t+i
- Trained for all layers 0 ≤ ℓ < 32
- Offsets: i ∈ {-3, -2, -1, 0, 1}
  - i=-1: previous token
  - i=-2: two tokens back
  - i=0: current token
  - i=1: next token

**Training details:**
- Dataset: 428k tokens from The Pile
- Optimizer: AdamW, 16 epochs, batch size 4, learning rate 0.1
- Hardware: 6-8 hours on RTX-A6000

### Erasure Effect Patterns

**For COUNTERFACT subjects (e.g., "Star Wars"):**
- Last token ("Wars"): accuracy for i=-1 drops from ~100% (layer 0) to ~20% (layer 9)
- Current token (i=0) also drops significantly
- Other tokens in dataset: maintain ~60-80% accuracy throughout

**For multi-token words (e.g., "northeastern" = [_n, ort, he, astern]):**
- Same erasure pattern observed at last token position
- Previous token accuracy: ~100% → ~20% by layer 9
- Current token accuracy also drops dramatically

**For named entities (identified by spaCy):**
- Consistent erasure effect across Wikipedia data
- Not an artifact of COUNTERFACT templates

### Critical Insight
"Erasure" doesn't mean information is lost - it's being **transformed** from token-level to lexical-level representation. The authors interpret this as a footprint of detokenization.

---

## 2. The "Implicit Vocabulary" Concept

### Definition
The **implicit vocabulary** refers to the set of semantically meaningful units that an LLM learns to represent, beyond its explicit token vocabulary. These are lexical items that "function as single units of meaning."

### Types of Implicit Vocabulary Items
1. **Multi-token words**: "northeastern" → [_n, ort, he, astern]
2. **Named entities**: "Neil Young", "Eiffel Tower", "Star Wars"
3. **Multi-word expressions**: "break a leg" (idiomatic)
4. **Compound words**: Words split by BPE in non-semantic ways
5. **LaTeX commands**: e.g., formulas and code snippets
6. **Domain-specific terms**: "research.gate", "Bloom.ington"

### Key Properties: Non-Compositionality
Lexical items are **non-compositional** - their meaning cannot be predicted from constituent tokens:
- "break a leg" ≠ "break" + "leg"
- "patrolling" (tokens: pat + rolling) ≠ "pat" + "rolling"
- "northeastern" ([_n, ort, he, astern]) has no relation to "north" or "east" at token level

### Why an Implicit Vocabulary is Necessary
The explicit token vocabulary (determined by BPE) creates arbitrary, semantically meaningless splits:
- "northeastern" → [_n, ort, he, astern] (Llama-2-7b)
- "Hawaii" → [_Hawai, i] (capitalized) vs [_ha, w, ai, i] (lowercase)

Despite these challenges, LLMs perform well, suggesting they develop internal representations that overcome tokenization artifacts.

### Storage Requirement
This arbitrariness necessitates a storage system (Murphy, 2010) - either explicit or implicit - to map token sequences to meanings.

---

## 3. How Multi-Token Concepts Get Assembled Across Layers

### The Assembly Process: "Detokenization"

**Early Layers (Layers 0-9): Subject Enrichment & Token Erasure**
- Information about multi-token entities/words is collected at the **last token position**
- Token-level information is "erased" (transformed) as lexical representation forms
- By layer 9: token erasure effect is most pronounced

**Why Last Token Position?**
Autoregressive constraint: Models cannot enrich early tokens until later tokens are seen
- Example: "_Space" could be "Space Jam", "Space Station", or "Space Needle"
- Only after seeing "Needle" can the model represent "The Space Needle"
- Exception: High-frequency entities with strong context cues (e.g., "_The, _E, iff" already suggests "The Eiffel Tower")

### Layer-by-Layer Pattern

**Layer 0-1 (Embedding):**
- High accuracy for all token predictions (80-100%)
- Token-level information fully preserved

**Layers 1-9 (Subject Enrichment):**
- Last token positions show rapid accuracy drop for i=-1, i=-2, i=0
- First/middle tokens maintain accuracy
- "Erasure" effect emerges and peaks

**Layers 10-32 (Later Processing):**
- Token-level information remains low for last tokens
- Next-token prediction (i=1) gradually improves
- Model shifts focus to generating subsequent tokens

### Evidence from Different Token Positions (Figure 13)

**First subject tokens:** No erasure effect - maintain previous token info
**Middle subject tokens:** No erasure effect
**Last subject tokens:** Strong erasure effect

This position-specific pattern strongly suggests a deliberate mechanism for lexical assembly.

### Subject Length Dependency (Figure 14)

**Unigrams:** Previous token accuracy remains ~100% (similar to non-subjects)
**Bigrams & Trigrams:** Clear erasure pattern emerges
- Suggests mechanism activates only for multi-token sequences

---

## 4. Models and Methodology

### Models Studied
1. **Llama-2-7b** (Touvron et al., 2023)
   - Vocabulary size: 32k tokens
   - 32 layers
   - Hidden dimension: d (not specified, typically 4096)

2. **Llama-3-8b** (Meta, 2024)
   - Vocabulary size: 128k tokens (4x larger)
   - 32 layers
   - Shows similar erasure patterns despite different tokenization

### Datasets

**Training (for probes):**
- The Pile: 428k tokens (random sample)
- Validation: 279k tokens (separate sample)

**Testing:**
1. **COUNTERFACT** (Meng et al., 2022)
   - Factual prompts: "Mount Passel is in Antarctica"
   - Filtered to correctly-answered: 5,063 (Llama-2-7b), 5,495 (Llama-3-8b)
   - Augmented with Wikidata [album/movie/series → creator] pairs
   - Total: 12,135 (Llama-2-7b), 13,995 (Llama-3-8b)

2. **Wikipedia**
   - 500 articles (~256k tokens) from 20220301.en dump
   - Words identified by whitespace splitting
   - Named entities identified by spaCy NER (excluding number-based classes)

3. **The Pile**
   - 500 documents for vocabulary extraction
   - Separate test set (273k tokens) for in-distribution validation

### Methodology Details

**Linear Probe Training:**
- Architecture: Simple linear layer R^d → R^|V|
- Loss: Cross-entropy (predicting token ID)
- Hyperparameter selection: Random sweep on validation set

**Avoiding Training Imbalances (Appendix B):**
- Verified that entity tokens appear at similar frequencies as non-entity tokens in training data
- Median occurrences for both: 0 (most test n-grams not in training)
- Removing frequent non-subject sequences doesn't change results
- Conclusion: Erasure effect is NOT due to probe training imbalance

**Implementation:**
- Used nnsight library (Fiotto-Kaufman et al., 2024)
- Experiments run on Center for AI Safety Compute Cluster
- Code/data: footprints.baulab.info

---

## 5. Key Results: At Which Layers Does Erasure Happen? What Does This Mean?

### Primary Finding: Early Layer Erasure (Layers 1-9)

**Llama-2-7b Results (Figure 2, 3):**
- **Last token of COUNTERFACT subjects:**
  - Previous token (i=-1): 100% (layer 0) → 20% (layer 9) → stable ~20%
  - Current token (i=0): 100% (layer 0) → 20% (layer 9) → stable ~20%
  - Two tokens back (i=-2): Similar drop

- **Multi-token words (Wikipedia):**
  - Same erasure pattern
  - Last token of "intermittent" forgets "inter", "mitt", "ent"

- **Other tokens (non-last-position):**
  - Maintain 60-80% accuracy throughout all layers
  - No erasure effect

**Llama-3-8b Results (Figures 9, 10, 11):**
- **Same erasure pattern** despite 4x larger vocabulary
- Suggests phenomenon is fundamental to architecture, not tokenization scheme
- Last tokens of multi-token words and entities show identical erasure
- Different implicit vocabulary content (more multi-word expressions, less multi-token words)

### Layer Choice: L=9
Authors chose layer 9 for their erasure score based on:
- Maximum drop in token-level information
- Consistent across both models
- Ablation study (Table 3) shows L=9 gives best recall

### What Does Erasure Mean? Interpretation

**Not information loss, but transformation:**
1. Token-level encoding → Lexical-level encoding
2. Information about "Star" and "Wars" as separate tokens → Information about "Star Wars" as a concept
3. Enables downstream processing to work with meaningful units

**Connection to "Subject Enrichment" (Geva et al., 2023):**
- Factual knowledge about entities concentrates at last token
- "The Space Needle" → factual info (location: Seattle) stored at "le"
- Erasure may be the mechanism that enables enrichment

**Comparison to In-Distribution Tokens (Figure 4):**
- On regular Pile tokens: smoother, gradual forgetting trajectory
- Suggests entity processing is distinct from general token processing

**Retokenization vs. Detokenization:**
- Elhage et al. (2022) found late-layer neurons for retokenization (internal → output tokens)
- Current work shows early-layer detokenization (input tokens → internal lexical units)
- Symmetric processes at different stages

---

## 6. Relation to Compound Noun Processing (like "washing machine")

### Direct Relevance to Your Research

**"Washing machine" is likely an implicit vocabulary item:**
1. Multi-word expression (2 tokens in most tokenizers)
2. Non-compositional meaning (not just "washing" + "machine")
3. Specific semantic concept (household appliance)

**Predicted erasure behavior:**
- If "washing machine" is treated as lexical unit:
  - Last token ("machine") should show token erasure
  - Hidden state at "machine" would forget "washing" token-level info
  - Hidden state would represent compound concept "washing machine"

### Testing Hypotheses for "Washing Machine"

**High erasure score would suggest:**
- Model treats "washing machine" as single semantic unit
- Compound concept is stored/assembled in early layers
- Processing differs from compositional phrases like "red machine"

**Low erasure score would suggest:**
- Model processes "washing" and "machine" compositionally
- Less likely to have unified concept representation
- May still combine meanings, but through different mechanism

### Comparison Points

**Similar to named entities:**
- "Neil Young" - proper noun, arbitrary token split
- "Washing machine" - common compound noun, arbitrary word split
- Both require non-compositional semantic storage

**Differences from examples in paper:**
- Paper focuses on: single multi-token words, named entities
- "Washing machine": multi-word compound
- But principle is same: non-compositional lexical item

### Implications for Compound Noun Research

**If washing machine shows erasure:**
1. Supports hypothesis of unified representation
2. Last token position is where concept "lives"
3. Intervening on "machine" token may affect entire compound
4. Word order matters (autoregressive constraint)

**If washing machine doesn't show erasure:**
1. May use different assembly mechanism
2. Could be more compositional than expected
3. Representation may be distributed across both tokens

---

## 7. Code/Tools/Datasets Released

### Released Resources
**Repository:** footprints.baulab.info
- Code for linear probe training
- Token erasure scoring algorithm (Algorithm 1)
- Implicit vocabulary extraction pipeline

### Key Tools/Libraries Used

**nnsight library** (Fiotto-Kaufman et al., 2024)
- For accessing LLM internals
- Hook-based intervention framework
- Citation: arXiv:2407.14561

**spaCy**
- Named Entity Recognition pipeline
- Used to identify multi-token entities

### Datasets

**COUNTERFACT** (Meng et al., 2022)
- Factual knowledge prompts
- Available from original ROME paper

**Wikidata** (Vrandečić and Krötzsch, 2014)
- [album/movie/series → creator] pairs
- Augmented test set

**The Pile** (Gao et al., 2020)
- 800GB diverse text dataset
- Used for probe training and vocabulary extraction

**Wikipedia Dump**
- 20220301.en split
- 500 articles for analysis

### Reproducibility

**Linear Probes:**
- All hyperparameters specified
- Training time: 6-8 hours per probe (170 probes total)
- Hardware: RTX-A6000

**Erasure Score Computation:**
- Layer L=9 (with ablation for L ∈ {5,9,13,17,21})
- Offsets: i ∈ {-2, -1}
- Equation 1 fully specified

---

## 8. Specific Experimental Methodology We Could Adapt

### Method 1: Linear Probe Analysis for "Washing Machine"

**Adapted Procedure:**
1. **Collect dataset of "washing machine" contexts**
   - Extract sentences containing "washing machine" from Wikipedia/Common Crawl
   - Include contexts where it's clearly non-compositional
   - Control group: "red machine", "broken machine" (compositional)

2. **Train position-specific probes**
   - Train probes on general text (as in paper)
   - Test on "washing machine" instances
   - Measure: Can probe recover "washing" from "machine" hidden state?

3. **Compare across layers**
   - Plot accuracy curves like Figure 2
   - Look for erasure pattern in "washing machine"
   - Compare to compositional controls

**Expected Results:**
- Strong erasure → "washing machine" is lexical unit
- No erasure → compositional processing
- Intermediate → partial lexicalization

### Method 2: Erasure Score for Compound Nouns

**Adapted Algorithm 1:**
1. **Segment documents containing compound nouns**
   - Run Algorithm 1 on documents with known compounds
   - Calculate ψ score for "washing machine"
   - Compare to other compound nouns: "coffee machine", "fire truck", "hot dog"

2. **Build compound noun implicit vocabulary**
   - Extract high-scoring multi-word sequences
   - Filter for noun compounds specifically
   - Compare Llama-2-7b vs Llama-3-8b (different tokenizations)

3. **Ranking by lexicality**
   - High ψ: "hot dog" (idiom), "washing machine" (compound)
   - Low ψ: "red car" (compositional)
   - Validates which compounds are stored as units

**Advantages:**
- No training required (uses pre-trained probes if available)
- Can be run on large corpus
- Provides quantitative "lexicality" measure

### Method 3: Layer-by-Layer Intervention Study

**New contribution (not in paper):**
1. **Intervention at different layers**
   - At layer 5: Replace "washing" hidden state with "coffee"
   - At layer 15: Same intervention
   - Question: When is it "too late" to change compound meaning?

2. **Causal tracing for compounds**
   - Adapt Meng et al. (2022) causal tracing to compounds
   - Where is "washing machine" → "appliance" association stored?
   - Compare to "washing" → "cleaning" association

3. **Critical layer identification**
   - If erasure happens at layer 9, intervening after layer 9 shouldn't work
   - Test window: layers 1-15
   - Map when compound becomes "fixed"

### Method 4: Cross-Model Comparison

**Exploit different tokenizations:**
1. **Llama-2-7b vs Llama-3-8b**
   - "washing machine" may have different token splits
   - Check if erasure happens regardless of tokenization
   - Tests if phenomenon is semantic (compound meaning) vs syntactic (token pattern)

2. **Other model families**
   - GPT-2, Pythia models (different sizes)
   - Check if erasure correlates with model capability
   - Smaller models: less erasure? Different layers?

### Method 5: Compositional vs Non-Compositional Control

**Critical experiment for "washing machine":**

**Dataset construction:**
- **Non-compositional:** "washing machine" (appliance), "washing powder" (detergent)
- **Compositional:** "washing bear", "washing process"
- **Ambiguous:** "washing line" (could be literal or idiomatic)

**Prediction:**
- Non-compositional → high erasure scores
- Compositional → low erasure scores
- Correlation = validation of method

**Measurements:**
- Erasure score ψ
- Probe accuracy at layer 9
- Contextual variation (does context affect erasure?)

### Practical Implementation Steps

**Step 1: Reproduce paper's probes**
- Train probes on Llama-2-7b using their setup
- Validate on COUNTERFACT dataset
- Confirm we see same erasure pattern

**Step 2: Test on compound nouns**
- Create compound noun test set (100-200 instances)
- Run probe analysis (like Figure 2)
- Calculate erasure scores (Algorithm 1)

**Step 3: Build compound vocabulary**
- Run on large corpus (Wikipedia subset)
- Extract high-scoring sequences
- Analyze: What compounds are "lexical" to the model?

**Step 4: Validate with interventions**
- Patch hidden states at different layers
- Test causal necessity of erasure for compound understanding
- Compare to paper's findings on named entities

### Dataset Requirements

**Size:**
- Minimum: 1000 instances of target compound
- Ideal: 10,000+ for robust statistics
- Multiple compounds: 20-50 different compound nouns

**Sources:**
- Wikipedia (encyclopedic context)
- Common Crawl (diverse contexts)
- Books corpus (varied usage)

**Annotations needed:**
- Token boundaries
- Compound span
- Compositionality rating (human-judged or proxy)

### Compute Requirements

**Based on paper:**
- Probe training: 6-8 hours × number of probes
- If reusing paper's probes: just inference (fast)
- Vocabulary extraction: ~few hours for 500 documents
- Total: manageable on single GPU (RTX A6000 or similar)

### Novel Contributions We Could Make

**Beyond the paper:**
1. **First analysis of common compound nouns** (paper focuses on entities + multi-token words)
2. **Compositionality spectrum** (paper: binary lexical/non-lexical)
3. **Semantic interventions** (paper: observational only)
4. **Cross-linguistic** (if we extend to other languages)

### Key Metrics to Collect

1. **Erasure score (ψ)** - direct from paper
2. **Probe accuracy at layer 9** - quantitative measure
3. **Layer of maximum erasure** - may vary by compound
4. **Context sensitivity** - does erasure depend on context?
5. **Intervention effect size** - causal validation

---

## Summary: Implications for "Washing Machine" Research

### Core Insight from Paper
LLMs develop an **implicit vocabulary** of lexical items beyond their token vocabulary. Multi-token sequences are assembled into unified representations in early layers (1-9), leaving a "footprint" of token erasure at the last position.

### Direct Application
"Washing machine" is likely an implicit vocabulary item. We can:
1. **Measure its erasure score** to quantify lexicality
2. **Map layer-wise assembly** to see when compound forms
3. **Compare to compositional phrases** to validate non-compositionality
4. **Intervene at critical layers** to test causal role

### Methodology Advantages
- **Quantitative:** Erasure score provides numerical measure
- **Layer-specific:** Identifies where representation forms
- **Model-agnostic:** Works across Llama-2/3, likely other models
- **Scalable:** Can analyze many compounds efficiently

### Open Questions for Our Research
1. Do all compound nouns show erasure, or only lexicalized ones?
2. Does erasure correlate with human judgments of compound unity?
3. Can we predict which compounds will be lexicalized by a model?
4. How does tokenization affect compound representation?
5. Are there different erasure patterns for different compound types (N+N, Adj+N, etc.)?

### Recommended Next Steps
1. **Immediate:** Calculate erasure score for "washing machine" using released code
2. **Short-term:** Build compound noun dataset and run probe analysis
3. **Medium-term:** Extend to intervention experiments
4. **Long-term:** Develop comprehensive theory of compound representation in LLMs

---

## Additional Notes

### Related Work Connections

**Elhage et al. (2022) - "Softmax Linear Units":**
- Found neurons activating on last tokens of multi-token constructs
- Observed retokenization neurons in late layers
- Complements this work: early detokenization + late retokenization

**Geva et al. (2023) - "Dissecting Recall of Factual Associations":**
- Subject enrichment at last token
- Factual knowledge concentrated in early layers
- Current work explains mechanism: erasure enables enrichment

**Nanda et al. (2023) - "Fact Finding":**
- Athlete → sport lookups
- Neuron-level analysis
- Confirms last-token storage pattern

**Gurnee et al. (2023) - "Finding Neurons in a Haystack":**
- Polysemantic neurons for multi-token constructs
- Examples: "apple developer", "research.gate"
- Provides neuron-level evidence for implicit vocabulary

### Limitations Noted by Authors

1. **No ground truth for implicit vocabulary**
   - Evaluation challenging
   - Compared to multi-token words + entities (proxy)
   - May miss other lexical item types

2. **Unknown entities not tested**
   - Only known entities from COUNTERFACT/Wikidata
   - Unclear if plausible fictional names show erasure
   - Could indicate training data memorization vs. compositional assembly

3. **Limited to Llama family**
   - Not tested on GPT, other architectures
   - Not tested on larger Llama models (70B+)
   - May be architecture-specific

4. **English only**
   - Authors acknowledge language bias
   - Low-resource languages would benefit from this analysis
   - Different languages may have different erasure patterns

### Extensions & Future Work (Suggested)

1. **Other languages** - especially low-resource
2. **Larger models** - does erasure scale with capacity?
3. **Other architectures** - GPT, PaLM, etc.
4. **Intermediate entities** - fictional but plausible names
5. **Training data correlation** - does ψ correlate with frequency?
6. **Fine-tuning effects** - does erasure change after fine-tuning?

---

## Key Equations & Algorithms

### Erasure Score (Equation 1)
```
ψ_{p,q} = 1/(1+2n) * [δ(q,0) + Σ_{t=p}^q Σ_{i=-2}^{-1} 1_within(t,i) · δ(t,i)]
```
Where:
- n = q - p + 1 (sequence length)
- δ(t,i) = change in predicted probability from layer 1 to L
- 1_within(t,i) = -1 if t+i < p, else 1

### Probability Change (Equation 2)
```
δ(t,i) = P_{p_i^(1)}(t+i | h_t^(1)) - P_{p_i^(L)}(t+i | h_t^(L))
```
- Measures "forgetting" of token t+i between layers

### Within-Boundary Indicator (Equation 3)
```
1_within(t,i) = { -1 if t+i < p
                 { 1  else
```
- Penalizes sequences where erasure extends beyond boundaries

### Algorithm 1: Document Segmentation
1. Score all possible n-grams in document
2. Sort by ψ score descending
3. Greedily select non-overlapping high-scoring segments
4. Result: segmentation into lexical units

---

## Citation Information

**Paper:**
Feucht, S., Atkinson, D., Wallace, B. C., & Bau, D. (2024). Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs. arXiv:2406.20086v3 [cs.CL].

**Code/Data:**
footprints.baulab.info

**Key Dependencies:**
- nnsight (Fiotto-Kaufman et al., 2024)
- Llama-2-7b (Touvron et al., 2023)
- Llama-3-8b (Meta, 2024)
- COUNTERFACT (Meng et al., 2022)
- The Pile (Gao et al., 2020)
