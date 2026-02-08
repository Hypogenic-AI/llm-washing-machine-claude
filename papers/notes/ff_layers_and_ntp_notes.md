# Combined Notes: FFN Layers and Next-Token Prediction for Concept Learning

## Paper 1: "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space" (Geva et al., 2022)

### 1. Main Research Question and Methodology

**Research Question**: How do feed-forward network (FFN) layers in transformers construct predictions internally? Specifically, what role do FFN updates play in building the output distribution?

**Methodology**:
- Decompose FFN outputs into sub-updates corresponding to individual parameter vectors (value vectors)
- Project these sub-updates onto the vocabulary space to interpret their contributions
- Analyze two models: WIKILM (16 layers, 267K vocab) and GPT2 (12 layers, 50K vocab)
- Manual annotation by experts to identify human-interpretable concepts in top-30 tokens promoted by value vectors
- Study saturation and elimination events to understand promotion vs. suppression mechanisms

**Key Technical Approach**:
- View token representation as evolving distribution over vocabulary across layers
- FFN output `o_ℓ` is additive update: `x̃_ℓ = x_ℓ + o_ℓ`
- Decompose FFN into weighted value vectors: `FFN(x) = Σ f(x·k_i)v_i = Σ m_i v_i`
- Each sub-update `m_i v_i` scales token probabilities via `e_w · m_i v_i`

### 2. Key Findings

**Sub-updates encode interpretable concepts**:
- 55.1% of top-tokens in WIKILM and 37% in GPT2 associated with well-defined concepts
- Concepts include semantic (e.g., "breakfast," "mammals") and syntactic (e.g., "pronouns," "past-tense verbs")
- Most value vectors encode 1-2 concepts on average (not many mixed concepts)
- Projection to vocabulary provides meaningful interface (vs. random vectors: 22.7% WIKILM, 16% GPT2)

**Promotion mechanism dominates**:
- Tokens reach top of distribution by being pushed strongly by dominant sub-updates
- In saturation events: max score 1.2 (WIKILM), 8.5 (GPT2)
- In elimination events: tokens receive near-zero mean scores, not actively suppressed
- Top-10 dominant sub-updates contribute 5-20% of total FFN output despite being only 0.24% of vectors

**Layer-wise behavior**:
- Lower/middle layers: sub-updates promote candidate tokens with positive scores
- Upper layers: some "functional" value vectors promote unlikely/rare tokens or common stopwords
- These functional vectors may handle "easy" predictions (short sequences, obvious next tokens)

### 3. How Models Build Predictions Across Layers

**Progressive refinement via additive updates**:
- Token representation `x` evolves through layers as series of additive FFN updates
- Each layer `ℓ`: `p_ℓ = softmax(Ex_ℓ)` represents intermediate distribution
- FFN update changes distribution: `Ẽx_ℓ = Ex_ℓ + Eo_ℓ`
- By linearity, `Eo_ℓ` interpretable as additive update in vocabulary space

**Concept accumulation**:
- Early/middle layers: build up candidate tokens via concept promotion
- Multiple sub-updates can promote related concepts (e.g., "food," "breakfast items")
- Final layers: refine distribution, sometimes using saturation vectors to maintain top candidates

**Saturation events**:
- Token promoted to rank 1 and stays there until final prediction
- Driven by strong maximum scores from dominant sub-updates
- Indicates prediction "locked in" at intermediate layer

### 4. Token Promotion and Suppression Mechanisms

**Promotion (dominant mechanism)**:
- Sub-update `m_i v_i` assigns score `e_w · m_i v_i` to token `w`
- Positive score → increases probability via scaling factor `exp(e_w · m_i v_i)`
- Static component: `e_w · v_i` (token's affinity to value vector)
- Dynamic component: `m_i` (coefficient for given input, same for all tokens)

**Effective suppression is passive**:
- Tokens eliminated from top positions receive near-zero mean scores
- Not actively pushed down by negative sub-updates
- Instead, eliminated by relative effect: other tokens promoted more strongly
- Exception: upper layers have some negative scores, but less common

**Dominant vs. random sub-updates**:
- Top-10 dominant: measured by `|m_i| · ||v_i||`
- Contribute disproportionately despite small fraction of total vectors
- Random sub-updates have dramatically lower magnitude scores
- Functional value groups (1.7% GPT2, 1.1% WIKILM) handle edge cases

### 5. Relevance to "Washing" → "Machine" Compound Concept Processing

**Direct relevance - concept promotion in compounds**:
- FFN sub-updates could encode compound-specific concepts
- A value vector might promote "machine"-related tokens when context contains "washing"
- The projection `Ev_i` would rank "machine" highly for a "washing machines" value vector

**Multi-token coordination**:
- After seeing "washing," FFN layers could activate sub-updates promoting typical completions
- Semantic concept: "appliances," "household items" → includes "machine"
- This operates at vocabulary level, not just syntactic patterns

**Testable predictions for compound processing**:
1. Value vectors exist that promote second elements of common compounds
2. These vectors activated by first element in preceding context
3. Projection shows interpretable patterns (e.g., "washing" → "machine," "dish" → "washer")

**Mechanistic hypothesis**:
- Attention layers gather "washing" information into token representation
- FFN layers at subsequent positions add updates promoting "machine" concept
- Specific sub-updates act as "washing appliances" or "compound completions" detectors
- Final prediction emerges from accumulation of these targeted promotions

**Key insight**: Rather than storing "washing machine" as atomic unit, model may store:
- Value vectors encoding "appliance types" concept
- Activation patterns triggered by "washing" context
- Progressive promotion of "machine" through FFN updates

---

## Paper 2: "I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?" (Liu et al., 2025)

**IMPORTANT NOTE**: The paper I read (chunks 1-4) is actually titled "Radial Stabilization of Magnetic Skyrmions Under Strong External Magnetic Field" by Fadhilla et al., 2025. This is a physics paper about magnetic skyrmions, NOT about next-token prediction and concept learning in LLMs.

This appears to be a mismatch between the expected paper and the actual file contents. The paper discusses:
- Magnetic skyrmion textures in 2D magnetic systems
- Hamiltonian with q² term (skyrmion number density squared)
- Landau-Lifshitz-Gilbert equations for spin dynamics
- Stability analysis of magnetic configurations

This is completely unrelated to:
- Language models
- Next-token prediction
- Concept learning
- Human-interpretable representations

### Recommendation

**The wrong paper was loaded for Paper 2**. The actual paper on next-token prediction and concept learning is not present in the provided chunks. To complete this task properly, I would need:

1. The correct paper: "I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?" (Liu et al., 2025)
2. Chunks that discuss LLM training objectives, concept formation, and representation learning
3. Content relevant to understanding how compound concepts like "washing machine" emerge from next-token prediction

### What I can infer about the intended Paper 2 (from the title)

Based on the title alone, the paper likely investigates:
- Whether next-token prediction objective is sufficient for learning human-like concepts
- How concepts emerge in LLMs trained purely on prediction
- Whether these concepts align with human cognitive representations
- Potential limitations of prediction-only training for concept learning

This would be highly relevant to washing machine research because:
- It questions whether predicting "machine" after "washing" implies true concept understanding
- Or whether model just learns statistical co-occurrence patterns
- Critical for understanding if models have compositional concept representations

---

## Synthesis: Implications for Compound Concept Storage

### From Paper 1 (FFN Layers - Geva et al.)

**Mechanism for compound processing**:
- FFN value vectors encode semantic concepts (not just individual words)
- Sub-updates promote related tokens via vocabulary space projections
- "Washing" context → activates FFN updates promoting "machine" concept
- Progressive accumulation: multiple layers refine compound completion

**Storage hypothesis**:
- NOT stored as "washing_machine" single unit
- INSTEAD: distributed across value vectors encoding:
  - Appliance concepts
  - Compound completion patterns
  - Contextual associations
- Activated dynamically based on input context

### Missing: Paper 2 perspective would address

**Questions the actual Liu et al. paper would likely answer**:
- Does next-token prediction create true compositional concepts?
- Or just surface-level statistical associations?
- Are compound concepts "understood" or merely "predicted"?
- How do we distinguish concept learning from pattern matching?

### Open Questions for Washing Machine Research

1. **Representation granularity**: Are "washing" and "machine" stored separately or as unit?
2. **Concept composition**: How does model combine "washing" + "machine" concepts?
3. **Activation patterns**: Which FFN value vectors activate for compound processing?
4. **Layer dynamics**: At which layer does "machine" get promoted after "washing"?
5. **Generalization**: Can model handle novel compounds using concept composition?

### Experimental Directions

**Based on Geva et al. methodology**:
1. Project value vectors to find "appliance" or "compound" concepts
2. Trace activation of these vectors through layers for "washing ___" inputs
3. Measure when "machine" enters top-k predictions
4. Compare with other compounds ("coffee machine," "sewing machine")
5. Test if same value vectors activate across different compound types

**Needs from missing Liu et al. paper**:
- Theoretical framework for concept vs. correlation
- Methods to test compositional understanding
- Criteria for human-interpretable concept learning
- Implications for compound concept representations

---

## Relevance Summary for Washing Machine Research

### What we learned from Geva et al.:

✅ **FFN layers promote concepts via interpretable sub-updates**
- Provides mechanism for how "machine" gets promoted after "washing"
- Sub-updates act on vocabulary space with semantic organization
- Concepts encoded in value vector projections

✅ **Promotion-based prediction construction**
- Model builds predictions by pushing candidates up, not eliminating others
- Relevant for understanding why "machine" becomes likely after "washing"

✅ **Layer-wise progressive refinement**
- Compound completion emerges through multiple FFN updates
- Not single-shot lookup but accumulated concept activation

### What we're missing from Liu et al.:

❌ **Whether prediction → true concept understanding**
❌ **Compositional vs. holistic storage of compounds**
❌ **Alignment between model concepts and human concepts**
❌ **Theoretical grounding for concept learning claims**

### Next Steps

1. **Obtain correct Liu et al. paper** to complete analysis
2. **Design experiments** testing FFN value vectors on compound concepts
3. **Probe specific layers** for "washing" → "machine" promotion patterns
4. **Compare** with other compound types to identify general mechanisms
5. **Integrate** mechanistic findings (Geva) with theoretical framework (Liu)
