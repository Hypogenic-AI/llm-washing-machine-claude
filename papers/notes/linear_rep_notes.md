# The Linear Representation Hypothesis and the Geometry of Large Language Models

**Paper:** Park, K., Choe, Y.J., & Veitch, V. (2024). arXiv:2311.03658v2
**Code:** https://github.com/KihoPark/linear_rep_geometry
**Date Read:** 2026-02-07

---

## 1. What is the Linear Representation Hypothesis?

The **Linear Representation Hypothesis** is the idea that high-level concepts are represented linearly as directions in some representation space of a language model.

The paper identifies **three distinct interpretations** that have been used informally:

1. **Subspace notion**: Each concept is represented as a 1-dimensional subspace (direction). Example: Rep("woman") - Rep("man") and Rep("queen") - Rep("king") belong to a common subspace representing Male/Female.

2. **Measurement notion**: The probability of a concept value can be measured with a linear probe. Example: whether text is French is logit-linear in the representation.

3. **Intervention notion**: The value a concept takes can be changed by adding a steering vector without changing other concepts. Example: changing output from English to French by adding a direction vector.

**Key insight**: The paper shows these three notions are NOT the same but are formally connected through mathematical theorems.

---

## 2. What Does "Concept as a Direction" Mean Formally?

### Concepts as Counterfactual Variables

A **concept** W is formalized as a latent variable that:
- Is caused by context X
- Acts as a cause of output Y
- Can be changed in isolation (binary concepts: 0 or 1)
- Has an ordering (e.g., male⇒female)

For example, male⇒female has counterfactual pairs: {"man", "woman"}, {"king", "queen"}, {"roi", "reine"}

**Causal separability**: Two concepts W and Z are causally separable if they can vary freely and independently. Example: English⇒French and male⇒female are causally separable (can have king/queen/roi/reine), but English⇒French and English⇒Russian are NOT.

### Two Representation Spaces

Language models have TWO distinct representation spaces:
- **Embedding space (Λ)**: Maps context x to vector λ(x) ∈ ℝ^d
- **Unembedding space (Γ)**: Maps output word y to vector γ(y) ∈ ℝ^d
- Output probability: P(y | x) ∝ exp(λ(x)^T γ(y))

### Formal Definitions

**Unembedding representation (Definition 2.1)**:
γ̄_W is an unembedding representation of concept W if:
- γ(Y(1)) - γ(Y(0)) ∈ Cone(γ̄_W) almost surely
- Where Cone(v) = {αv : α > 0}
- Unique up to positive scaling

**Embedding representation (Definition 2.3)**:
λ̄_W is an embedding representation of concept W if:
- λ₁ - λ₀ ∈ Cone(λ̄_W) for context embeddings λ₀, λ₁ that:
  - P(W=1|λ₁)/P(W=1|λ₀) > 1 (relevant to target concept)
  - P(W,Z|λ₁)/P(W,Z|λ₀) = P(W|λ₁)/P(W|λ₀) (not relevant to off-target concepts Z)

---

## 3. How Do They Test It? Experiments and Probes

### Main Theoretical Results

**Theorem 2.2 (Measurement)**: The unembedding representation γ̄_W connects to linear probing:
- logit P(Y = Y(1) | Y ∈ {Y(0), Y(1)}, λ) = α λ^T γ̄_W
- The direction is the same for all counterfactual pairs (only scalar α varies)

**Theorem 2.5 (Intervention)**: The embedding representation λ̄_W enables steering:
- Adding λ̄_W to context changes probability of target concept W
- Does NOT change probability of causally separable concepts Z

**Lemma 2.4 (Connection)**: Embedding and unembedding representations are related:
- λ̄_W^T γ̄_W > 0 (positive for same concept)
- λ̄_W^T γ̄_Z = 0 (orthogonal for causally separable concepts)

### Experimental Setup

**Model**: LLaMA-2-7B (7 billion parameters, 32K vocabulary, 4096 dimensions)

**Concepts tested**: 27 total
- Morphological: verb⇒3pSg, verb⇒Ving, noun⇒plural, etc. (22 concepts from BATS 3.0)
- Language: English⇒French, French⇒German, French⇒Spanish, German⇒Spanish
- Semantic: small⇒big, male⇒female, frequent⇒infrequent

**Data sources**:
- BATS 3.0 (Bigger Analogy Test Set) for morphological concepts
- word2word bilingual lexicon for language pairs
- ChatGPT-4 generated pairs for some concepts
- Wikipedia contexts for measurement experiments

### Three Types of Experiments

**Experiment 1: Subspace Validation**
- Estimate γ̄_W as normalized mean of counterfactual differences
- Project each counterfactual pair onto γ̄_W using leave-one-out
- Compare to projections of random word pairs
- **Result**: 26/27 concepts show clear directional structure (only "thing⇒part" fails)

**Experiment 2: Linear Probing**
- Use γ̄_W as linear probe on contexts from Wikipedia
- Test if γ̄_W^T λ(x) predicts concept value
- **Result**: Concept directions successfully act as linear probes (e.g., French vs Spanish contexts)

**Experiment 3: Intervention/Steering**
- Add α λ̄_W to context embedding: λ_C,α(x_j) = λ(x_j) + α λ̄_C
- Measure change in logits for target vs off-target concepts
- **Result**: Steering vectors successfully change target concept without affecting causally separable concepts
- Example: "Long live the [king]" → "queen" becomes top-1 prediction with α=0.4

---

## 4. Models and Datasets

### Models
- **Primary**: LLaMA-2-7B (Touvron et al., 2023)
  - 7 billion parameters
  - Trained on 2 trillion SentencePiece tokens (90% English)
  - 32,000 token vocabulary
  - 4,096 dimensional embeddings
  - Decoder-only Transformer

- **Validation**: Gemma-2B (Mesnard et al., 2024)
  - Used to compare Euclidean vs causal inner products

### Datasets

**Counterfactual pairs**:
- BATS 3.0 (Gladkova et al., 2016): Morphological concepts
  - Only single-token words used
  - Example counts: 32 pairs for verb⇒3pSg, 63 for noun⇒plural
- word2word bilingual lexicon: Language pairs (35-46 pairs per language pair)
- ChatGPT-4 generated: Semantic concepts

**Context data**:
- Wikipedia: Random-length contexts sampled from language-specific pages
  - French⇒Spanish: 209 French contexts, 231 Spanish contexts
  - French⇒German: 278 French, 205 German
  - Used for measurement experiments
- ChatGPT-4: 15 contexts for intervention experiments
  - Example: "Long live the", "The lion is the", etc.

**Tokenization challenge**: Words must be single tokens in LLaMA-2's vocabulary (can't use multi-token words like "princess" = "prin" + "cess")

---

## 5. Key Results: Which Concepts Are Linear?

### Strong Linear Representations (26/27 concepts)

All tested concepts showed linear structure EXCEPT "thing⇒part":

**Morphological concepts** (22 from BATS):
- Verb inflections: verb⇒3pSg, verb⇒Ving, verb⇒Ved
- Derivations: verb⇒V+able, verb⇒V+er, verb⇒V+tion
- Adjective forms: adj⇒comparative, adj⇒superlative, adj⇒adj+ly
- Noun forms: noun⇒plural, pronoun⇒possessive
- Case: lower⇒upper (capitalization)

**Language concepts** (4):
- English⇒French
- French⇒German
- French⇒Spanish
- German⇒Spanish

**Semantic concepts** (4):
- male⇒female (strong signal)
- small⇒big
- frequent⇒infrequent
- country⇒capital

### Failed Concept
- **thing⇒part**: Does NOT show linear representation
  - Example pairs: (bus, seats), (ant, black)
  - Too abstract/compositional?

### Quantitative Evidence

**Projection distributions** (Figure 2, Figure 7):
- Counterfactual pairs show strong right skew when projected onto concept direction
- Clear separation from random pairs
- Leave-one-out validation confirms consistency

**Causal orthogonality** (Figure 3):
- Most causally separable concept pairs have inner product ≈ 0
- Block diagonal structure emerges for semantically similar concepts
- Example: verb concepts (1-10) cluster together, language pairs (24-27) cluster together

**Non-trivial structure**:
- lower⇒upper has non-zero inner product with English⇒German but not French⇒Spanish
- Explanation: Different capitalization rules across languages

---

## 6. Relation to Compound/Multi-token Concepts

### Direct Relevance to "Washing Machine" Research

**Critical limitation**: The paper explicitly EXCLUDES multi-token concepts:
- Only single-token words used in experiments
- Words like "princess" (tokenized as "prin"+"cess") were discarded
- This is a major constraint for studying "washing machine" which is definitely multi-token

### Implications for Compound Concepts

**What the paper DOES tell us**:

1. **Compositionality through vector arithmetic**: The successful intervention experiments show that concepts combine linearly:
   - Adding male⇒female direction to "king" context → "queen"
   - Adding English⇒French to "king" → "roi"
   - This suggests compound concepts MIGHT be representable as sums of component directions

2. **Causal separability is key**:
   - Concepts must be causally separable to be orthogonal
   - For "washing machine": Need to identify which aspects are causally separable
   - Candidates: appliance type (washer/dryer), size (compact/full), power source (electric/gas)?

3. **Failed abstract concepts**: "thing⇒part" failed to show linear structure
   - This is compositional/relational like "washing machine"
   - Suggests purely compositional concepts may not have simple linear representations

**What remains unknown**:
- How are multi-token concepts like "washing machine" represented?
- Is "washing" + "machine" a simple sum, or more complex interaction?
- Do compound concepts have their own emergent direction beyond components?

### Potential Extensions

The paper's framework could be extended:
- Define counterfactual pairs for "washing machine":
  - (washing machine, dryer)
  - (washing machine, dishwasher)
  - (washing machine, machine à laver) [French translation]
- Check if γ("washing machine") - γ("dryer") forms consistent direction
- BUT: Need to handle multi-token representation somehow

**Hypothesis for future work**: "washing machine" might be represented as:
- γ̄_appliance-type (linear component)
- γ̄_washing-function (linear component)
- Possibly non-linear interaction term
- Embedding space might have unified representation even if tokens are separate

---

## 7. Implications for Our "Washing Machine" Research

### Major Findings Relevant to Our Project

**1. Inner Product Choice is Fundamental**

The most important finding: **The choice of inner product matters critically**

- Euclidean inner product does NOT respect semantic structure
- **Causal inner product** is needed: <γ̄, γ̄'>_C = γ̄^T Cov(γ)^(-1) γ̄'
- This makes causally separable concepts orthogonal

**For washing machine research**: We should:
- NOT use standard cosine similarity on raw embeddings
- Use the causal inner product: M = Cov(γ)^(-1) where γ is uniformly sampled from vocabulary
- This will give meaningful distances between concept directions

**2. Two Representation Spaces Must Be Considered**

- Unembedding space (output): Where next-word predictions happen
- Embedding space (input): Where context is represented
- They are unified through the causal inner product via Riesz isomorphism

**For washing machine**:
- Study both λ("washing machine in the...") AND γ("washing machine")
- They should be related: λ̄_W = Cov(γ)^(-1) γ̄_W
- Can construct steering vectors from unembedding representations

**3. Counterfactual Pairs are the Key Methodology**

The paper's approach:
- Find pairs that differ only in concept value
- Check if differences align to common direction
- Use leave-one-out validation

**For washing machine**:
- Create counterfactual pairs:
  - Appliance type: (washing machine, dryer), (washing machine, dishwasher)
  - Function: (washing machine, cleaning device), (washing machine, appliance)
  - Language: (washing machine, machine à laver, Waschmaschine, lavadora)
- Check if differences form consistent directions
- Estimate concept representations from these pairs

**4. Linear Probing vs Unembedding Representations**

Key distinction (Section 2.2):
- Unembedding representation: Pure concept direction, no correlations
- Linear probe: Includes information about correlated concepts

**Example**: If French text is often about men, a linear probe learns this, but unembedding representation does not.

**For washing machine**:
- Unembedding representation will capture "pure" washing-machine-ness
- Linear probe might capture spurious correlations (e.g., if "washing machine" appears more in home contexts)
- We want the unembedding representation for interpretability

**5. Multi-token Limitation**

This is a critical gap for our research:
- Paper only uses single-token words
- "washing machine" is definitely multi-token
- No formal theory provided for how multi-token concepts compose

**Possible approaches**:
1. **Average token embeddings**: λ("washing machine") = mean(λ_washing, λ_machine)?
2. **Last token only**: Use embedding after processing full phrase in context
3. **Define concept via contexts**: Use contexts where "washing machine" appears as counterfactuals
4. **Phrase-level unembedding**: Can we extract γ("washing machine") from model's next-token probabilities?

### Specific Research Questions Enabled

**Question 1**: Is there a linear "appliance type" direction?
- Counterfactuals: (washing machine, dryer), (refrigerator, oven), (dishwasher, microwave)
- Project differences onto estimated direction
- Check orthogonality with other concepts (size, brand, etc.)

**Question 2**: How does "washing" + "machine" compose?
- Compare: γ̄_washing-machine vs (γ̄_washing + γ̄_machine)
- Is composition linear?
- Use causal inner product to measure

**Question 3**: Can we steer model to generate "washing machine" vs alternatives?
- Construct λ̄_washing-vs-drying from counterfactual pairs
- Add to context: "I need to clean clothes, I'll use my [___]"
- Does adding λ̄ increase P(washing machine)?

**Question 4**: Cross-lingual washing machine representations
- Are (washing machine, machine à laver, Waschmaschine) aligned?
- Do they share common direction separate from language direction?
- Test: γ̄_washing-machine should be orthogonal to γ̄_English⇒French

### Methodological Framework for Our Research

Based on this paper, here's a concrete plan:

**Step 1: Estimate Causal Inner Product**
```
1. Extract all unembedding vectors γ from LLaMA-2 vocabulary
2. Compute Cov(γ) where γ ~ Uniform(vocabulary)
3. Set M = Cov(γ)^(-1)
4. Use <v, w>_C = v^T M w for all similarity computations
```

**Step 2: Collect Counterfactual Pairs**
```
Appliance contrasts:
- (washing machine, dryer)
- (washing machine, dishwasher)
- (washing machine, oven)
- (refrigerator, microwave) [control]

Function contrasts:
- (washing machine, cleaning tool)
- (washing machine, appliance)

Translations:
- (washing machine, machine à laver, Waschmaschine, lavadora, lavatrice)
```

**Step 3: Extract Multi-token Representations**
```
Challenge: γ("washing machine") doesn't exist directly

Option A: Context-based
- Generate contexts: "I put clothes in the washing machine"
- Extract λ(context) at position before "washing"
- Use as proxy for concept embedding

Option B: Compositional
- γ̄_WM = f(γ("washing"), γ("machine"))
- Try: mean, sum, concatenation
- Validate against contexts

Option C: Phrase unembedding
- Look at P(next_word | "washing machine") probabilities
- Extract effective unembedding from distribution
```

**Step 4: Test Linear Representation Hypothesis**
```
1. Compute differences: d_i = γ(variant1_i) - γ(variant2_i)
2. Estimate direction: γ̄ = normalize(mean(d_i))
3. Project with LOO: <γ̄_(−i), d_i>_C
4. Compare to random pairs
5. Check if distribution is right-skewed
```

**Step 5: Construct Probes and Steering Vectors**
```
Linear probe: Use γ̄_WM to predict P(washing machine | context)
Steering: λ̄_WM = M γ̄_WM to change outputs toward "washing machine"
Test intervention: Add λ̄_WM to prompts about appliances
```

---

## 8. Code and Tools Available

### Repository
**GitHub**: https://github.com/KihoPark/linear_rep_geometry

### Key Components (Based on Paper Methods)

**1. Causal Inner Product Estimation**
- Compute vocabulary covariance: Cov(γ)
- Invert to get M = Cov(γ)^(-1)
- Implements <γ̄, γ̄'>_C = γ̄^T M γ̄'

**2. Unembedding Representation Extraction**
```python
# Pseudocode based on paper
def estimate_unembedding_rep(concept_pairs, model):
    """
    concept_pairs: List of (word0, word1) tuples
    Returns: Normalized concept direction γ̄_W
    """
    diffs = []
    for (y0, y1) in concept_pairs:
        gamma_0 = model.get_unembedding(y0)
        gamma_1 = model.get_unembedding(y1)
        diffs.append(gamma_1 - gamma_0)

    gamma_tilde = mean(diffs)
    # Normalize using causal inner product
    gamma_bar = gamma_tilde / sqrt(<gamma_tilde, gamma_tilde>_C)
    return gamma_bar
```

**3. Embedding Representation via Riesz Isomorphism**
```python
def get_embedding_rep(gamma_bar, M):
    """
    Theorem 3.2: <γ̄_W, ·>_C = λ̄_W^T
    So: λ̄_W = M γ̄_W
    """
    lambda_bar = M @ gamma_bar
    return lambda_bar
```

**4. Linear Probing**
```python
def probe_concept(contexts, gamma_bar, model):
    """
    Theorem 2.2: Test if γ̄^T λ(x) predicts concept
    """
    scores = []
    for context in contexts:
        lambda_x = model.get_embedding(context)
        score = gamma_bar.T @ lambda_x
        scores.append(score)
    return scores
```

**5. Intervention/Steering**
```python
def steer_output(context, lambda_bar, alpha, model):
    """
    Theorem 2.5: Add α λ̄_W to change output
    """
    lambda_original = model.get_embedding(context)
    lambda_steered = lambda_original + alpha * lambda_bar

    # Generate with modified embedding
    output = model.generate_from_embedding(lambda_steered)
    return output
```

**6. Visualization Tools**
- Histogram plotting for counterfactual projections (Figure 2, 7)
- Heatmap for concept inner products (Figure 3)
- Intervention trajectory plots (Figure 5, 12)

### Model Access
```python
# Using HuggingFace
from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Key attributes:
# - model.embed_tokens: Embedding layer (context → λ)
# - model.lm_head: Unembedding matrix (γ → logits)
# - Dimensions: 32000 vocab × 4096 hidden
```

### Datasets Provided
Based on paper, likely includes:
- BATS 3.0 counterfactual pairs (filtered for single tokens)
- Language translation pairs (from word2word)
- Wikipedia context samples
- ChatGPT-4 generated prompts

### Analysis Scripts
Likely includes:
1. Concept direction estimation with LOO validation
2. Causal inner product computation
3. Linear probe evaluation
4. Intervention experiments with top-K word tracking
5. Cross-lingual alignment tests

### Practical Use for Our Research

**Immediate applications**:
1. Clone repo and run on LLaMA-2-7B
2. Extend to our "washing machine" concept pairs
3. Handle multi-token issue (may need modification)
4. Estimate M matrix for causal inner product
5. Test if washing-machine-related concepts have linear structure

**Modifications needed**:
- Multi-token concept extraction
- Domain-specific counterfactual pairs (appliances)
- Possibly larger model (LLaMA-2-13B or 70B for better representations)
- Cross-model validation (GPT, Claude, etc.)

---

## Summary Table: Key Theorems

| Theorem | Statement | Interpretation | Use Case |
|---------|-----------|----------------|----------|
| **2.2 Measurement** | logit P(Y(1)\|Y∈{Y(0),Y(1)}, λ) = α λ^T γ̄_W | Unembedding rep acts as linear probe | Predicting concept from context |
| **2.5 Intervention** | P(Y(W,1)\|..., λ+cλ̄_W) is increasing in c | Embedding rep enables steering | Changing model output |
| **Lemma 2.4** | λ̄_W^T γ̄_W > 0, λ̄_W^T γ̄_Z = 0 | Embedding/unembedding connection | Relating input/output spaces |
| **3.2 Unification** | <γ̄_W, ·>_C = λ̄_W^T | Causal inner product unifies representations | Converting unembedding → embedding |
| **3.4 Explicit Form** | M^(-1) = GG^T, G^T Cov(γ)^(-1) G = D | Causal inner product from vocabulary | Computing M = Cov(γ)^(-1) |

---

## Critical Takeaways for "Washing Machine" Project

1. **Inner product choice is NOT arbitrary** - must use causal inner product M = Cov(γ)^(-1)

2. **Linear representations exist** but only for 26/27 tested concepts (thing⇒part failed)

3. **Multi-token concepts are unexplored** - this is the main gap for our research

4. **Counterfactual methodology is powerful** - we should collect washing-machine counterfactual pairs

5. **Two spaces must be unified** - use Riesz isomorphism to connect embedding/unembedding

6. **Causally separable concepts are orthogonal** - need to identify what aspects of "washing machine" are separable

7. **Steering is possible** if linear representations exist - can we steer toward "washing machine"?

8. **Code is available** - practical implementation exists, needs extension to multi-token case

---

## Open Questions for Our Research

1. How do we extract γ("washing machine") when it's multiple tokens?
2. Is "washing machine" compositional: γ̄_WM ≈ f(γ̄_washing, γ̄_machine)?
3. Does the linear representation hypothesis hold for compound noun phrases?
4. Can we find causally separable aspects of appliances (type, function, size, brand)?
5. How does this extend to larger models (70B, GPT-4)?
6. Do different languages share the same "washing machine" direction?
7. Can we use steering to fix model errors about washing machines?

---

## Next Steps

1. **Replicate paper experiments** on LLaMA-2-7B with their code
2. **Estimate causal inner product** M = Cov(γ)^(-1) for our model
3. **Collect counterfactual pairs** for washing machine and related appliances
4. **Develop multi-token extraction method** (context-based or compositional)
5. **Test linear representation hypothesis** for "washing machine"
6. **Build steering vectors** if linear structure exists
7. **Cross-lingual validation** (washing machine across languages)
8. **Scale to larger models** and compare representations
