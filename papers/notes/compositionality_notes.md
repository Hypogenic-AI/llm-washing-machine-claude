# Combined Notes: Compositionality and Idiomaticity in Language Models

## Research Context
These papers examine how multi-word expressions (MWEs) and compound concepts are represented in language models - specifically whether they are stored compositionally (built from component parts) or holistically (as unified concepts). This directly relates to our research question about how compound concepts like "washing machine" are stored in LLMs.

---

## Paper 1: "Interpreting token compositionality in LLMs: A robustness analysis" (Aljaafari et al., 2024)

### Main Research Question and Methodology

**Core Question**: How do large language models handle compositional linguistic structures, particularly at the token and phrase level?

**Methodology - Constituent-Aware Pooling (CAP)**:
- A systematic perturbation method that groups token-level activations into larger constituent units (words or phrases) at different model layers
- Applied at two levels:
  - **Word-level (TW-CAP)**: Groups tokens that form a single word
  - **Phrase-level (TP-CAP)**: Groups tokens that form syntactic phrases using Benepar parser
- Three aggregation protocols tested:
  - **Max**: Selects maximum activation within a segment
  - **Mean**: Computes average activation
  - **Sum**: Sums activations
- CAP intervention applied progressively across layers to test where semantic composition occurs

### Models and Datasets Used

**Models Tested**:
- GPT-2 (small: 124M, medium: 355M, large: 774M parameters)
- Gemma-1 (2B parameters)
- Llama (3B and 8B parameters)
- Qwen (0.5B, 1.5B, 3B parameters)
- Tested both pre-trained and fine-tuned versions

**Datasets**:
- Three WordNet-derived tasks:
  1. **Inverse Definition Modelling (IDM)**: Predicting a term from its definition
  2. **Synonym Prediction (SP)**: Generating synonyms for words
  3. **Hypernym Prediction (HP)**: Generating more general terms
- Only examples correctly predicted by original models were used as baseline

### Key Findings About Compositional vs. Holistic Representations

**1. No Single Layer Integrates Composition**:
- No specific layer reliably integrates tokens into unified semantic representations based on constituent parts
- Compositional inference is NOT a purely incremental process across layers
- Rather than progressive semantic buildup, models show "fragmented information processing"

**2. Larger Models Are More Fragile**:
- Contrary to expectations, larger models exhibit GREATER sensitivity to compositional perturbations
- Example: Gemma-2B accuracy drop at 1% layer position: 97.91% (Max protocol) vs. Llama3-8B: 25.49%
- Larger models rely more heavily on fine-grained token-level features in early layers

**3. Information Distribution Patterns**:
- Models distribute information in a fragmented manner across layers
- Max aggregation showed most dramatic impact, suggesting lack of integration of compositional information
- Sum aggregation performed best, indicating cumulative information loss rather than systematic composition

**4. Fine-tuning Helps But Doesn't Solve the Problem**:
- Fine-tuning improved performance, especially in 75%-100% layer positions
- However, it doesn't fully resolve the challenge of forming stable compositional semantic representations
- Points to architectural limitation rather than training limitation

**5. Context Effects**:
- Early-layer CAP interventions cause severe accuracy drops
- Performance fluctuations across layers suggest token dependencies modeled by aggregation paths spanning multiple layers rather than localized composition

### How They Probe for Compositionality

**Information-Theoretic Framework**:
- Analyzes relationship between predicted token Y and input token representations R_l(X) at each layer l
- Key insight: Models maximize information gain (IG) at each layer for next-token prediction
- This creates incentive to DELAY aggregation to later layers to maximize layer-wise information
- Results in minimized mutual information I(R_l(X_i), R_l(X_j)) between tokens at same layer
- Longer aggregation paths = more distributed composition

**Evaluation Metrics**:
- **Original accuracy (A_o)**: Model accuracy before CAP
- **Grouped accuracy (A_c)**: Model accuracy after CAP (averaged across protocols)
- **Accuracy drop (ΔA)**: Performance loss due to CAP intervention
- Lower ΔA = more robust compositional behavior

**Token Reduction Analysis**:
- Examined how CAP reduces sequence length (K → G)
- Found token reduction percentage is a factor but NOT the sole determinant of performance degradation
- Architectural factors (model size, depth, MLP dimensions) also play major roles

### Relevance to "Washing Machine" Question

**Critical Implications**:

1. **Distributed Rather Than Localized**: "Washing machine" is unlikely to be represented as a unified concept at any single layer - instead distributed across multiple layers with long dependency paths

2. **Token-Level Fragmentation**: The compound would be broken into tokens ("wash", "ing", "machine" or similar) with representations that remain largely separate rather than fully integrated

3. **Vulnerability to Perturbations**: Replacing individual components (e.g., "cleaning machine" vs "washing machine") would substantially disrupt the representation, suggesting models don't form robust holistic concepts

4. **Larger Models Worse**: Counterintuitively, larger models may handle "washing machine" WORSE than smaller ones due to greater fragmentation and reliance on token-level features

5. **Architectural Constraint**: The findings suggest transformers' training objective (next-token prediction) and architecture inherently work against systematic compositional semantics

**Key Quote**: "Compositional semantics are not reliably localisable within any fixed layer of standard Transformer models. This holds across model scales, supervision types, and inference tasks, and instead appears tied to architectural depth."

---

## Paper 2: "Probing for idiomaticity in vector space models" (Garcia et al., 2021)

### Main Research Question and Methodology

**Core Question**: Whether and to what extent idiomaticity in multiword expressions (MWEs), particularly noun compounds (NCs), is accurately incorporated by word representation models.

**Methodology - Four Probing Measures**:

**P1: NC vs. Synonym Similarity**
- Compares NC to its synonym (e.g., "grey matter" vs. "brain")
- Expected: High similarity (~1.0) regardless of idiomaticity
- Measured at sentence level and NC level in context

**P2: NC vs. Component Word Similarity**
- Compares NC to its individual components (head or modifier)
- Expected: Lower similarity for idiomatic NCs (e.g., "grey matter" vs. "matter")
- Should correlate positively with compositionality scores

**P3: NC vs. Component-Synonym NC**
- Compares NC to artificial NC made from synonyms of components
- Example: "grey matter" vs. "silvery material"
- Tests sensitivity to lack of substitutability in idiomatic expressions
- Should show higher similarity for compositional than idiomatic NCs

**P4: NC in Context vs. Out of Context**
- Measures how much context affects NC representation
- Expected: Greater difference for idiomatic NCs in informative contexts
- Compared between naturalistic (NAT) and neutral (NEU) contexts

### Models and Datasets Used

**Models Tested**:
- **GloVe**: Static baseline (non-contextualised)
- **ELMo**: Contextualised embeddings
- **BERT**: Standard multilingual BERT
- **DistilBERT**: Distilled version of BERT
- **Sentence-BERT (SBERT)**: Sentence-level BERT variant
- **BERTRAM**: BERT with type-level vectors for NCs (improved results)
- Tested on both English and Portuguese

**Dataset - Noun Compound Senses (NCS)**:
- Based on NC Compositionality dataset (Reddy et al., 2011; Cordeiro et al., 2019)
- **280 English NCs** and **180 Portuguese NCs**
- Total: 9,220 sentences (5,620 English, 3,600 Portuguese)
- Human compositionality scores on 0-5 scale (0=idiomatic, 5=compositional)
- Two conditions:
  - **NAT**: Naturalistic corpus sentences (avg 23.39 words English, 13.03 Portuguese)
  - **NEU**: Neutral context sentences ("This is a/an <NC>") (5 words both languages)
- Variants created with:
  - Synonyms of whole NC (NC_syn)
  - Synonyms of each component (NC_synW)
  - Individual components

### Key Findings About Compositional vs. Holistic Representations

**1. High Sentence Similarity Masks Poor NC Representation**:
- P1 sentence-level similarities very high (suggesting idiomaticity captured)
- BUT P1 NC-level similarities show moderate correlation with idiomaticity
- Lower similarities for idiomatic than compositional cases
- High sentence similarity likely due to lexical overlap in context words, NOT semantic understanding

**2. Models Cannot Distinguish Component Overlap**:
- P2 showed HIGH similarities between idiomatic NCs and their components
- Example: "poison pill" (emergency exit) had 0.94 similarity with "pill"
- Models prioritize LEXICAL overlap over SEMANTIC overlap
- Cannot distinguish partial overlap (compositional) from no overlap (idiomatic)

**3. Lack of Substitutability Not Captured**:
- P3 showed high similarities across idiomaticity spectrum
- Average similarities HIGHER than P1 (opposite of expected)
- Example: "wet blanket" BERT similarity 0.91 with "damp cloak" vs. 0.77 with "loser"
- Models fail to detect meaning change from component substitution

**4. Context Plays Limited Role**:
- P4 showed high similarity between NC in-context and out-of-context (>0.8 for BERT)
- NC out-of-context often better approximation than NC synonym
- Only weak correlation with idiomaticity
- Suggests context not playing bigger role for idiomatic than compositional NCs

**5. NAT vs NEU Conditions Highly Correlated**:
- Very strong correlations between naturalistic and neutral conditions
- For SBERT: ρ > 0.85 (English) and > 0.76 (Portuguese) across P1, P2, P3
- Indicates neutral sentences as effective as naturalistic ones for these probes
- Models NOT adequately incorporating context to capture idiomaticity

**6. Sentence Length Confound**:
- Moderate to strong correlation between sentence length and cosine similarities (Table 5)
- Vector averaging means high similarities may reflect word count overlap rather than semantic similarity
- Especially problematic for contextualised models (e.g., DistilB ρ=0.89 for P2 English)

### How They Probe for Compositionality

**Inspired by Semantic Priming Paradigm**:
- Related stimuli should achieve greater similarity than unrelated stimuli
- Minimal modifications between conditions (paraphrases) to isolate effects

**Similarity Measures**:
- Cosine similarity between embeddings
- Three embedding types compared:
  - NC out of context (NC)
  - NC in context of sentence (NC ⊂ S)
  - Sentence containing NC (S ⊃ NC)
- Spearman ρ correlation with human idiomaticity scores

**Standard Composition Operation**:
- Vector averaging of (sub-)tokens to create NC embedding
- Also tested concatenation (slightly better results, ~0.06 improvement)
- BERTRAM approach: Type-level vectors for whole NCs (best results)

**Controlled Design**:
- Neutral sentences remove contextual confounds
- Naturalistic sentences test real-world usage
- Expert review ensures grammaticality after substitutions
- Multiple sentences per NC to reduce variance

### Relevance to "Washing Machine" Question

**Critical Implications**:

1. **Idiomatic Meaning Not Accurately Represented**: Models like BERT and ELMo fail to distinguish between compositional compounds (like "washing machine") and idiomatic ones (like "grey matter"), suggesting even literal compound concepts may not be properly integrated

2. **Component-Level Bias**: "Washing machine" would likely show very high similarity to both "washing" and "machine" individually, even though the compound has a specific unified meaning beyond just the components

3. **Substitutability Issues**: Replacing components with synonyms (e.g., "cleaning apparatus") would likely show artificially high similarity to "washing machine" in these models, despite semantic differences in real usage

4. **Context Independence**: The representation of "washing machine" appears relatively invariant to context, suggesting models may have a fixed (possibly component-based) representation rather than context-sensitive holistic understanding

5. **Anisotropy Effect**: Contextualised embeddings occupy narrow cone in vector space, leading to inflated similarity scores - so high similarities between "washing machine" and related terms may be artificial

6. **Better Methods Needed**: BERTRAM's approach of learning type-level vectors for whole compounds showed improvement, suggesting explicit training on compound-as-unit is beneficial

**Key Quote**: "The probing measures suggest that the standard and widely adopted composition operations display a limited ability to capture NC idiomaticity."

**Specific Example Relevance**:
- "Field work" (compositional, score 4.54): BERT P1 similarity = 0.98 with synonym
- "Wet blanket" (idiomatic, score 0.21): BERT P1 similarity = 0.77 with synonym
- "Washing machine" would likely fall in compositional range (4-5) and show similar patterns to "field work"
- BUT high similarities don't mean holistic representation - may just reflect token overlap

---

## Synthesis: Converging Evidence

### Agreement Between Papers

1. **Distributed Not Localized**: Both papers agree compositional/idiomatic meanings are NOT localized to specific layers or representations

2. **Lexical Overlap Dominates**: Both find models prioritize token/word overlap over true semantic composition

3. **Context Underutilized**: Both show context plays surprisingly limited role in distinguishing meanings

4. **Standard Architectures Inadequate**: Both conclude current transformer architectures have fundamental limitations for compositional semantics

5. **Larger/More Complex ≠ Better**: Aljaafari et al. show larger models MORE fragile; Garcia et al. show contextualised models not necessarily better than static ones

### Key Differences

1. **Level of Analysis**:
   - Aljaafari et al.: Internal activation analysis across layers
   - Garcia et al.: Final embedding similarity analysis

2. **Languages**:
   - Aljaafari et al.: English-focused with multilingual models
   - Garcia et al.: Explicit English-Portuguese comparison

3. **Intervention Type**:
   - Aljaafari et al.: Structural perturbation via pooling
   - Garcia et al.: Lexical substitution via paraphrases

### Implications for "Washing Machine"

**Converging Evidence Suggests**:

1. "Washing machine" is NOT stored as a unified holistic concept in current LLMs

2. Instead, it's represented as a distributed pattern across:
   - Multiple layers (Aljaafari et al.)
   - Component tokens with high mutual information (both papers)
   - Context-dependent but weakly-integrated features (both papers)

3. The representation is:
   - **Fragile**: Vulnerable to token-level perturbations
   - **Compositional by default**: Built from parts, not stored as whole
   - **Context-weak**: Not strongly modified by surrounding context
   - **Lexically-biased**: Similarity to "washing" and "machine" individually very high

4. This applies even though "washing machine" is a LITERAL, COMPOSITIONAL compound - suggesting even worse representation for truly idiomatic expressions

5. Architectural changes needed to improve compound representation:
   - Type-level vectors for common compounds (BERTRAM approach)
   - Training objectives that encourage semantic integration
   - Specialized architectures that support phrasal composition

### Research Recommendations

For investigating "washing machine" storage specifically:

1. Apply CAP methodology (Aljaafari et al.) specifically to household object compounds

2. Use Garcia et al.'s probing measures on "washing machine" with variants:
   - Synonyms: "laundry machine", "washer"
   - Component synonyms: "cleaning machine", "washing apparatus"
   - Individual components: "washing", "machine"

3. Compare across:
   - Different tokenizations (how many tokens for "washing machine"?)
   - Different model sizes (expect larger = more fragmented)
   - Different contexts (kitchen vs. laundromat vs. neutral)

4. Test whether explicit compound training (BERTRAM-style) improves holistic representation

5. Investigate cross-lingual patterns (is "washing machine" stored similarly to "máquina de lavar"?)
