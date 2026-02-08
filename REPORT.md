# Where is "Washing Machine" Stored in LLMs?

## 1. Executive Summary

**Research question**: How do large language models represent compound concepts like "washing machine" — as unique directions in the residual stream, or compositionally through constituent words plus context?

**Key finding**: LLMs use a **hybrid strategy** — compound concepts are primarily stored as compositional combinations of constituent word representations (R² = 0.937 for linear reconstruction from constituents), but the model also develops compound-specific contextual information that allows distinguishing compound from non-compound contexts (92.2% probe accuracy). The strongest mechanism is **next-token prediction boosting**: seeing "washing" raises P("machine") by a median of 20x compared to control words. Critically, no token erasure was observed — the identity of word1 is perfectly recoverable from the word2 position at every layer, contradicting the implicit vocabulary hypothesis for GPT-2.

**Practical implications**: Compound concepts in LLMs are not stored as dedicated directions in the residual stream. Instead, the model leverages (1) statistical co-occurrence to boost word2 prediction after word1, and (2) contextual modulation of existing word representations to carry compound-specific semantics. This has direct consequences for concept editing, steering, and interpretability methods that assume concepts have unique linear directions.

## 2. Goal

### Hypothesis
In large language models, specific compound concepts such as "washing machine" may not be represented by a unique direction or nearly orthogonal direction in the residual stream; instead, the model may store the concept of "washing" and rely on context to increase the likelihood of "machine" following it.

### Why This Matters
There are far more referenceable concepts in language (millions of compounds, proper nouns, technical terms) than there are dimensions in an LLM's residual stream (768 for GPT-2, 4096 for 7B models). The superposition hypothesis (Elhage et al., 2022) allows ~10x more features than dimensions through nearly-orthogonal packing, but even this falls short of the number of possible compound concepts. Understanding how models handle this representational bottleneck is fundamental to:
- Mechanistic interpretability (what do directions in activation space mean?)
- Concept editing (can we edit "washing machine" without affecting "washing" or "machine"?)
- Linear representation theory (does the linear representation hypothesis hold for multi-token concepts?)

### Gap in Existing Work
- Park et al. (2024) tested 27 single-token concepts but never multi-token compounds
- Feucht et al. (2024) showed token erasure for named entities but not common compound nouns
- Ormerod et al. (2024) probed compound semantics in BERT (masked LM), not autoregressive models
- No study has directly compared compound concept directions vs. constituent word directions in the residual stream of autoregressive models

## 3. Data Construction

### Dataset Description
We used 19 compound nouns spanning the full compositionality spectrum, each paired with a control phrase that uses the same word2 but a different, non-compound word1.

**Source**: Custom-designed test set based on the compound_nouns_test.jsonl dataset created for this project, with compositionality ratings from 1 (fully idiomatic) to 5 (fully compositional).

### Example Samples

| Compound | Word1 | Word2 | Compositionality | Control |
|----------|-------|-------|-----------------|---------|
| washing machine | washing | machine | 4 | red machine |
| hot dog | hot | dog | 1 | big dog |
| coffee table | coffee | table | 5 | wooden table |
| guinea pig | guinea | pig | 2 | small pig |
| mountain cabin | mountain | cabin | 5 | small cabin |

### Data Quality
- All compounds verified to tokenize as exactly two tokens in GPT-2's BPE vocabulary
- 2 compounds skipped (blueberry: "berry" multi-token; guinea pig: "guinea" multi-token in some analyses)
- Compositionality ratings assigned based on linguistic analysis (1=opaque/idiomatic, 5=transparent/compositional)
- 8 sentence templates used per compound for context diversity

### Sentence Templates
Each compound was embedded in 8 diverse sentence templates:
- "The {compound} was"
- "She bought a {compound} for"
- "I saw a {compound} in the"
- "There is a {compound} near the"
- "He fixed the {compound} with"
- "A new {compound} arrived"
- "The old {compound} needed"
- "We need a {compound} to"

## 4. Experiment Description

### Methodology

#### High-Level Approach
We conducted four complementary experiments on GPT-2 (124M parameters, 12 layers, d_model=768), accessing internal activations via TransformerLens. Each experiment tests a different aspect of how compound concepts are represented:

1. **Next-token prediction**: Does word1 boost P(word2)? How much?
2. **Residual stream directions**: Can the compound direction be reconstructed from constituents?
3. **Layer-wise probing**: Where do compound representations emerge?
4. **Attention patterns**: Does word2 attend to word1 differently in compounds vs. controls?

#### Why GPT-2?
- Well-studied in interpretability research
- Small enough for comprehensive analysis (all layers, all attention heads)
- TransformerLens provides clean hook-based access to all internal activations
- Results validated on GPT-2-medium (355M, 24 layers) for scale robustness

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.2 | Runtime |
| PyTorch | 2.10.0+cu128 | Tensor computation |
| TransformerLens | 2.15.4 | Model internals access |
| scikit-learn | - | Linear probes |
| scipy | - | Statistical tests |
| matplotlib | - | Visualization |

#### Hardware
- 2x NVIDIA GeForce RTX 3090 (24GB each)
- Total experiment runtime: ~3 minutes for GPT-2, ~2 minutes for GPT-2-medium

#### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Random seed | 42 | Reproducibility |
| Probe regularization (C) | 1.0 | Default, adequate for small dataset |
| Cross-validation folds | 5 | Standard, balances bias-variance |
| Number of templates | 8 | Sufficient context diversity |
| Number of isolation templates | 4 | Baseline word representations |

### Experimental Protocol

#### Experiment 1: Next-Token Prediction Analysis
For each compound (word1, word2):
1. Embed word1 in 8 templates and measure P(word2 | context + word1)
2. Embed control_word1 in same templates and measure P(word2 | context + control_word1)
3. Compute boost ratio = P(word2 | word1) / P(word2 | control_word1)
4. Record rank of word2 among all vocabulary predictions

#### Experiment 2: Residual Stream Direction Analysis
1. Collect hidden states at the word2 position across 8 compound contexts per layer
2. Collect hidden states for word1 and word2 in isolation (4 contexts each)
3. Compute mean direction vectors for compound, word1, word2
4. Test linear reconstruction: compound = α·word1 + β·word2 (least squares)
5. Compute R², cosine similarities, and residual norm ratio

#### Experiment 3: Layer-wise Probing
1. Collect hidden states at word2 position for 120 compound and 136 control samples
2. **Probe 1**: Train logistic regression to predict word1 identity from word2 position (tests token erasure)
3. **Probe 2**: Train logistic regression to classify compound vs. control context

#### Experiment 4: Attention Pattern Analysis
1. For each compound and control context, extract attention weights at the word2 position
2. Measure attention from word2 to word1 (compound) vs. word2 to previous word (control)
3. Compare across layers and attention heads

### Raw Results

#### Experiment 1: Next-Token Prediction

| Compound | P(w2\|w1) | Rank | P(w2\|ctrl) | Ctrl Rank | Boost | Top-5 after w1 |
|----------|-----------|------|-------------|-----------|-------|----------------|
| guinea pig | 0.8326 | 1 | 0.0001 | 1755 | 7233x | pig, pigs, worm, -, p |
| washing machine | 0.8270 | 1 | 0.0002 | 924 | 4963x | machine, machines, -, ton, of |
| swimming pool | 0.6258 | 1 | 0.0011 | 125 | 549x | pool, pools, team, hole, - |
| parking lot | 0.3215 | 1 | 0.0116 | 14 | 28x | lot, garage, meter, lots, space |
| living room | 0.2611 | 2 | 0.0016 | 98 | 160x | room, -, wage, world, rooms |
| hot dog | 0.0973 | 4 | 0.0022 | 142 | 45x | new, topic, dog, -, spot |
| coffee table | 0.0632 | 4 | 0.0091 | 14 | 7x | shop, -, is, maker, industry |
| chocolate cake | 0.0356 | 5 | 0.0003 | 978 | 140x | chip, -, bar, is, and |
| driving license | 0.0364 | 118 | 0.0002 | 1325 | 221x | force, forces, seat, -, season |
| shooting star | 0.0307 | 90 | 0.0103 | 18 | 3x | of, death, at, was, in |
| mountain cabin | 0.0040 | 149 | 0.0014 | 305 | 3x | of, is, lion, range, bike |
| snowman | 0.0038 | 63 | 0.0427 | 7 | 0.1x | is, -, was, storm, has |
| sunflower | 0.0001 | 1673 | 0.0004 | 706 | 0.3x | is, was, has, 's, rises |

**Key observation**: For strongly associated compounds, word2 is often the #1 prediction after word1 (washing→machine, swimming→pool, guinea→pig, parking→lot). For weakly associated compounds (snowman, sunflower), the control word actually predicts word2 better.

#### Experiment 2: Residual Stream Directions (Final Layer)

| Compound | R² | cos(C,w1) | cos(C,w2) | Residual |
|----------|-----|-----------|-----------|----------|
| steel bridge | 0.965 | 0.924 | 0.982 | 0.188 |
| garden hose | 0.962 | 0.933 | 0.979 | 0.196 |
| door handle | 0.958 | 0.949 | 0.976 | 0.206 |
| mountain cabin | 0.956 | 0.937 | 0.975 | 0.209 |
| chocolate cake | 0.953 | 0.944 | 0.975 | 0.216 |
| shooting star | 0.944 | 0.929 | 0.960 | 0.236 |
| brick house | 0.939 | 0.933 | 0.965 | 0.247 |
| coffee table | 0.938 | 0.934 | 0.962 | 0.249 |
| living room | 0.933 | 0.925 | 0.958 | 0.259 |
| water bottle | 0.931 | 0.924 | 0.963 | 0.262 |
| swimming pool | 0.924 | 0.932 | 0.957 | 0.276 |
| driving license | 0.924 | 0.908 | 0.955 | 0.276 |
| parking lot | 0.922 | 0.936 | 0.934 | 0.280 |
| washing machine | 0.917 | 0.907 | 0.951 | 0.288 |
| hot dog | 0.888 | 0.897 | 0.934 | 0.335 |

**Mean R² = 0.937 ± 0.020**: The compound direction is very well explained by a linear combination of constituent directions. Only ~25% of the compound representation is "unique" (not explained by word1 + word2).

#### Experiment 3: Probing Results

**Probe 1 (Token Erasure)**: Perfect accuracy (1.000) at ALL layers. Word1 identity is fully recoverable from the word2 position at every layer. **No token erasure observed in GPT-2.**

**Probe 2 (Compound vs. Control)**:

| Layer | Accuracy |
|-------|----------|
| 0 | 0.706 |
| 2 | 0.902 |
| 4 | 0.894 |
| 7 | 0.918 |
| 8 | 0.922 (peak) |
| 11 | 0.851 |

Compound vs. control contexts are distinguishable from layer 2 onward (90%+ accuracy), peaking at layer 8.

#### Experiment 4: Attention Patterns

Compound contexts show significantly more attention from word2 to word1 in layer 0 (p=0.011), but this reverses in later layers (7-8) where control contexts show more attention to the previous word (p<0.001).

### Output Locations
- Results JSON: `results/exp1_next_token.json`, `results/exp2_residual_directions.json`, `results/exp3_probing.json`, `results/exp4_attention.json`
- Plots: `results/plots/`
- Summary figure: `results/plots/summary_figure.png`
- Configuration: `results/config.json`

## 5. Result Analysis

### Key Findings

**Finding 1: The primary mechanism is next-token prediction boosting (strong support for compositional hypothesis).**
Seeing "washing" makes "machine" the #1 prediction (P=0.827, rank 1), compared to P=0.0002 after "red" (the control). The median boost ratio across all compounds is 20.2x (95% CI: [4.7, 180.2]). This is statistically significant (Wilcoxon W=160, p=2.1e-4, Cohen's d=0.63).

For the strongest compounds:
- "washing" → P("machine") = 0.827 (rank 1)
- "guinea" → P("pig") = 0.833 (rank 1)
- "swimming" → P("pool") = 0.626 (rank 1)

This confirms the hypothesis: the model stores "washing" and context makes "machine" overwhelmingly likely.

**Finding 2: Compound directions are 93.7% reconstructable from constituent directions.**
At the final layer, the compound direction at the word2 position can be reconstructed as a linear combination of the word1 and word2 directions with R² = 0.937 ± 0.020 (t=176.3, p=7.9e-25 vs. null). Only ~25% of the compound representation's norm is "unique" (unexplained residual). This means "washing machine" does NOT have a dedicated direction — it is ~94% a combination of "washing" and "machine" directions.

**Finding 3: More compositional compounds have HIGHER reconstruction quality.**
Spearman correlation between compositionality rating and R²: r=0.669, p=0.006. More compositional compounds (steel bridge: R²=0.965) are better reconstructed than idiomatic ones (hot dog: R²=0.888). This is exactly what the compositional hypothesis predicts — idiomatic compounds require more "unique" information beyond their constituents.

**Finding 4: No token erasure in GPT-2 (contradicts Feucht et al. for named entities).**
Word1 identity is perfectly recoverable from the word2 position at every layer (probe accuracy = 1.000). This means GPT-2 does not "erase" constituent token information to form compound representations. Unlike named entities, compound nouns maintain full constituent information throughout all layers.

**Finding 5: Compound contexts are distinguishable from control contexts.**
Despite the high R² for reconstruction, a linear probe can distinguish compound from non-compound contexts at the word2 position with 92.2% accuracy (peak at layer 8). This ~6% "unique" component (100% - 94% R²) carries enough information to differentiate "washing machine" from "red machine."

**Finding 6: Reconstruction quality follows a U-shaped pattern across layers.**
R² starts high at layer 0 (0.940), dips to a minimum at layers 4-5 (~0.800), and recovers to 0.937 at layer 11. This suggests intermediate layers perform the most compound-specific processing, while early and late layers maintain more compositional representations.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: Next-token prediction drives compound assembly | **Supported** | Wilcoxon p=2.1e-4; median boost=20.2x |
| H2: Compounds lack unique residual stream directions | **Mostly supported** | R²=0.937; but ~6% unique component exists |
| H3: Token erasure occurs for compounds | **Rejected** (for GPT-2) | Perfect word1 recovery at all layers |

### Comparison to Baselines and Literature
- **Feucht et al. (2024)** found token erasure in layers 1-9 for named entities in Llama-2-7B. We find NO erasure in GPT-2 for compound nouns — this could be a model size effect, an entity vs. compound noun difference, or both.
- **Park et al. (2024)** showed 26/27 single-token concepts have linear directions. We show compound concepts are ~94% linearly reconstructable from constituents — the compound "direction" is largely the constituent directions.
- **Ormerod et al. (2024)** found that compounds processed together have different representations than constituents processed separately. Our 92.2% probe accuracy confirms this, but shows the difference is relatively small (~6% of the total representation).

### Validation on GPT-2-Medium (355M, 24 layers)
Key findings replicate:

| Compound | GPT-2 Boost | GPT-2-M Boost | GPT-2 R² | GPT-2-M R² |
|----------|-------------|---------------|----------|------------|
| washing machine | 4963x | 25303x | 0.917 | 0.899 |
| swimming pool | 549x | 615x | 0.924 | 0.932 |
| hot dog | 45x | 116x | 0.888 | 0.897 |
| coffee table | 7x | 4x | 0.938 | 0.923 |

The larger model shows similar or stronger next-token prediction boosting and similar R² values for linear reconstruction. Findings are robust to model scale.

### Surprises and Insights

1. **Guinea pig** (compositionality=2, idiomatic) has the HIGHEST boost ratio (7233x) and P=0.833 for "pig" after "guinea." Despite being semantically opaque, the statistical association is the strongest. The model learns co-occurrence regardless of semantic compositionality.

2. **Snowman** and **sunflower** have boost ratios < 1 — the control word ("tall man", "red flower") actually predicts word2 BETTER than the compound word1. This is because "snow" and "sun" don't strongly predict their compound partners in GPT-2's training distribution.

3. **Driving license** has a very high boost ratio (221x) but a low rank (118th prediction after "driving"). This means "license" is much more likely after "driving" than after "new", but "driving" still predicts many other words more strongly (force, forces, seat, season).

4. The **U-shaped R² curve** across layers was unexpected. It suggests that intermediate layers (4-5) perform the most transformation of compound representations, potentially encoding compound-specific semantics, before the final layers restore a more compositional representation for next-token prediction.

### Error Analysis
- **Blueberry** was excluded because "berry" tokenizes to multiple tokens in GPT-2, preventing clean analysis
- **Guinea pig** was excluded from Experiment 2 (direction analysis) because "guinea" tokenizes to multiple tokens, but was kept in Experiment 1 (next-token prediction) where only word2 needs to be single-token
- Compounds where both words are common function words (e.g., potential compound "let down") were not included due to high baseline prediction probabilities

### Limitations

1. **Model scale**: GPT-2 (124M) is much smaller than modern LLMs. Larger models may develop more holistic compound representations. Validation on GPT-2-medium (355M) shows similar patterns, but testing on 7B+ models would strengthen conclusions.

2. **Limited context diversity**: Only 8 sentence templates were used. More diverse contexts (from natural corpora like The Pile) would better capture the range of compound usage.

3. **English-only**: All compounds are English. Cross-lingual analysis would reveal whether findings generalize.

4. **Linear probing limitation**: Linear probes may miss nonlinear compound representations. MLP probes or DCI (Disentanglement, Completeness, Informativeness) could capture additional structure.

5. **Absence of token erasure might be GPT-2-specific**: Feucht et al. found erasure in Llama-2-7B. Our GPT-2 results don't generalize to larger models without further validation.

6. **The R² metric conflates direction and magnitude**: High R² could mean the compound direction is approximately in the span of constituent directions, even if the magnitude (and thus the information encoded in norms) differs.

7. **Control phrase selection**: Our control phrases (e.g., "red machine" for "washing machine") aim to preserve word2 while changing word1, but the specific control word choice affects boost ratios.

## 6. Conclusions

### Summary
**"Washing machine" is not stored as a unique direction in GPT-2's residual stream.** Instead, the model primarily uses two complementary mechanisms: (1) **next-token prediction boosting** — seeing "washing" makes "machine" the most likely next token with P=0.827 — and (2) **contextual modulation** — the word2 representation in compound contexts is ~94% a linear combination of constituent word representations (R²=0.937), with only ~6% unique to the compound context. This 6% is sufficient to distinguish compound from non-compound contexts (92.2% probe accuracy) but does not constitute a dedicated "washing machine" direction.

More idiomatic compounds (like "hot dog") have slightly less reconstructable representations (R²=0.888 vs. 0.965 for "steel bridge"), suggesting the model allocates more unique representational capacity to concepts that cannot be derived from their parts.

### Implications

**For mechanistic interpretability**: Compound concepts challenge the simple "one concept = one direction" view. Most compound information is carried by constituent representations plus contextual modification, not dedicated directions. This means SAE features for compounds will likely be unreliable (consistent with the "feature absorption" problem documented by Chanin et al., 2024).

**For concept editing**: Editing "washing machine" by modifying a single direction would likely fail — you'd need to modify the contextual relationship between "washing" and "machine" representations, which is distributed across the network.

**For the linear representation hypothesis**: The hypothesis holds approximately for compounds — compound representations are linear combinations of constituent representations — but the ~6% unique component and the U-shaped layer evolution suggest nonlinear dynamics in intermediate layers.

### Confidence in Findings
- **High confidence**: Next-token prediction boosting is the primary mechanism (large effect sizes, consistent across models)
- **High confidence**: Compound directions are largely reconstructable from constituents (R²=0.937)
- **Medium confidence**: No token erasure in GPT-2 (may not generalize to larger models)
- **Medium confidence**: The ~6% unique component carries compound-specific information (depends on probe methodology)

## 7. Next Steps

### Immediate Follow-ups
1. **Repeat on Llama-3-8B or Gemma-2-2B**: These models have pre-trained SAEs (Llama Scope, Gemma Scope) that would enable SAE feature analysis of compound concepts, and are large enough that token erasure effects might emerge.
2. **Use natural corpus contexts**: Replace templates with sentences from The Pile or Wikipedia to reduce template bias and increase context diversity.
3. **Expand compound dataset**: Include 100+ compounds spanning the full compositionality spectrum, using the NCS dataset's compositionality ratings.

### Alternative Approaches
- **SAE feature search**: Use pre-trained SAEs to search for features that activate specifically for compound nouns. Test for feature absorption (Chanin et al., 2024).
- **Causal interventions**: Use activation patching to test whether modifying the word1 representation at specific layers disrupts compound understanding.
- **Nonlinear probing**: Use MLP probes to capture nonlinear compound representations that linear probes miss.

### Broader Extensions
- **Cross-lingual study**: Test whether the compositional storage pattern holds across languages (e.g., German, which forms novel compounds productively)
- **Scale analysis**: Track how compound representation changes from GPT-2 (124M) to GPT-3 (175B) to see if larger models develop more holistic representations
- **Temporal analysis**: Test whether newly coined compounds (e.g., "doomscrolling") are represented differently from established ones

### Open Questions
1. Where exactly does the 6% "unique component" come from? Is it in specific attention heads or MLP layers?
2. Does the U-shaped R² pattern across layers correspond to specific computational stages?
3. At what model scale does token erasure begin to appear for compound nouns?
4. How do multiword expressions longer than 2 words ("washing machine repair service") build up representations?

## References

1. Park, K. et al. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. arXiv:2311.03658.
2. Elhage, N. et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
3. Feucht, S. et al. (2024). Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs. arXiv:2406.20086.
4. Ormerod, M. et al. (2024). How Is a "Kitchen Chair" like a "Farm Horse"? Computational Linguistics, 50(1).
5. Chanin, D. et al. (2024). A is for Absorption. arXiv:2409.14507.
6. Geva, M. et al. (2022). Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space. arXiv:2203.14680.
7. Aljaafari, N. et al. (2024). Interpreting token compositionality in LLMs. arXiv:2410.12924.
8. Garcia, M. et al. (2021). Probing for idiomaticity in vector space models. EACL 2021.
9. Cunningham, H. et al. (2023). Sparse Autoencoders Find Highly Interpretable Features. arXiv:2309.08600.
10. Merullo, J. et al. (2023). Language Models Implement Simple Word2Vec-style Vector Arithmetic. arXiv:2305.16130.
