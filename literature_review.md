# Literature Review: Where is "Washing Machine" Stored in LLMs?

## Research Area Overview

This literature review surveys work relevant to understanding how compound concepts such as "washing machine" are represented in large language models (LLMs). The central hypothesis is that LLMs may not store compound concepts as unique directions in the residual stream; instead, they may store constituent parts (e.g., "washing") and rely on context to increase the likelihood of the subsequent token ("machine"). This question sits at the intersection of three active research areas: (1) the linear representation hypothesis and concept geometry in neural networks, (2) mechanistic interpretability through sparse autoencoders, and (3) compositional semantics and multi-token concept processing in transformers.

---

## 1. Linear Representation Hypothesis and Concept Directions

### Foundational Theory

**Park et al. (2024)** — "The Linear Representation Hypothesis and the Geometry of Large Language Models" — provides the most rigorous formalization of what it means for a concept to be "stored as a direction" in an LLM. They distinguish three variants of the hypothesis:
- **Subspace hypothesis**: Concepts correspond to directions in activation space (e.g., γ("queen") − γ("king") ≈ gender direction)
- **Measurement hypothesis**: Linear probes can predict concept presence from activations
- **Intervention hypothesis**: Adding concept vectors to activations steers model behavior

They tested 27 concepts on LLaMA-2-7B using the BATS 3.0 dataset and found strong linear structure for 26/27 concepts. Critically, **the only concept that failed was the compositional relation "thing→part"**, suggesting that compositional/relational concepts may not follow the same linear representation pattern as simple attribute concepts. They also emphasize that the correct inner product for measuring concept geometry is the **causal inner product** (using the inverse covariance of unembeddings), not naive Euclidean distance. **Limitation**: Only single-token concepts were tested — multi-token compound concepts like "washing machine" remain unexplored.

**Code**: github.com/KihoPark/linear_rep_geometry

### Related Representation Work

**Gurnee & Tegmark (2023)** showed that LLMs develop linear representations of space and time, with specific directions encoding geographic coordinates and temporal information. **Merullo et al. (2023)** demonstrated that LLMs implement Word2Vec-style vector arithmetic for relational tasks, suggesting that the additive composition principle (king − man + woman = queen) may apply to some multi-token constructs. **Arora et al. (2016)** showed that polysemous word senses reside in linear superposition within word embeddings, recoverable via sparse coding — a precursor to modern SAE work.

### Implications for Compound Concepts
If "washing machine" has a unique direction, it should be recoverable via probing or steering. If it is compositionally derived, the direction should approximately equal some function of the "washing" and "machine" directions. The failure of "thing→part" relations in Park et al. suggests that relational/compositional concepts may require a different framework.

---

## 2. Superposition and Polysemanticity

### Theoretical Foundation

**Elhage et al. (2022)** — "Toy Models of Superposition" — established the theoretical framework for understanding how neural networks represent more features than they have dimensions. Key findings:
- **Superposition**: Models store sparse features as nearly-orthogonal directions, tolerating interference between features
- **Phase transitions**: Whether features use superposition depends on their sparsity and importance — sparse features (like "washing machine" occurrences) are MORE likely to be in superposition
- **Geometric structure**: Features arrange into regular polytopes (triangles, pentagons, tetrahedra), following solutions to the Thomson problem
- **Implication**: "Washing machine" as a concept is almost certainly NOT stored in a dedicated neuron but rather as a direction in superposition with other concepts

**Scherlis et al. (2022)** — "Polysemanticity and Capacity in Neural Networks" — further analyzed why neurons become polysemantic, finding it emerges naturally from capacity constraints.

### Practical Consequence
Any experiment probing for "washing machine" must account for superposition. Looking at individual neurons will fail; instead, we need directional analysis (probing classifiers, SAE features, or concept vectors in the residual stream).

---

## 3. Sparse Autoencoders for Feature Discovery

### Core SAE Work

**Cunningham et al. (2023)** introduced using SAEs to decompose polysemantic activations into interpretable, monosemantic features. **Gao et al. (2024)** scaled SAEs significantly and established evaluation metrics. **Rajamanoharan et al. (2024)** introduced Gated SAEs for improved dictionary learning. **Bussmann et al. (2024)** proposed BatchTopK SAEs. **Lieberum et al. (2024)** released Gemma Scope, providing pre-trained SAEs across all layers of Gemma-2 models. **He et al. (2024)** released Llama Scope with 256 SAEs across Llama-3.1-8B.

### Feature Absorption Problem

**Chanin et al. (2024)** — "A is for Absorption" — identified a critical problem for using SAEs to study compound concepts:
- **Feature absorption**: When features form hierarchies (e.g., "washing machine" → "appliance"), SAEs optimize for sparsity by absorbing parent features into child features, creating unreliable classifiers with arbitrary false negatives
- **Universal**: Every SAE tested (across Gemma-2-2B, Qwen2, Llama 3.2, with hundreds of SAEs) exhibited absorption
- **No fix**: No hyperparameter configuration solves it; it's inherent to the L1-sparsity objective
- **Implication**: SAE features for compound concepts may be unreliable — a "washing machine" feature might fail to fire on arbitrary instances because more specific features absorb it

**Code**: github.com/lasr-spelling/sae-spelling

### Feature Hedging

**Chanin et al. (2025)** — "Feature Hedging" — showed that if SAEs are narrower than the true number of features and features are correlated, SAEs merge components into composite representations. This could cause compound concepts to be merged with semantically related concepts.

### Dark Matter in SAEs

**Engels et al. (2024)** analyzed the "dark matter" of SAEs — unexplained variance that SAEs fail to capture. This is relevant because compound concept representations may partially reside in this unexplained variance.

### Practical Guidance for Experiments
- Use multiple SAE widths (16k, 65k, 128k) to check for feature splitting
- Verify features with ablation studies, not just max-activating examples
- Check for absorption: does a "washing machine" feature fire on all instances, or does it get absorbed into sub-features?
- Consider using the footprints/token erasure approach instead of or alongside SAEs

---

## 4. Multi-Token Concept Processing

### Token Erasure (Most Directly Relevant)

**Feucht et al. (2024)** — "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs" — is the most directly relevant work. Key findings:

- **Token erasure**: At the last token position of multi-token words/entities, information about preceding tokens is rapidly "erased" (transformed) in early layers (1–9)
- **Implicit vocabulary**: LLMs develop internal representations for semantically meaningful units beyond their BPE token vocabulary — an "implicit vocabulary" of multi-token items
- **Mechanism**: Information transforms from token-level encoding to lexical-level encoding at the last token position, due to the autoregressive constraint (the model can only represent "Space Needle" after seeing "Needle")
- **Models**: Tested on Llama-2-7B and Llama-3-8B; similar patterns despite very different tokenizations
- **Measurement**: Linear probes trained on hidden states; accuracy for predicting preceding tokens drops from ~100% (layer 0) to ~20% (layer 9) at the last position of multi-token entities

**Direct application to "washing machine"**: If "washing machine" is treated as a lexical unit by the model, we should observe token erasure at the "machine" token position — the hidden state for "machine" would "forget" the "washing" token information in early layers as it assembles the compound concept representation. Comparing this to compositional phrases like "red machine" (which should show less erasure) would test whether the model treats "washing machine" as a unified concept.

**Code**: footprints.baulab.info, github.com/sfeucht/footprints

### FFN Layer Concept Promotion

**Geva et al. (2022)** — "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space" — showed that:
- FFN layers build predictions through sub-updates that "promote" specific tokens/concepts
- Each value vector in the FFN encodes interpretable concepts
- Predictions are built gradually across layers via promotion rather than elimination
- **Relevance**: After processing "washing", FFN layers may promote "machine" through value vectors that encode the "appliance" concept, gradually increasing P("machine") across layers

### Compositional Probe Evidence

**Ormerod et al. (2024)** — "How Is a 'Kitchen Chair' like a 'Farm Horse'?" — provides the most directly relevant probing study for compound noun semantics:
- Used Representational Similarity Analysis (RSA) to compare transformer representations with human judgments of compound noun thematic relations
- **Key experiment**: Compared representations when head+modifier are processed **together** (as compound) vs. **separately** (in different sentences)
- **Results**: RoBERTa and DistilRoBERTa show significantly stronger compound semantic representations in the "together" condition — evidence of genuine compositional processing, not just co-occurrence memorization
- **XLNet**: Relied more on distributional co-occurrence information from individual words
- **Datasets**: 300 compounds from Gagné (2001), 60 compounds with fine-grained 18-dimensional relation vectors from Devereux & Costello (2005)
- **Limitation**: Used BERT-family masked language models, not autoregressive GPT-family models

---

## 5. Compositionality and Idiomaticity

### Token Compositionality Analysis

**Aljaafari et al. (2024)** — "Interpreting token compositionality in LLMs" — introduced Constituent-Aware Pooling (CAP) to test compositional processing:
- **Finding**: No specific layer integrates tokens into unified semantic representations based on constituent parts
- **Fragmented processing**: Information is distributed across layers with long dependency paths
- **Larger models are worse**: Larger models exhibit greater information dispersion and fragmentation, suggesting scaling doesn't improve compositional integration
- **Implication**: LLMs may process "washing machine" through fragmented, distributed mechanisms rather than composing a unified compound representation at any single layer

### Idiomaticity Probing

**Garcia et al. (2021)** — "Probing for idiomaticity in vector space models" — tested whether BERT captures idiomatic vs. compositional meanings of noun compounds:
- **Finding**: Contextualised models fail to accurately capture idiomaticity
- Models prioritize lexical overlap over semantic understanding
- Context plays a surprisingly limited role in disambiguating compositional vs. idiomatic readings
- Developed the Noun Compound Senses (NCS) dataset with 280 English and 180 Portuguese compounds

### Transformer Compositional Processing

**Dankers et al. (2022)** — "Can Transformer be Too Compositional?" — analyzed idiom processing in NMT:
- Transformers tend to over-generate compositional translations of idioms
- Models struggle with non-compositional multi-word expressions
- Suggests a bias toward compositional processing even when inappropriate

**Buijtelaar et al. (2023)** — "A Psycholinguistic Analysis of BERT's Representations of Compounds" — found that BERT's representations of compounds align with human semantic intuitions about compound meaning, but the alignment varies by layer.

---

## 6. Common Methodologies and Baselines

### Probing Methods
- **Linear probes**: Train linear classifiers on hidden states to predict concept properties (Park et al., Feucht et al.)
- **RSA (Representational Similarity Analysis)**: Compare model RDMs with human judgment RDMs (Ormerod et al.)
- **Causal interventions**: Patch/ablate hidden states to test causal role of representations (Turner et al., Hernandez et al.)
- **SAE feature analysis**: Decompose activations using SAEs and inspect features (Cunningham et al., Chanin et al.)

### Standard Baselines
- **Random baseline**: Permuted/shuffled labels for probing
- **Separate processing baseline**: Process constituent words in separate contexts (Ormerod et al.'s "Separate" condition)
- **Compositional control**: Compare compound nouns to clearly compositional phrases (e.g., "washing machine" vs. "red machine")
- **Different tokenizations**: Compare models with different BPE vocabularies to control for tokenization effects

### Evaluation Metrics
- **Probe accuracy**: How well can a linear probe recover concept information
- **Erasure score (ψ)**: Quantifies token-level information loss (Feucht et al.)
- **RSA correlation**: Second-order correlation between model and human RDMs
- **Steering effect size**: Change in model behavior when adding concept vectors
- **SAE feature F1**: How reliably an SAE feature tracks a target concept

---

## 7. Datasets in the Literature

| Dataset | Used In | Type | Size | Relevance |
|---------|---------|------|------|-----------|
| Gagné (2001) compounds | Ormerod et al. | 300 compounds, 16 relations | 300 | High |
| Devereux & Costello (2005) | Ormerod et al. | 60 compounds, 18-dim relation vectors | 60 | High |
| NCS (Garcia et al.) | Garcia et al. | Compounds with compositionality ratings | 280 EN, 180 PT | High |
| COUNTERFACT | Feucht et al. | Factual prompts with entities | 12K+ | Medium |
| MAGPIE | Idiom research | Idiom instances with labels | 56K | Medium |
| BATS 3.0 | Park et al. | Analogy/relation pairs | ~40 relations | Medium |
| The Pile | Feucht et al. | General text (probe training) | 800GB | Medium |

---

## 8. Gaps and Opportunities

### Key Gap: No Study Directly Tests Compound Noun Directions in Residual Stream
- Park et al. tested single-token concepts but not multi-token compounds
- Ormerod et al. used masked LMs (BERT), not autoregressive models
- Feucht et al. focused on named entities, not common compound nouns
- No study has used SAEs to find features specifically for compound nouns like "washing machine"

### Opportunity 1: Token Erasure for Compound Nouns
Adapt Feucht et al.'s methodology to test whether "washing machine" shows erasure patterns (suggesting unified representation) or maintains token-level information (suggesting compositional processing). Compare to compositional controls.

### Opportunity 2: SAE Feature Analysis of Compound Concepts
Use pre-trained SAEs (Gemma Scope, Llama Scope) to search for features that specifically activate for "washing machine" as a compound, vs. features for "washing" and "machine" separately. Test for absorption effects.

### Opportunity 3: Directional Probing with Causal Inner Product
Apply Park et al.'s framework to multi-token compounds: estimate the causal inner product, find concept directions for "washing machine" vs. constituent words, and test whether the compound direction is linearly independent of constituent directions.

### Opportunity 4: Compositional vs. Holistic Representation Spectrum
Build a controlled dataset of compound nouns varying in compositionality (from "coffee table" to "hot dog") and measure representation properties across this spectrum — erasure scores, SAE feature patterns, probe accuracy, and steering effects.

---

## 9. Recommendations for Experiments

### Primary Methodology: Token Erasure Analysis
1. Use Feucht et al.'s linear probes on Llama-2-7B or Llama-3-8B
2. Measure erasure scores for compound nouns varying in compositionality
3. Compare: "washing machine" (compound) vs. "red machine" (compositional) vs. "hot dog" (idiomatic)

### Secondary Methodology: SAE Feature Search
1. Use pre-trained SAEs (Gemma Scope or Llama Scope)
2. Search for features that activate on compound nouns
3. Test for feature absorption: does the compound feature reliably fire?
4. Compare feature patterns across compositionality spectrum

### Tertiary Methodology: Concept Direction Analysis
1. Collect contexts where "washing machine" appears vs. related contexts
2. Estimate concept directions using mean-difference or probing
3. Test: is the "washing machine" direction independent of "washing" and "machine" directions?
4. Measure causal effects of steering with compound vs. constituent vectors

### Recommended Models
- **Llama-2-7B** or **Llama-3-8B**: Best supported by existing tools and pre-trained probes
- **Gemma-2-2B**: Best SAE coverage via Gemma Scope
- **GPT-2-small**: Good for initial experiments due to size and tooling support

### Recommended Metrics
1. Erasure score (ψ) — quantifies lexicality
2. Probe accuracy across layers — maps representation formation
3. SAE feature reliability (F1) — tests feature absorption
4. Steering effect size — causal validation
5. Cosine similarity between compound and constituent directions

---

## References

1. Aljaafari, N. et al. (2024). Interpreting token compositionality in LLMs. arXiv:2410.12924.
2. Arora, S. et al. (2016). Linear Algebraic Structure of Word Senses. arXiv:1601.03764.
3. Buijtelaar, L. et al. (2023). A Psycholinguistic Analysis of BERT's Representations of Compounds.
4. Bussmann, B. et al. (2024). BatchTopK Sparse Autoencoders. arXiv:2409.06981.
5. Chanin, D. et al. (2024). A is for Absorption. arXiv:2409.14507.
6. Chanin, D. et al. (2025). Feature Hedging. arXiv:2503.01370.
7. Cunningham, H. et al. (2023). Sparse Autoencoders Find Highly Interpretable Features. arXiv:2309.08600.
8. Dankers, V. et al. (2022). Can Transformer be Too Compositional? arXiv:2205.15301.
9. Elhage, N. et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
10. Engels, J. et al. (2024). Decomposing The Dark Matter of SAEs. arXiv:2410.14670.
11. Feucht, S. et al. (2024). Token Erasure as a Footprint of Implicit Vocabulary Items. arXiv:2406.20086.
12. Gao, L. et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.
13. Garcia, M. et al. (2021). Probing for idiomaticity in vector space models. EACL 2021.
14. Geva, M. et al. (2022). Transformer Feed-Forward Layers Build Predictions. arXiv:2203.14680.
15. Gurnee, W. & Tegmark, M. (2023). Language Models Represent Space and Time. arXiv:2310.02207.
16. He, Z. et al. (2024). Llama Scope. arXiv:2410.20526.
17. Hernandez, E. et al. (2023). Linearity of Relation Decoding. arXiv:2308.09124.
18. Lieberum, T. et al. (2024). Gemma Scope. arXiv:2408.05147.
19. Merullo, J. et al. (2023). Language Models Implement Simple Word2Vec-style Vector Arithmetic. arXiv:2305.16130.
20. Ormerod, M. et al. (2024). How Is a "Kitchen Chair" like a "Farm Horse"? Computational Linguistics, 50(1).
21. Park, K. et al. (2024). The Linear Representation Hypothesis. arXiv:2311.03658.
22. Rajamanoharan, S. et al. (2024). Improving Dictionary Learning with Gated SAEs. arXiv:2404.16014.
23. Scherlis, A. et al. (2022). Polysemanticity and Capacity in Neural Networks. arXiv:2210.01892.
24. Turner, A. et al. (2023). Steering Language Models With Activation Engineering. arXiv:2308.10248.
