# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Where is Washing Machine Stored in LLMs?", including papers, datasets, and code repositories.

---

## Papers
**Total papers downloaded: 36**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs | Feucht et al. | 2024 | `2406.20086_token_erasure_implicit_vocabulary.pdf` | **Most relevant** — studies how multi-token concepts are assembled in early layers via "token erasure" |
| 2 | How Is a "Kitchen Chair" like a "Farm Horse"? | Ormerod et al. | 2024 | `ormerod2024_kitchen_chair_compound_semantics.pdf` | **Directly relevant** — probes compound noun semantics in transformers using RSA |
| 3 | The Linear Representation Hypothesis | Park et al. | 2024 | `2311.03658_linear_representation_hypothesis.pdf` | Formalizes concept directions; tested 27 concepts on LLaMA-2-7B |
| 4 | Toy Models of Superposition | Elhage et al. | 2022 | `2209.10652_toy_models_superposition.pdf` | Theoretical foundation for feature superposition in neural networks |
| 5 | A is for Absorption | Chanin et al. | 2024 | `2409.14507_absorption_feature_splitting.pdf` | Feature absorption problem in SAEs — hierarchical features are unreliable |
| 6 | Sparse Autoencoders Find Highly Interpretable Features | Cunningham et al. | 2023 | `2309.08600_sparse_autoencoders_interpretable.pdf` | Foundational SAE interpretability paper (838 citations) |
| 7 | Scaling and evaluating sparse autoencoders | Gao et al. | 2024 | `2406.04093_scaling_evaluating_sae.pdf` | Large-scale SAE training and evaluation (313 citations) |
| 8 | Transformer FF Layers Build Predictions by Promoting Concepts | Geva et al. | 2022 | `2203.14680_ff_layers_promoting_concepts.pdf` | How FFN layers promote next-token predictions (475 citations) |
| 9 | Interpreting token compositionality in LLMs | Aljaafari et al. | 2024 | `2410.12924_token_compositionality.pdf` | CAP methodology; no single layer integrates tokens compositionally |
| 10 | Probing for idiomaticity in vector space models | Garcia et al. | 2021 | `garcia2021_probing_idiomaticity.pdf` | BERT fails to capture idiomaticity; NCS dataset |
| 11 | Language Models Implement Word2Vec-style Arithmetic | Merullo et al. | 2023 | `2305.16130_lm_word2vec_arithmetic.pdf` | Vector arithmetic for relational tasks in LLMs |
| 12 | Linear Algebraic Structure of Word Senses | Arora et al. | 2016 | `1601.03764_linear_structure_word_senses.pdf` | Polysemous senses in linear superposition |
| 13 | Language Models Represent Space and Time | Gurnee & Tegmark | 2023 | `2310.02207_lm_represent_space_time.pdf` | Linear representations of spatial/temporal concepts |
| 14 | Linearity of Relation Decoding | Hernandez et al. | 2023 | `2308.09124_linearity_relation_decoding.pdf` | Relations decoded via single linear transformation |
| 15 | Steering Language Models With Activation Engineering | Turner et al. | 2023 | `2308.10248_activation_engineering.pdf` | Concept steering via activation vectors |
| 16 | Gemma Scope | Lieberum et al. | 2024 | `2408.05147_gemma_scope.pdf` | Open SAE suite for Gemma-2 models |
| 17 | Improving Dictionary Learning with Gated SAEs | Rajamanoharan et al. | 2024 | `2404.16014_gated_sae.pdf` | Gated SAE architecture |
| 18 | Llama Scope | He et al. | 2024 | `2410.20526_llama_scope.pdf` | 256 SAEs for Llama-3.1-8B |
| 19 | BatchTopK Sparse Autoencoders | Bussmann et al. | 2024 | `2409.06981_batchtopk_sae.pdf` | TopK SAE training approach |
| 20 | Polysemanticity and Capacity | Scherlis et al. | 2022 | `2210.01892_polysemanticity_capacity.pdf` | Why neurons become polysemantic |
| 21 | Can Transformer be Too Compositional? | Dankers et al. | 2022 | `2205.15301_transformer_compositional_idioms.pdf` | Idiom processing in NMT; over-composition bias |
| 22 | A Psycholinguistic Analysis of BERT's Compounds | Buijtelaar et al. | 2023 | `2302.07788_bert_compounds.pdf` | BERT's compound noun representations |
| 23 | Probing BERT for German Compound Semantics | Miletić et al. | 2025 | `2501.16927_probing_bert_german_compounds.pdf` | German compound probing |
| 24 | Decomposing The Dark Matter of SAEs | Engels et al. | 2024 | `2410.14670_dark_matter_sae.pdf` | Unexplained variance in SAE reconstructions |
| 25 | Feature Hedging | Chanin et al. | 2025 | `2503.01370_feature_hedging.pdf` | Correlated features break narrow SAEs |
| 26 | Temporal Sparse Autoencoders | Bhalla et al. | 2025 | `2501.14404_temporal_sae.pdf` | Sequential structure in SAE features |
| 27 | Sparse Feature Circuits | Marks et al. | 2024 | `2403.19647_sparse_feature_circuits.pdf` | Causal feature circuits in LLMs |
| 28 | Mechanistic Interpretability for AI Safety | Bereska et al. | 2024 | `2404.14082_mechanistic_interpretability_safety.pdf` | Comprehensive MI review |
| 29 | Rethinking SAE Evaluation via Polysemous Words | Minegishi et al. | 2025 | `2501.07096_rethinking_sae_polysemous.pdf` | SAE evaluation through polysemy |
| 30 | Origins of Representation Manifolds | Modell et al. | 2025 | `2501.15832_origins_representation_manifolds.pdf` | Theory of representation manifolds |
| 31 | Emergent Linguistic Structure | Manning et al. | 2020 | `2004.14601_emergent_linguistic_structure.pdf` | Self-supervised linguistic learning |
| 32 | From Tokens to Words: Inner Lexicon of LLMs | — | 2024 | `2410.05864_tokens_to_words_inner_lexicon.pdf` | Inner lexicon formation |
| 33 | Syntax-guided Neural Module Distillation | Pandey | 2023 | `2301.08998_syntax_compositionality.pdf` | Compositional probing via syntax |
| 34 | Linear Spaces of Meanings | Trager et al. | 2023 | `2302.14383_linear_spaces_meanings.pdf` | Compositional structures in VLMs |
| 35 | I Predict Therefore I Am | Liu et al. | 2025 | `2503.08980_next_token_prediction_concepts.pdf` | NTP and concept learning theory |
| 36 | Transformer Dictionary Learning | Yun et al. | 2021 | `2103.15949_transformer_dictionary_learning.pdf` | Early dictionary learning for transformers |

See `papers/notes/` for detailed reading notes on key papers.

---

## Datasets
**Total datasets downloaded: 4**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Noun Compound Senses (NCS) | github.com/marcospln/noun_compound_senses | 280 EN + 180 PT compounds | Compositionality probing | `datasets/noun_compound_senses/` | Includes compositionality ratings and sentence variants |
| MAGPIE Idiom Corpus | github.com/hslh/magpie-corpus | 56,622 instances | Idiomaticity detection | `datasets/magpie/` | Binary and 5-way idiom labels with context |
| Custom Compound Nouns Test Set | Created for this project | 35 compounds | Compositionality spectrum | `datasets/compound_nouns_test.jsonl` | Covers compositionality range 1-5 |
| Gagné (2001) Compounds | Embedded in NCS | 300 compounds, 16 relations | Compound relation classification | In NCS dataset | Groups of 5 compounds with controlled conditions |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories
**Total repositories cloned: 7**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| **Footprints** | github.com/sfeucht/footprints | Token erasure analysis for multi-token concepts | `code/footprints/` | **Primary tool** — includes pre-trained probes, erasure scoring |
| **SAELens** | github.com/jbloomAus/SAELens | SAE training and analysis library | `code/SAELens/` | Supports pre-trained SAEs, Neuronpedia integration |
| **TransformerLens** | github.com/TransformerLensOrg/TransformerLens | Hook-based LLM interpretability toolkit | `code/TransformerLens/` | 50+ model support, activation caching |
| **Linear Rep Geometry** | github.com/KihoPark/linear_rep_geometry | Linear representation hypothesis tools | `code/linear_rep_geometry/` | Causal inner product, concept direction analysis |
| **SAE Spelling** | github.com/lasr-spelling/sae-spelling | Feature absorption analysis | `code/sae-spelling/` | Attribution, ablation, probing tools |
| **Neuronpedia** | github.com/hijohnnylin/neuronpedia | SAE feature browser platform | `code/neuronpedia/` | Browse 4+ TB of activations |
| **SAEDashboard** | github.com/jbloomAus/SAEDashboard | Feature visualization dashboards | `code/SAEDashboard/` | Anthropic-style feature dashboards |

See `code/README.md` for detailed descriptions of each repository.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper search**: Used paper-finder with 7 targeted queries covering: compound concepts in LLMs, superposition/polysemanticity, linear representation hypothesis, sparse autoencoders, compositional semantics, next-token prediction, and feature absorption
2. **Dataset search**: Searched HuggingFace, Papers with Code, GitHub, and academic sources
3. **Code search**: Identified tools from paper references, GitHub search, and community recommendations

### Selection Criteria
- **Papers**: Prioritized work on (a) multi-token/compound concept representation, (b) linear representation theory, (c) SAE feature discovery, (d) compositional semantics probing
- **Datasets**: Selected datasets with compound noun compositionality annotations and controlled experimental conditions
- **Code**: Focused on tools that enable residual stream analysis, SAE feature extraction, and multi-token concept probing

### Challenges Encountered
- Several arxiv IDs from paper-finder search results pointed to wrong papers (different papers with same ID); required manual verification and re-downloading
- The Gagné (2001) compound dataset is not freely available as standalone but is embedded in the NCS dataset
- No dataset exists specifically designed for probing compound concept directions in modern autoregressive LLMs — this is a gap our research can fill

### Gaps and Workarounds
- **No autoregressive compound probing dataset**: Created custom test set (`compound_nouns_test.jsonl`) and can generate more from Wikipedia
- **Pre-trained SAEs may not cover all models we want**: SAELens supports training custom SAEs; Gemma Scope and Llama Scope provide pre-trained options
- **Token erasure probes are only available for Llama-2-7B and Llama-3-8B**: Can retrain for other models using the footprints codebase

---

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Custom compound noun test set** (`compound_nouns_test.jsonl`) — 35 compounds across compositionality spectrum, expandable
- **NCS dataset** — 280 English compounds with compositionality ratings and context sentences
- **Wikipedia contexts** — Extract sentences containing target compounds for erasure analysis

### 2. Baseline Methods
- **Token erasure score** (Feucht et al.): Measures whether compound is treated as lexical unit
- **Compositional control baseline**: Compare compound nouns to clearly compositional phrases (e.g., "washing machine" vs. "red machine")
- **Separate processing baseline** (Ormerod et al.): Process constituent words in separate contexts
- **Random/shuffled baseline**: Permuted labels for probe accuracy

### 3. Evaluation Metrics
- **Erasure score (ψ)**: Quantifies token-level information loss at last compound token
- **Probe accuracy across layers**: Maps where compound representation forms
- **SAE feature F1**: Tests whether SAE features reliably track compound concepts
- **Cosine similarity**: Between compound direction and constituent directions
- **Steering effect size**: Causal validation via activation addition

### 4. Code to Adapt/Reuse
- **footprints**: Primary tool for erasure analysis — train probes, compute scores, extract vocabulary
- **TransformerLens + SAELens**: For hooking into model internals and analyzing SAE features
- **linear_rep_geometry**: For computing concept directions with causal inner products
- **sae-spelling**: For testing feature absorption in compound concept features

### 5. Recommended Experimental Workflow
1. **Phase 1**: Compute erasure scores for compound nouns using footprints
2. **Phase 2**: Analyze SAE features for compound concepts using SAELens + Gemma Scope
3. **Phase 3**: Estimate concept directions using linear_rep_geometry framework
4. **Phase 4**: Perform causal interventions (steering, ablation) to validate findings
