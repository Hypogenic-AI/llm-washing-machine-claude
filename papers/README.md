# Downloaded Papers

## Core Papers (Most Relevant to Research Question)

1. **Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs** (Feucht et al., 2024)
   - File: `2406.20086_token_erasure_implicit_vocabulary.pdf`
   - arXiv: 2406.20086
   - Why relevant: Directly studies how multi-token concepts are assembled in early layers; proposes "implicit vocabulary" concept

2. **How Is a "Kitchen Chair" like a "Farm Horse"?** (Ormerod et al., 2024)
   - File: `ormerod2024_kitchen_chair_compound_semantics.pdf`
   - DOI: 10.1162/coli_a_00495
   - Why relevant: Probes compound noun semantics in transformers; "together vs separate" comparison directly tests compositional vs holistic representation

3. **The Linear Representation Hypothesis** (Park et al., 2024)
   - File: `2311.03658_linear_representation_hypothesis.pdf`
   - arXiv: 2311.03658
   - Why relevant: Formalizes concept directions; framework for testing whether "washing machine" has a unique linear direction

4. **Toy Models of Superposition** (Elhage et al., 2022)
   - File: `2209.10652_toy_models_superposition.pdf`
   - arXiv: 2209.10652
   - Why relevant: Theoretical foundation for feature superposition; explains why concepts aren't stored in individual neurons

5. **A is for Absorption** (Chanin et al., 2024)
   - File: `2409.14507_absorption_feature_splitting.pdf`
   - arXiv: 2409.14507
   - Why relevant: Shows SAE features for hierarchical concepts are unreliable due to absorption; critical caveat for SAE-based compound analysis

## Sparse Autoencoder Papers

6. **Sparse Autoencoders Find Highly Interpretable Features** (Cunningham et al., 2023) — `2309.08600_sparse_autoencoders_interpretable.pdf`
7. **Scaling and evaluating sparse autoencoders** (Gao et al., 2024) — `2406.04093_scaling_evaluating_sae.pdf`
8. **Gated Sparse Autoencoders** (Rajamanoharan et al., 2024) — `2404.16014_gated_sae.pdf`
9. **BatchTopK Sparse Autoencoders** (Bussmann et al., 2024) — `2409.06981_batchtopk_sae.pdf`
10. **Gemma Scope** (Lieberum et al., 2024) — `2408.05147_gemma_scope.pdf`
11. **Llama Scope** (He et al., 2024) — `2410.20526_llama_scope.pdf`
12. **Decomposing The Dark Matter of SAEs** (Engels et al., 2024) — `2410.14670_dark_matter_sae.pdf`
13. **Feature Hedging** (Chanin et al., 2025) — `2503.01370_feature_hedging.pdf`
14. **Temporal Sparse Autoencoders** (Bhalla et al., 2025) — `2501.14404_temporal_sae.pdf`
15. **Rethinking SAE Evaluation** (Minegishi et al., 2025) — `2501.07096_rethinking_sae_polysemous.pdf`
16. **Sparse Feature Circuits** (Marks et al., 2024) — `2403.19647_sparse_feature_circuits.pdf`

## Representation and Concept Direction Papers

17. **Language Models Implement Word2Vec-style Arithmetic** (Merullo et al., 2023) — `2305.16130_lm_word2vec_arithmetic.pdf`
18. **Linear Algebraic Structure of Word Senses** (Arora et al., 2016) — `1601.03764_linear_structure_word_senses.pdf`
19. **Language Models Represent Space and Time** (Gurnee & Tegmark, 2023) — `2310.02207_lm_represent_space_time.pdf`
20. **Linearity of Relation Decoding** (Hernandez et al., 2023) — `2308.09124_linearity_relation_decoding.pdf`
21. **Steering Language Models** (Turner et al., 2023) — `2308.10248_activation_engineering.pdf`
22. **Origins of Representation Manifolds** (Modell et al., 2025) — `2501.15832_origins_representation_manifolds.pdf`
23. **Linear Spaces of Meanings** (Trager et al., 2023) — `2302.14383_linear_spaces_meanings.pdf`

## Compositionality and Compound Semantics Papers

24. **Interpreting token compositionality in LLMs** (Aljaafari et al., 2024) — `2410.12924_token_compositionality.pdf`
25. **Probing for idiomaticity** (Garcia et al., 2021) — `garcia2021_probing_idiomaticity.pdf`
26. **Can Transformer be Too Compositional?** (Dankers et al., 2022) — `2205.15301_transformer_compositional_idioms.pdf`
27. **BERT's Representations of Compounds** (Buijtelaar et al., 2023) — `2302.07788_bert_compounds.pdf`
28. **Probing BERT for German Compounds** (Miletić et al., 2025) — `2501.16927_probing_bert_german_compounds.pdf`
29. **Syntax-guided Compositionality Probing** (Pandey, 2023) — `2301.08998_syntax_compositionality.pdf`
30. **From Tokens to Words: Inner Lexicon** (2024) — `2410.05864_tokens_to_words_inner_lexicon.pdf`

## Transformer Internals Papers

31. **FF Layers Promote Concepts** (Geva et al., 2022) — `2203.14680_ff_layers_promoting_concepts.pdf`
32. **I Predict Therefore I Am** (Liu et al., 2025) — `2503.08980_next_token_prediction_concepts.pdf`
33. **Emergent Linguistic Structure** (Manning et al., 2020) — `2004.14601_emergent_linguistic_structure.pdf`
34. **Transformer Dictionary Learning** (Yun et al., 2021) — `2103.15949_transformer_dictionary_learning.pdf`
35. **Polysemanticity and Capacity** (Scherlis et al., 2022) — `2210.01892_polysemanticity_capacity.pdf`
36. **Mechanistic Interpretability for AI Safety** (Bereska et al., 2024) — `2404.14082_mechanistic_interpretability_safety.pdf`

## Detailed Reading Notes

See `papers/notes/` directory for detailed reading notes on key papers:
- `token_erasure_notes.md` — Comprehensive notes on Feucht et al. (2024)
- `absorption_notes.md` — Notes on feature absorption in SAEs
- `linear_rep_notes.md` — Notes on linear representation hypothesis
- `superposition_notes.md` — Notes on superposition theory
- `compositionality_notes.md` — Notes on compositionality probing papers
- `ff_layers_and_ntp_notes.md` — Notes on FFN concept promotion
