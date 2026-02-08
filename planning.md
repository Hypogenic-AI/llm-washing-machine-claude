# Research Plan: Where is "Washing Machine" Stored in LLMs?

## Motivation & Novelty Assessment

### Why This Research Matters
Large language models must represent millions of compound concepts ("washing machine", "hot dog", "coffee table"), but the residual stream has limited dimensionality (~4096 for 7B models). The superposition hypothesis (Elhage et al., 2022) suggests that sparse concepts share directions, but no study has directly investigated how multi-token compound nouns — as opposed to single-token concepts — are represented. Understanding this is crucial for mechanistic interpretability, concept editing, and understanding the limits of linear representation theory.

### Gap in Existing Work
- Park et al. (2024) tested 27 single-token concepts but never multi-token compounds
- Feucht et al. (2024) showed token erasure for named entities but not common compound nouns
- Ormerod et al. (2024) probed compound semantics in BERT (masked LM), not autoregressive models
- No study has directly compared compound concept directions vs. constituent word directions in the residual stream
- The core question — does "washing machine" have its own direction or is it derived from "washing" + context — remains unanswered

### Our Novel Contribution
We conduct three complementary experiments on GPT-2 (small, accessible, well-studied) to test whether compound concepts are stored holistically or compositionally:
1. **Next-token probability analysis**: Measure how strongly "washing" predicts "machine" vs. alternative contexts
2. **Residual stream direction analysis**: Extract and compare concept directions for compounds vs. constituents across layers
3. **Compositional vs. holistic representation probing**: Test whether compound representations are linearly separable from constituent representations

### Experiment Justification
- **Experiment 1** (Next-Token Prediction): Directly tests the hypothesis that LLMs "store washing and then machine becomes more likely." If P(machine|"The washing") >> P(machine|baseline), this supports the compositional hypothesis.
- **Experiment 2** (Residual Stream Directions): Tests whether compound concepts have unique directions in the residual stream. If compound directions are well-predicted by constituent directions, compounds are compositionally derived.
- **Experiment 3** (Layer-wise Probing): Tests where in the network compound concepts emerge. If compound representations only appear at the last token position after several layers, this supports the token erasure / implicit vocabulary hypothesis.

## Research Question
In large language models, are compound concepts like "washing machine" represented as unique directions in the residual stream, or are they derived compositionally from constituent words plus context?

## Background and Motivation
There are far more referenceable concepts than available dimensions in an LLM's residual stream. For single-token concepts, superposition allows nearly-orthogonal storage. But multi-token compounds raise a unique challenge: the model must somehow represent "washing machine" as a unified concept despite processing it as two separate tokens. Three possibilities exist:
1. **Holistic**: The model develops a unique direction for "washing machine" as a compound (assembled at the second token position)
2. **Compositional**: "Washing" has a direction that activates contextual circuits, making "machine" more likely; no unified compound direction exists
3. **Hybrid**: Some blend — early layers are compositional, but later layers develop a holistic compound representation

## Hypothesis Decomposition

### H1: Next-token prediction drives compound assembly
- **Testable prediction**: P(machine | "The washing") >> P(machine | "The red") >> P(machine | baseline)
- **Metric**: Next-token probability and rank of "machine" after "washing" vs. controls
- **Null hypothesis**: Context does not significantly affect P(machine)

### H2: Compound concepts lack unique residual stream directions
- **Testable prediction**: The direction for "washing machine" (at the second token) is well-predicted by a linear combination of "washing" direction + "machine" direction
- **Metric**: Cosine similarity between compound direction and span of constituent directions; reconstruction R²
- **Null hypothesis**: Compound directions are linearly independent of constituent directions

### H3: Compound representations emerge through token erasure
- **Testable prediction**: Information about preceding tokens is erased/transformed at the last compound token in early-to-mid layers, more so for conventional compounds than compositional phrases
- **Metric**: Linear probe accuracy for preceding token across layers; erasure score
- **Null hypothesis**: No differential erasure between compounds and compositional controls

## Proposed Methodology

### Approach
We use GPT-2 (124M parameters, 12 layers, d_model=768) as our primary model because:
- Well-studied in interpretability (TransformerLens support)
- Small enough to run multiple experiments on available GPUs
- 12 layers provide sufficient depth for layer-wise analysis
- Represents the architecture class (decoder-only transformer) used by modern LLMs

We'll also validate key findings on GPT-2-medium (355M, 24 layers) to check for scale effects.

### Experimental Steps

#### Experiment 1: Next-Token Prediction Analysis
1. Construct prompt templates: "The [word1]", "A [word1]", "This [word1]"
2. For each compound in our dataset, measure P(word2 | context + word1)
3. Compare against baselines:
   - P(word2 | random context word)
   - P(word2 | unconditional)
   - P(word2 | semantically related but non-compound word)
4. Analyze how compositionality rating correlates with P(word2)

#### Experiment 2: Residual Stream Direction Analysis
1. Collect hidden states from 100+ contexts containing each compound/constituent
2. Compute mean activation vectors for:
   - "washing machine" at the "machine" position (compound direction)
   - "washing" in isolation (word1 direction)
   - "machine" in isolation (word2 direction)
3. Compute cosine similarity between compound and constituent directions
4. Test linear reconstruction: can compound = α·word1 + β·word2 + bias?
5. Compare reconstruction quality across compositionality levels

#### Experiment 3: Layer-wise Probing / Token Erasure
1. For each compound, extract hidden states at the last token position across all layers
2. Train linear probes to predict:
   - Identity of the preceding token (word1)
   - Whether the context is compound vs. non-compound
3. Track probe accuracy across layers
4. Compare erasure patterns between:
   - Conventional compounds ("washing machine")
   - Compositional phrases ("red machine")
   - Idiomatic compounds ("hot dog")

### Baselines
- Random baseline: shuffled labels for probing
- Compositional control: "adj + noun" phrases (e.g., "red machine", "blue car")
- Separate word baseline: constituents in non-compound contexts
- Prior layer baseline: compare early vs. late layer representations

### Evaluation Metrics
- Next-token probability P(word2 | context)
- Next-token rank of word2
- Cosine similarity between direction vectors
- Linear reconstruction R² (how well compound = f(constituents))
- Probe accuracy (%) across layers
- Erasure score: change in probe accuracy from layer 0 to layer N
- Pearson/Spearman correlation between compositionality rating and metrics

### Statistical Analysis Plan
- Paired t-tests or Wilcoxon signed-rank tests for comparing conditions
- Spearman rank correlation for compositionality effects
- Bootstrap confidence intervals (1000 resamples) for key metrics
- Multiple comparison correction (Bonferroni) when testing multiple compounds
- Significance level: α = 0.05

## Expected Outcomes

### If compounds are stored holistically:
- Compound directions will be linearly independent of constituent directions
- Token erasure will be strong for compounds (high erasure score)
- P(word2|word1) will be high but compounds will have representations beyond just word prediction

### If compounds are stored compositionally:
- Compound directions will be well-reconstructed from constituent directions (high R²)
- P(word2|word1) will be the primary mechanism
- No significant token erasure beyond what's needed for prediction

### If hybrid:
- Early layers show compositional processing (high R² for reconstruction)
- Late layers show unique compound directions (low R², high cosine distance from constituents)
- Token erasure occurs in middle layers

## Timeline and Milestones
1. Environment setup & data prep: 15 min
2. Experiment 1 (next-token prediction): 30 min
3. Experiment 2 (residual stream directions): 45 min
4. Experiment 3 (layer-wise probing): 45 min
5. Analysis & visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- GPT-2's tokenizer may split compound words unexpectedly → verify tokenization first
- Context effects: need diverse contexts to avoid overfitting to specific prompts
- Small model may not develop compound representations that larger models do → validate on GPT-2-medium
- Some compounds in dataset are single tokens (e.g., "butterfly") → filter to multi-token only

## Success Criteria
1. Clear evidence for/against each of the three hypotheses (H1, H2, H3)
2. Quantitative metrics showing relationship between compositionality and representation
3. Layer-wise analysis showing where compound representations emerge
4. Statistical significance for key comparisons
5. Reproducible results with documented methodology
