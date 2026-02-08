# A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders

**Paper:** arXiv:2409.14507v6 [cs.CL] 17 Nov 2025
**Authors:** David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, Joseph Bloom
**Affiliations:** LASR Labs, UCL, Tübingen AI Center, MATS, Decode Research
**Conference:** NeurIPS 2025

## 1. Feature Splitting vs Feature Absorption

### Feature Splitting
- **Definition:** A phenomenon where a feature represented in a single latent in a smaller SAE splits into two or more latents in a larger SAE
- **Example:** "math" may split into "algebra", "geometry", etc.
- A single "starts with L" latent may split into "starts with capital L" and "starts with lowercase l" latents
- **Not necessarily problematic** - split features are still easily identifiable and can be useful depending on context
- **Detection:** Measured using k-sparse probing - if increasing k from n to n+1 causes a significant jump in F1 score (threshold: 0.03), this indicates feature splitting
- **Occurs more frequently with higher sparsity** (higher L0 values)

### Feature Absorption
- **Definition:** A problematic variant of feature splitting where seemingly monosemantic latents fail to fire where they should, and instead get "absorbed" into their children features
- **Core problem:** SAE latents appear to track human-interpretable concepts but have arbitrary false negatives
- **Example (from paper):**
  - A "starts with S" feature always fires alongside "short"
  - Instead of learning an interpretable "starts with S" latent, the SAE absorbs this feature into the "short" latent
  - The "starts with S" latent fails to activate on "short" tokens despite "short" starting with "S"
  - The "short" latent (latent 1085) absorbs the "starts with S" direction and fires instead

### Key Distinction
- **Feature splitting:** Hierarchical features split into finer-grained features (interpretable)
- **Feature absorption:** Parent features fail to fire on arbitrary cases, absorbed by child features (uninterpretable/unreliable)

## 2. How SAE Features Relate to Compound/Hierarchical Concepts

### Hierarchical Features Definition
- Features f1 and f2 form a hierarchy with f1 as parent and f2 as child if: f2 ⟹ f1
- Meaning: every time f2 fires, f1 must also fire
- Example: "short" ⟹ "starts with S" (every time "short" fires, "starts with S" should also fire)

### The Problem with Hierarchies
**Even if all underlying features are linear and sparsely activating, an SAE will fail to recover true features if they form a hierarchy.**

Key findings:
1. **Perfect reconstruction maintained:** SAEs can achieve perfect reconstruction while still exhibiting absorption
2. **Sparsity drives absorption:** Optimizing for sparsity (L1 loss) incentivizes absorption
3. **The mechanism:** If a dense feature and sparse feature co-occur, absorbing the dense feature into the sparse feature increases sparsity while maintaining reconstruction

### Mathematical Proof (δ-absorption)
The paper provides formal proof that for hierarchical features:
- Reconstruction loss is unaffected by absorption (remains 0)
- Sparsity loss decreases with absorption: dL_sp/dδ = -p11 (where p11 is co-occurrence probability)
- Therefore, gradient descent favors absorption when hierarchical features exist

### Implications for Compound Concepts Like "Washing Machine"
This is **directly relevant** to studying compound concepts:
1. **Compound concepts likely form hierarchies** in LLMs (e.g., "washing machine" ⟹ "appliance", "washing machine" ⟹ "household item")
2. **SAE latents may not reliably track compound concepts** due to absorption
3. **More specific latents may absorb parent features:** A "washing machine" latent might absorb "appliance" or "starts with W" directions
4. **False negatives are likely:** SAE latents tracking compound concepts may fail to fire on seemingly arbitrary examples

## 3. Datasets and Models Used

### Primary Model
- **Gemma-2-2B** (base model, not instruction-tuned)
- Used for most experimental studies

### SAEs Tested
1. **Gemma Scope SAEs** (primary focus)
   - Residual stream SAEs
   - Widths: 16k and 65k latents
   - Layers: 0-25 (analyzed 0-17 for absorption due to attention movement)
   - Various L0 sparsity levels (25-400+)

2. **Custom-trained SAEs:**
   - **Qwen2 0.5B:** Layers 0-8, L1 loss, L0: 25-50, explained variance: 0.77-0.83
   - **Llama 3.2 1B:** Layers 0-8, both L1 loss and TopK architectures, L0: 27-110, explained variance: 0.74-0.89

### Task/Dataset
- **First-letter spelling task** used as primary evaluation
- In-context learning (ICL) prompts with format:
  ```
  {token} has the first letter: {capitalized_first_letter}
  ```
- Example:
  ```
  tartan has the first letter: T
  mirth has the first letter: M
  dog has the first letter:
  ```
- Tokens: English alphabet (a-z, A-Z) with optional leading space
- Train/test split: 80%/20%

## 4. Key Experimental Methodology

### Toy Model Experiments
**Setup:**
- 4 true orthogonal features in 50-dimensional space
- Each feature fires with magnitude 1.0
- Feature f0 fires with probability 0.25
- Features f1, f2, f3 fire with probability 0.05
- SAE with 4 latents trained using SAELens
- L1 coefficient: 3e-5, learning rate: 3e-4, 100M training activations

**Hierarchical condition:**
- Feature 1 only fires if feature 0 fires (f1 ⟹ f0)
- Demonstrates clear absorption in encoder/decoder weights

**Results:**
- Independent features: Perfect recovery
- Hierarchical features: Clear absorption pattern with gerrymandered encoder

### Linear Probing
- **Method:** Logistic regression (LR) trained on hidden activations
- Used as baseline/ground truth for comparison with SAE latents
- Achieves F1 scores 0.6-0.8+ depending on layer

### K-Sparse Probing
- Train LR probe with L1 loss term
- Select k latents with largest weights
- Evaluate if increasing k improves F1 score (indicates splitting)
- L1 coefficient: 0.01 for latent selection
- k values tested: 1-15

### SAE Latent Selection Methods
Two approaches tested (gave similar results):
1. **Encoder cosine similarity:** Find latent with highest cosine similarity to LR probe
2. **k=1 sparse probing:** Use L1-regularized probe to select single best latent

### Ablation Studies
**Algorithm:**
1. Insert SAE in model computation (including error term)
2. Define scalar metric on model output distribution
3. Calculate baseline metric for test prompt
4. For each SAE latent:
   - Set latent activation to 0
   - Recalculate metric
   - Compute ablation effect (baseline - new metric)

**Metric used:**
```
m = g[y] - (1/(|L|-1)) * Σ g[l]  for l ∈ {L\y}
```
where g = final token logits, L = uppercase letters, y = correct letter

**Integrated Gradients:** Used as approximation for faster ablation computation

### Absorption Rate Metric
**Definition:**
```
absorption_rate = num_absorptions / lr_probe_true_positives
```

**Criteria for detecting absorption:**
1. Find k feature splits using k-sparse probe
2. Identify false-negative tokens (all k main latents fail to activate but LR probe classifies correctly)
3. Run integrated-gradients ablation on those tokens
4. If latent with largest negative ablation effect has:
   - Cosine similarity with LR probe > 0.025
   - Ablation effect at least 1.0 larger than second-highest
   - Then absorption has occurred

**Conservative estimate:** Only detects single dominant absorbing latent, misses multiple latents or weak main latent firing

## 5. Key Results and Findings Relevant to Compound Concept Representation

### Main Finding: Absorption is Widespread
- **Every LLM SAE tested showed absorption** (hundreds of open-source SAEs)
- Occurs across different architectures: L1 loss SAEs, TopK SAEs, BatchTopK SAEs
- Occurs across different models: Gemma-2-2B, Qwen2 0.5B, Llama 3.2 1B

### Precision vs Recall Trade-offs
- **Low L0 SAEs:** High precision, low recall (fewer false positives, many false negatives)
- **High L0 SAEs:** Low precision, high recall (fewer false negatives, more false positives)
- **No SAE matches LR probe performance** across all tested configurations

### F1 Score Patterns
- Best F1 scores vary by layer and feature
- Layers 0-12: Best F1 with L0 near 25-50
- Layers 13-25: Best F1 with L0 near 50-100
- Wide variance when broken down by individual letters

### Absorption Rate Findings
1. **Increases with sparsity:** Higher L0 → higher absorption rate
2. **Increases with SAE width:** 65k width shows more absorption than 16k
3. **No clear pattern by layer** (layers 0-17 analyzed)
4. **Varies significantly by letter/feature:** Some letters show much higher absorption (see Figure 23 in paper)
5. **Mean absorption rate:** 5-35% depending on SAE configuration

### Case Study: "Starts with S" (Layer 3, 16k, L0=59)
**Main latent (6510):**
- F1 score: 0.81
- Appears to be "starts with S" classifier
- **Fails to activate on _short token** despite "short" starting with "S"

**Absorbing latent (1085):**
- Token-aligned latent for "short" variants
- Cosine similarity with "starts with S" probe: 0.12 (vs 0.52 for main latent)
- **Activates with 5x magnitude of main latent on _short**
- Shows dominant ablation effect on _short token
- Ablation effect disappears when probe direction is projected out → confirms probe component is causal

### Partial Absorption
Also observed in toy models:
- Main latent fires weakly instead of turning off completely
- Occurs when:
  - Parent/child magnitude ratios vary
  - Imperfect co-occurrence (95% instead of 100%)
- Represents intermediate state between full separation and full absorption

### Feature Splitting Patterns
- Detected via k-sparse probing with F1 jump threshold of 0.03
- Example: "L" splits into uppercase "L" and lowercase "l" latents
- Mean splits per letter: 0-2.5 depending on L0
- More common at higher sparsity levels

### Limitations of SAE Sizes/Sparsity
**Critical finding:** "Varying SAE sizes or sparsity is insufficient to solve this issue"
- No "sweet spot" configuration eliminates absorption
- Trade-offs between precision and recall persist
- Fundamental theoretical problem, not just hyperparameter tuning issue

## 6. Code and Tools Released

### GitHub Repository
**URL:** https://github.com/lasr-spelling/sae-spelling

**Contents:**
- Toy model implementations demonstrating absorption
- Experimental code for real LLM SAEs
- Absorption metric implementation
- K-sparse probing implementation
- Ablation study algorithms

### Online Explorer
**URL:** https://feature-absorption.streamlit.app

**Features:**
- Interactive exploration of results
- Visualization of absorption patterns
- Browse absorption examples across different SAEs
- Examine specific latent behaviors

### SAELens Library
**URL:** https://github.com/jbloomAus/SAELens
- Used for training and evaluating SAEs
- Standard library for SAE work

### Neuronpedia Integration
**URL:** https://www.neuronpedia.org
- Feature dashboards for analyzed latents
- Examples shown for latents 1085 and 6510 (layer 3, Gemma Scope)
- Max activating examples and activation patterns

## 7. Implications for "Washing Machine" Representation Research

### Direct Implications

#### 1. SAE Latents May Be Unreliable for Tracking Compound Concepts
- **Problem:** A latent appearing to track "washing machine" may fail to activate on arbitrary examples
- **Reason:** Hierarchical structure (washing machine ⟹ appliance, household item, starts with W, etc.)
- **Risk:** False sense of understanding from inspecting max-activating examples

#### 2. Circuit Discovery Will Be More Difficult
**Quote from paper:** "Techniques which seek to describe circuits in terms of a sparse combination of latents will also be more difficult in the presence of feature absorption"

For "washing machine":
- May need many more latents than expected to fully capture behavior
- More specific latents (e.g., "Maytag washing machine", "front-load washer") may absorb parent features
- Circuit analysis requiring complete coverage will miss absorbed features

#### 3. High-Stakes Classification Is Problematic
**Quote from paper:** "Feature absorption poses an obstacle... particularly important for applications where we need confidence that latents are fully tracking behaviors"

For compound concept research:
- Cannot rely on single latent to track "washing machine" concept
- Need to account for absorption into more specific variants
- Must verify latents fire on all expected examples, not just max-activating ones

#### 4. Dense Features More Likely to Be Absorbed
- First-letter features are relatively dense (fire frequently)
- Token-specific features are sparse
- **Dense features like "appliance" or "household item" more likely to be absorbed into "washing machine" latent**
- **Sparse features like specific brands/models less likely to be absorbed**

### Methodological Recommendations for Washing Machine Research

#### 1. Don't Rely on Feature Dashboards Alone
- Max-activating examples can be misleading
- Must systematically test false negatives
- Use ablation studies to verify causal importance

#### 2. Use Multiple Detection Methods
Combine:
- Linear probing (ground truth)
- K-sparse probing (detect splitting)
- Ablation studies (verify causality)
- Cosine similarity analysis (detect absorption candidates)

#### 3. Expect Hierarchies
Compound concepts like "washing machine" likely form hierarchies:
- "washing machine" ⟹ "appliance"
- "washing machine" ⟹ "laundry-related"
- "washing machine" ⟹ "household item"
- "washing machine" ⟹ "starts with W"
- Specific types ⟹ "washing machine" (e.g., "front-load" ⟹ "washing machine")

#### 4. Check for Absorption Systematically
Use absorption metric components:
- Train LR probe for "washing machine" concept
- Find SAE latents with high cosine similarity
- Test on comprehensive dataset (not just max-activating)
- Run ablations on false negatives
- Check if more specific latents are absorbing

#### 5. Consider Multiple SAE Configurations
- Different L0/sparsity levels reveal different absorption patterns
- Lower L0: High precision, may miss compound concept instances
- Higher L0: Better recall, may have more feature splitting
- No single configuration is optimal

#### 6. Look for Token-Aligned Absorbing Latents
Pattern from paper: Token-aligned latents (e.g., "_short") absorb parent features
- For "washing machine": Look for latents tracking specific washing machine-related tokens
- These may absorb higher-level "washing machine" concept
- Check variants: "washing machine", "washer", "washing-machine", etc.

### Potential Solutions Mentioned in Paper

#### 1. Meta-SAEs (Most Promising)
**Reference:** Bussmann et al. 2024
- Train SAE on decoder of another SAE
- Can decompose "Einstein" into "German" + "Physicist" + "starts with E"
- **Could decompose "washing machine" into constituent features**
- May help separate absorbed features

#### 2. Attribution Dictionary Learning
**Reference:** Olah et al. 2024
- Alternative to sparse dictionary learning
- May avoid absorption by design

#### 3. Structured Sparsity Techniques
- **Group Lasso:** Encourage related features to activate together
- **Hierarchical Sparse Coding:** Explicitly model feature hierarchies
- Could allow both parent and child features to fire appropriately

#### 4. Accepting Absorption and Using It
- Use absorption patterns to **recover feature hierarchies**
- Asymmetric encoder/decoder patterns signal absorption
- Could map out hierarchy of "washing machine" subfeatures

### Open Questions for Washing Machine Research

1. **What hierarchies does "washing machine" participate in?**
   - Parent concepts it should activate with?
   - Child concepts that might absorb it?

2. **How distributed is "washing machine" representation?**
   - Single latent or multiple latents?
   - How much absorption is occurring?

3. **Which SAE configuration best captures "washing machine"?**
   - Need systematic evaluation across L0/width/layer
   - May need different configurations for different research questions

4. **Can we decompose "washing machine" using Meta-SAEs?**
   - Into: appliance + laundry + household + ...?
   - Would reveal compositional structure

5. **How does absorption affect different compound concepts differently?**
   - "Washing machine" vs "coffee maker" vs "smartphone"
   - Frequency, specificity, hierarchy depth effects?

### Concrete Next Steps

1. **Replicate absorption detection** on "washing machine" relevant latents
2. **Map out hypothesized hierarchies** for washing machine concept
3. **Test multiple SAE configurations** systematically
4. **Try Meta-SAE decomposition** if single latents show absorption
5. **Compare with linear probing** as ground truth
6. **Document false negatives** carefully - where do expected "washing machine" latents fail?
7. **Check for token-aligned absorbing latents** for washing machine-related tokens

## Key Takeaway

**SAEs may not reliably extract compound concepts like "washing machine" due to fundamental absorption issues caused by feature hierarchies. Researchers must go beyond max-activating examples and use systematic testing, ablation studies, and multiple detection methods to understand how compound concepts are truly represented.**

---

## Additional Resources

- **Paper:** arXiv:2409.14507v6
- **Code:** https://github.com/lasr-spelling/sae-spelling
- **Explorer:** https://feature-absorption.streamlit.app
- **Neuronpedia:** https://www.neuronpedia.org
