# Toy Models of Superposition - Research Notes

**Paper**: Elhage et al., 2022, "Toy Models of Superposition"
**Context**: Understanding how compound concepts like "washing machine" are stored in LLMs

---

## 1. What is Superposition? How Does it Relate to Polysemanticity?

### Superposition Definition
**Superposition** is a phenomenon where neural networks represent **more features than they have dimensions** by exploiting properties of high-dimensional spaces. Specifically:

- Features are represented as **almost-orthogonal directions** in activation space
- Models can store n features in m dimensions where n >> m
- This works by tolerating "interference" - when one feature activates, it looks like other features slightly activating
- The model uses **nonlinear filtering** (ReLU activations) to clean up this noise

**Key insight**: Neural networks may be "noisily simulating larger, highly sparse networks" - a small network in superposition can approximate a much larger network with no interference.

### Relationship to Polysemanticity
**Polysemanticity** is when a single neuron responds to multiple unrelated features (e.g., a neuron firing for both "cat ears" and "car wheels").

**The connection**: Superposition CAUSES polysemanticity!

- When features are in superposition, they don't align with individual neurons
- Multiple features share the same neurons, creating polysemantic neurons
- Polysemanticity is essentially what we observe when looking at individual neurons in a model using superposition

**Two competing forces**:
1. **Privileged Basis**: Some representations (with activation functions) encourage features to align with neurons
2. **Superposition**: Pressure to represent more features than dimensions, pushing features away from neurons

---

## 2. Key Theoretical Framework (Toy Model Setup)

### The Linear Representation Hypothesis
The paper assumes features are represented as **directions in activation space**:
- Each feature f_i has a representation direction W_i
- Multiple features (f1, f2, ...) with values (x_f1, x_f2, ...) are represented as: x_f1·W_f1 + x_f2·W_f2 + ...
- This is a LINEAR representation even though features themselves are nonlinear functions of input

### Toy Model Architecture
The paper uses simple ReLU networks with synthetic data:

**Model**: x ∈ R^n → h ∈ R^m → x' ∈ R^n
- x = high-dimensional "feature vector" (n features)
- h = low-dimensional hidden layer (m neurons, m < n)
- Goal: compress x into h, then recover it

**Synthetic Data Properties**:
1. **Feature Sparsity** (S): Each feature x_i is zero with probability S, otherwise uniform [0,1]
   - Sparse features are critical - they activate rarely, reducing interference

2. **More Features Than Neurons**: n > m (the fundamental tension)

3. **Feature Importance** (I_i): Different features have different weights in the loss function
   - Some features (like "floppy ear detector" for ImageNet) matter more than others

**Key Finding**: With **dense features** (S=0%), models learn PCA-like solutions (orthogonal basis of top features). With **sparse features** (S=90%+), models use superposition to represent ALL features, even though there are fewer dimensions!

---

## 3. Phase Transitions in Superposition

### The Phase Change Phenomenon
Whether features are stored in superposition is governed by a **phase change** (like water/ice transitions):

**Parameters that control the phase**:
- **Sparsity (1-S)**: How often features activate
- **Feature Importance (I)**: How much each feature matters for loss

**The phase diagram shows**:
- **Dense regime** (low sparsity): Features NOT in superposition - model learns orthogonal basis of most important features
- **Sparse regime** (high sparsity): Features IN superposition - model represents more features than dimensions
- **Critical point**: Sharp transition between regimes

### Why Sparsity Enables Superposition
The intuition:
- If features activate together frequently, they interfere constantly → bad loss
- If features are sparse (rarely active), they rarely interfere → tolerable noise
- Nonlinear activation (ReLU) can filter small interference when only one feature is active

**Mathematical insight**:
- Linear models (no ReLU) cannot do better than PCA
- Even slight nonlinearity (ReLU) enables radically different behavior - superposition!

### Fractional Dimensionality
Between "not learned" (0 dimensions) and "dedicated neuron" (1 dimension), features can have **fractional dimensionality** (e.g., 0.5, 0.67, 0.75). This IS superposition - the regime where features share dimensions.

**Analogy**: Just as ice has multiple phases (hexagonal ice, cubic ice), superposition has multiple "phases" corresponding to different geometric configurations.

---

## 4. Feature Geometry (How Features Are Arranged in Superposition)

### Geometric Structures
One of the most surprising findings: features in superposition organize into **uniform polytope geometries**!

**Observed structures** (for m=2 dimensions):
- **Digon** (antipodal pair): 2 features, dimensionality = 1/2
- **Triangle**: 3 features, dimensionality = 2/3
- **Pentagon**: 5 features, dimensionality = 2/5
- **Square antiprism**: 8 features, dimensionality = 3/8
- **Tetrahedron**: 4 features, dimensionality = 3/4

### Connection to Thomson Problem
The model is essentially solving a **generalized Thomson problem**: placing n points on an m-dimensional sphere to minimize interference (like electrons repelling on a sphere).

When features are equally important and sparse:
- Solutions correspond to **uniform polyhedra**
- All vertices have same geometry → all features have same dimensionality
- The model finds symmetric, elegant geometric arrangements

### W^T W Correspondence
There's an exact correspondence between:
- **Polytopes** (geometric configurations of feature directions)
- **Symmetric, positive-definite, low-rank matrices** (W^T W)

Each superposition strategy corresponds to a polytope. For example:
- 3 equally-important features in 2D → equilateral triangle
- The "minimal superposition" sets W ⊥ (1,1,1,...) which creates a regular n-simplex

### Tegum Products
Many Thomson solutions are **tegum products** (combining polytopes in orthogonal subspaces):
- Features in different tegum factors have ZERO interference
- Example: octahedron = tegum product of 3 antipodal pairs
- This explains why we see 3D Thomson solutions even in higher dimensions

### Non-Uniform Superposition
Real-world features aren't uniform. Key behaviors:

**Varying importance/sparsity**:
- Polytopes **smoothly deform** as parameters change
- At critical points, they **snap** to different polytope configurations (phase transitions)

**Correlated features** (features that co-occur):
- Prefer to be **orthogonal** (separate tegum factors) - no interference
- Create "local almost-orthogonal bases" - subsets that aren't in superposition
- When forced together, prefer **positive interference** (side-by-side)
- Can **collapse** into principal component (PCA-like behavior)

**Anti-correlated features** (mutually exclusive):
- Prefer to be in **same tegum factor** with **negative interference**
- Ideally antipodal (opposite directions)

**Implication**: Correlational structure strongly influences which features group together in polysemantic neurons - NOT random!

---

## 5. Implications for Compound Concepts - Could "Washing Machine" Be Stored in Superposition?

### Evidence that "Washing Machine" Could Be in Superposition

**Strong arguments**:

1. **Sparsity**: "Washing machine" is a rare concept
   - Most text tokens don't refer to washing machines
   - Most images don't contain washing machines
   - Meets the sparsity criterion that enables superposition

2. **Compound nature**: "Washing machine" could decompose into:
   - "Washing" (cleaning, water, soap...)
   - "Machine" (mechanical, electrical...)
   - Each component feature could be in superposition

3. **Feature count >> neuron count**:
   - LLMs need to represent millions of concepts
   - Even large models don't have enough neurons for 1-to-1 mapping
   - Superposition is necessary for coverage

4. **Correlation structure**: Related concepts likely co-occur:
   - "Washing machine" + "laundry" + "detergent" + "clothes"
   - Paper shows correlated features can form local orthogonal bases
   - BUT washing machine correlates with many different contexts

**Storage mechanism hypothesis**:
- "Washing machine" might not have a dedicated neuron
- Instead: represented as a **direction in activation space**
- This direction could be **in superposition** with other appliances, household items, etc.
- When "washing machine" activates, causes small activation of related features (interference)

### Testing This Hypothesis

**What to look for in experiments**:

1. **Polysemantic neurons**: Neurons that respond to washing machines AND unrelated concepts
2. **Distributed representation**: Washing machine activates many neurons weakly, not one strongly
3. **Interference patterns**: Activating "washing machine" slightly activates related concepts
4. **Directional probing**: Can we find a "washing machine direction" that isn't a neuron?

---

## 6. Practical Implications for Our Experiments

### For Interpretability Research

**Challenge**: If superposition is pervasive:
- Can't just look at individual neurons to understand "washing machine"
- Need to find **feature directions** that don't align with neurons
- Polysemantic neurons are the NORM in LLMs, not the exception

**Opportunities**:

1. **Local non-superposition assumption**:
   - For specific sub-distributions (correlated features), might have local orthogonal bases
   - Could use PCA within specific contexts
   - Example: when processing "laundry" text, relevant features might not be in superposition

2. **Geometric structure**:
   - Could look for polytope patterns in activation space
   - W^T W analysis to find interference patterns
   - Related features might form recognizable geometric configurations

3. **Correlational analysis**:
   - Features that co-occur should group in predictable ways
   - Could compare across models - polysemantic groupings should be consistent
   - Anti-correlated features (e.g., "washing machine" vs "dishwasher"?) might be antipodal

### Experimental Design Implications

**What to measure**:

1. **Feature dimensionality**:
   - Is "washing machine" taking up <1 dimension?
   - How is this shared with other concepts?

2. **Interference patterns**:
   - When we activate "washing machine", what else activates?
   - Is the interference predictable from correlational structure?

3. **Sparsity effects**:
   - Compare common concepts (low sparsity) vs rare concepts (high sparsity)
   - Rare concepts MORE likely to be in superposition

4. **Compound decomposition**:
   - Can we find separate directions for "washing" and "machine"?
   - How do they combine?

**Methods to try**:

1. **Activation probing**:
   - Present washing machine stimuli
   - Map which neurons activate (expect: many neurons, weakly)

2. **Direction finding**:
   - Use difference vectors (like gender direction in word embeddings)
   - "Washing machine" - "washing" = "machine" direction?

3. **W^T W analysis**:
   - Look at dot products between feature directions
   - Find polytope structures
   - Identify tegum factors

4. **Cross-model comparison**:
   - Do different models group same features in superposition?
   - Paper suggests correlation structure drives grouping → should be consistent

### Potential Obstacles

1. **Non-uniform superposition**: Real features have variable importance/sparsity
   - Makes geometry more complex than toy models
   - But paper shows it's still tractable - deformations of uniform polytopes

2. **Nonlinear compression**: Models might use even more exotic encoding
   - Paper argues this is unlikely to be pervasive
   - Still interpretable as circuits computing compression

3. **High dimensionality**: Real models have thousands of dimensions
   - Can't visualize like 2D/3D toy models
   - But mathematical framework (W^T W) still applies

### Key Takeaways for Our Research

1. **Don't expect dedicated "washing machine" neurons** - look for directions
2. **Sparsity is key** - rare concepts most likely in superposition
3. **Correlation matters** - related concepts group predictably
4. **Superposition is gradual** - fractional dimensionality, not binary
5. **Geometry provides structure** - even in high-dimensional spaces
6. **Local orthogonality possible** - within specific contexts, might escape superposition

---

## Critical Open Questions

1. **How much superposition in real LLMs?** Toy models prove it's possible, but how pervasive is it?

2. **Can we "decode" superposition?** If we know features are in superposition, can we separate them?

3. **Does this explain the full story?** Or are there other mechanisms beyond superposition?

4. **Computation in superposition**: Paper shows simple circuits (absolute value) can work in superposition. How complex can this get?

5. **Connection to adversarial examples**: Paper hints at link - interference patterns might explain vulnerabilities

---

## References for Deep Dives

- **Compressed sensing**: Mathematics of sparse signal recovery - direct connection to superposition
- **Thomson problem**: Placing points on sphere - explains geometric structures
- **Distributed representations (neuroscience)**: Population codes - similar ideas
- **Disentanglement literature**: Related but different - focuses on VAEs/GANs

---

**Bottom line for washing machine research**: Superposition theory strongly suggests "washing machine" is NOT stored in a single neuron, but as a direction in activation space that shares dimensions with other concepts. The degree of sharing depends on sparsity and correlation structure. Our experiments should focus on finding these directions and understanding their interference patterns rather than looking for dedicated neurons.
