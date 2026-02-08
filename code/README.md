# LLM Washing Machine Research - Code Repositories

This directory contains code repositories for studying how compound concepts like "washing machine" are stored in Large Language Models (LLMs). The focus is on multi-token representations, sparse autoencoders, probing classifiers, and linear representation analysis.

## Repository Overview

All repositories have been cloned with `--depth 1` to save space.

---

## 1. SAELens - Sparse Autoencoder Training and Analysis

**Repository:** `SAELens/`
**GitHub:** https://github.com/jbloomAus/SAELens
**Purpose:** Primary library for training and analyzing Sparse Autoencoders (SAEs) in LLMs

### Key Capabilities
- Train sparse autoencoders on any PyTorch-based model
- Download and analyze pre-trained SAEs from multiple sources
- Deep integration with TransformerLens via `HookedSAETransformer`
- Compatible with HuggingFace Transformers, NNsight, and other frameworks
- Generate feature dashboards with SAE-Vis Library

### Entry Points
- **Training:** `tutorials/training_a_sparse_autoencoder.ipynb`
- **Analysis:** `tutorials/basic_loading_and_analysing.ipynb`
- **Logit Lens:** `tutorials/logits_lens_with_features.ipynb`
- **Neuronpedia Integration:** `tutorials/tutorial_2_0.ipynb`

### Dependencies
- Python: ^3.10
- Core: transformer-lens >=2.16.1, transformers ^4.38.1, torch
- Visualization: plotly, plotly-express
- Data: datasets >=3.1.0, safetensors
- Optional: mamba-lens for Mamba model support

### Installation
```bash
cd SAELens
pip install sae-lens
# or for development:
poetry install
```

### Documentation
https://decoderesearch.github.io/SAELens/

---

## 2. TransformerLens - Hook-Based LLM Analysis Toolkit

**Repository:** `TransformerLens/`
**GitHub:** https://github.com/TransformerLensOrg/TransformerLens
**Purpose:** Mechanistic interpretability library for reverse-engineering transformer algorithms

### Key Capabilities
- Load 50+ different open source language models
- Access and cache any internal activation in the model
- Add functions to edit, remove, or replace activations during model execution
- Comprehensive hook system for intervention experiments
- Designed for mechanistic interpretability research

### Entry Points
- **Basic Usage:**
  ```python
  import transformer_lens
  model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
  logits, activations = model.run_with_cache("Hello World")
  ```
- **Tutorials:** ARENA Mechanistic Interpretability course (arena3-chapter1-transformer-interp.streamlit.app)

### Dependencies
- Python: >=3.8,<4.0
- Core: torch, transformers, einops, jaxtyping
- Utilities: datasets, pandas, wandb, tqdm
- Visualization: rich, circuitsvis (dev)

### Installation
```bash
cd TransformerLens
pip install transformer_lens
```

### Notable Research Using This Library
- Progress Measures for Grokking (ICLR 2023)
- Finding Neurons in a Haystack
- Automated Circuit Discovery
- Othello-GPT Linear Emergent World Representation

---

## 3. Token Erasure / Footprints - Multi-Token Concept Analysis

**Repository:** `footprints/`
**GitHub:** https://github.com/sfeucht/footprints
**Paper:** https://arxiv.org/abs/2406.20086
**Website:** https://footprints.baulab.info
**Purpose:** Tools for studying implicit vocabulary and multi-token word representations via token erasure

### Key Capabilities
- Discover "token erasure" patterns in named entities and multi-token words
- Segment documents to identify high-scoring token sequences
- Train and test linear probes on hidden states
- "Read out" implicit vocabulary from autoregressive LLMs
- Pre-trained probes available for Llama-2-7b and Llama-3-8b

### Entry Points
- **Segment a document:** `segment.py --document my_doc.txt --model meta-llama/Llama-2-7b-hf`
- **Read out vocabulary:** `readout.py --model meta-llama/Meta-Llama-3-8B --dataset data/wikipedia_test_500.csv`
- **Train probes:** `scripts/train_probe.py --layer 12 --target_idx -2`
- **Test probes:** `scripts/test_probe.py --checkpoint checkpoints/... --test_data counterfact_expanded.csv`

### Dependencies (from requirements.txt)
- Core: torch 2.3.1, transformers 4.41.2
- NLP: spacy 3.7.5, sentencepiece 0.2.0
- Analysis: nnsight 0.2.19, pandas, numpy
- Utilities: tqdm, wandb 0.17.3

### Installation
```bash
cd footprints
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Pre-trained Probes
Available at https://huggingface.co/sfeucht/footprints for Llama-2-7b and Llama-3-8b

### Datasets Included
- CounterFact: `data/counterfact_expanded.csv`
- Pile: `data/train_tiny_1000.csv`, `data/val_tiny_500.csv`, `data/test_tiny_500.csv`
- Wikipedia: `data/wikipedia_test_500.csv`

---

## 4. Linear Representation Geometry

**Repository:** `linear_rep_geometry/`
**GitHub:** https://github.com/KihoPark/linear_rep_geometry
**Paper:** https://arxiv.org/abs/2311.03658
**Purpose:** Tools for studying the linear representation hypothesis and causal inner products

### Key Capabilities
- Formalize the linear representation hypothesis in LLMs
- Define and compute causal inner products respecting semantic structure
- Estimate unembedding representations from counterfactual word pairs
- Measure concept directions and their orthogonality
- Perform intervention experiments on concept representations

### Experiments (Jupyter Notebooks)
1. **`1_subspace.ipynb`** - Compare projections of counterfactual pairs onto concept directions
2. **`2_heatmap.ipynb`** - Visualize orthogonality between concepts using causal inner product
3. **`3_measurement.ipynb`** - Verify concept directions act as linear probes
4. **`4_intervention.ipynb`** - Test interventions on target concepts without affecting off-target concepts
5. **`5_sanity_check.ipynb`** - Verify causal inner product satisfies theoretical assumptions

### Entry Points
- **Setup:** Run `store_matrices.py` first to create matrices directory
- **Data:** Counterfactual word pairs in `word_pairs/` directory
- **Contexts:** Paired multilingual contexts in `paired_contexts/` directory

### Dependencies
- Core: transformers, torch, numpy
- Visualization: seaborn, matplotlib
- Utilities: json, tqdm
- Hardware: GPU recommended for efficient implementation

### Installation
```bash
cd linear_rep_geometry
# Install required packages
pip install transformers torch numpy seaborn matplotlib tqdm
# Create matrices directory and run setup
mkdir matrices
python store_matrices.py
```

---

## 5. SAE Spelling/Absorption - Feature Splitting and Absorption

**Repository:** `sae-spelling/`
**GitHub:** https://github.com/lasr-spelling/sae-spelling
**Paper:** https://arxiv.org/abs/2409.14507
**Purpose:** Study feature splitting and absorption in Sparse Autoencoders

### Key Capabilities
- **Feature Attribution:** Calculate SAE feature attribution using integrated gradients
- **Feature Ablation:** Ablate individual SAE latents to measure downstream effects
- **Probing:** Train logistic regression probes on SAE features
- **Spelling Tasks:** Generate and grade spelling prompts (ICL format)
- **Absorption Analysis:** Quantify feature absorption across different SAEs

### Code Organization
- `sae_spelling.feature_attribution` - Attribution experiments
- `sae_spelling.feature_ablation` - Ablation experiments
- `sae_spelling.probing` - Train multi-class and binary probes
- `sae_spelling.prompting` - ICL prompt generation for spelling
- `sae_spelling.vocab` - Token vocabulary utilities
- `sae_spelling.sae_utils` - Apply SAEs and run inference
- `sae_spelling.experiments` - Main paper experiments

### Experiments
- `latent_evaluation` - Compare top SAE latents vs. LR probes on first-letter tasks
- `k_sparse_probing` - Train k-sparse probes to detect feature splitting
- `feature_absorption` - Quantify absorption on first-letter tasks

### Dependencies
- Python: ^3.10
- Core: sae-lens ^3.16.0, transformer-lens ^2.2.2, torch
- Visualization: matplotlib, seaborn, tueplots
- Dev: ruff, pytest, pyright, pre-commit

### Installation
```bash
cd sae-spelling
poetry install
```

### Toy Models
Available in Colab: https://colab.research.google.com/drive/1MMKKGxHk34Q823hBHbhYMW5ArvCqUhHw

---

## 6. Neuronpedia - SAE Feature Browser and Analysis Platform

**Repository:** `neuronpedia/`
**GitHub:** https://github.com/hijohnnylin/neuronpedia
**Website:** https://neuronpedia.org
**Purpose:** Open source interpretability platform for browsing and analyzing SAE features

### Key Capabilities
- Browse 4+ terabytes of activations, explanations, and metadata
- API for feature access and analysis
- Steering, activation testing, and inference
- Circuit/graph visualization with attribution graphs
- Auto-interpretation and scoring using EleutherAI's Delphi
- Search and filter features by semantic similarity
- Dashboards, benchmarks, exports, and uploads
- Support for probes, latents/features, custom vectors, concepts

### Architecture (Microservices)
1. **Webapp** - Next.js/React frontend + API (localhost:3000)
2. **Database** - PostgreSQL for features, activations, explanations, users
3. **Inference** - Python/Torch server for model inference (localhost:5002)
4. **Autointerp** - Python server using EleutherAI's Delphi for auto-interpretation
5. **Graph** - Circuit tracer for attribution graph generation

### Entry Points
- **Local Setup:** `make webapp-localhost-build && make webapp-localhost-run`
- **Development:** `make webapp-localhost-dev`
- **Inference:** `make inference-localhost-dev MODEL_SOURCESET=gpt2-small.res-jb`
- **Admin Panel:** http://localhost:3000/admin (for importing data)

### Dependencies
- Frontend: Node.js, Next.js, React, TypeScript
- Backend: Python, Poetry, FastAPI
- Database: PostgreSQL
- ML: torch, transformers, transformer-lens, sae-lens

### Pre-loaded Inference Configs
- `gpt2-small.res-jb`
- `gemma-2-2b-it.gemmascope-res-16k`
- `deepseek-r1-distill-llama-8b.llamascope-slimpj-res-32k`

### Installation
See detailed setup instructions in `neuronpedia/README.md` for:
- Local database setup
- Webapp development
- Inference server configuration
- Graph server setup
- Autointerp server setup

---

## 7. SAEDashboard - Feature Visualization at Scale

**Repository:** `SAEDashboard/`
**GitHub:** https://github.com/jbloomAus/SAEDashboard
**Purpose:** Visualize and analyze SAE features with Anthropic-style dashboards

### Key Capabilities
- Generate feature-centric visualizations (activations, logits, correlations)
- Support for any SAE in SAELens library
- Neuronpedia integration for hosting dashboards
- Handle large datasets and models efficiently
- Cross-Layer Transcoder (CLT) support

### Entry Points
- **Basic Usage:**
  ```python
  from sae_dashboard.sae_vis_runner import SaeVisRunner
  from sae_dashboard.sae_vis_data import SaeVisConfig

  config = SaeVisConfig(hook_point="...", features=[...])
  data = SaeVisRunner(config).run(encoder=sae, model=model, tokens=tokens)
  ```
- **Neuronpedia Runner:** Generate dashboards for Neuronpedia at scale
- **Demo Notebook:** https://colab.research.google.com/drive/1oqDS35zibmL1IUQrk_OSTxdhcGrSS6yO

### Dependencies
- Core: sae-lens, transformer-lens, torch
- Visualization: plotly
- Dev: Poetry, pytest, ruff, pyright

### Installation
```bash
cd SAEDashboard
pip install sae-dashboard
# or for development:
poetry install
```

---

## Research Application: Studying "Washing Machine" Representations

### Relevant Techniques from These Repositories

1. **Token Erasure Analysis (footprints)**
   - Segment "washing machine" to measure erasure patterns
   - Train probes to detect multi-token representations
   - Compare with single-token concepts

2. **SAE Feature Analysis (SAELens + SAEDashboard)**
   - Train SAEs on layers processing "washing machine"
   - Identify features that activate for compound concepts
   - Analyze feature splitting vs. absorption

3. **Linear Representation Geometry (linear_rep_geometry)**
   - Compute concept directions for "washing" vs. "machine" vs. "washing machine"
   - Measure orthogonality and compositionality
   - Test intervention effects

4. **Probing Classifiers (TransformerLens + footprints)**
   - Train probes to detect "washing machine" vs. component words
   - Compare probe performance across layers
   - Identify where compound concept emerges

5. **Feature Visualization (Neuronpedia + SAEDashboard)**
   - Browse pre-computed features for compound concepts
   - Generate custom dashboards for "washing machine" features
   - Analyze activation patterns and top examples

### Suggested Workflow

1. **Data Collection**
   - Collect corpus with "washing machine" and variants
   - Include separate "washing" and "machine" examples
   - Prepare counterfactual pairs

2. **Token Erasure Analysis**
   - Use `footprints/segment.py` to analyze tokenization patterns
   - Train probes on "washing machine" representations
   - Measure erasure scores across layers

3. **SAE Training and Analysis**
   - Train SAEs on relevant layers using SAELens
   - Generate dashboards with SAEDashboard
   - Identify features specific to compound concept

4. **Linear Analysis**
   - Compute concept directions using linear_rep_geometry approach
   - Test compositionality: is "washing machine" = "washing" + "machine"?
   - Measure causal inner products

5. **Visualization and Exploration**
   - Use Neuronpedia to browse related features
   - Compare with existing compound concepts
   - Document findings with feature dashboards

---

## Quick Start Guide

### Prerequisites
```bash
# Python 3.10+ recommended
python --version

# Install Poetry (for most repos)
curl -sSL https://install.python-poetry.org | python3 -

# Install Node.js (for Neuronpedia)
# Follow instructions at https://nodejs.org/
```

### Basic Example: Loading and Analyzing an SAE

```python
# Install SAELens and TransformerLens
# pip install sae-lens transformer-lens

from sae_lens import SAE
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# Load pre-trained SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device="cuda"
)

# Analyze activations for "washing machine"
text = "I bought a new washing machine yesterday."
logits, cache = model.run_with_cache(text)
activations = cache[sae.cfg.hook_name]

# Encode with SAE
feature_acts = sae.encode(activations)
print(f"Active features: {(feature_acts > 0).sum()}")
print(f"Top features: {feature_acts.topk(10).indices}")
```

### Basic Example: Token Erasure Analysis

```bash
cd footprints

# Segment a document
python segment.py \
    --document my_text.txt \
    --model meta-llama/Llama-2-7b-hf \
    --output_html

# Results saved to logs/html/
```

---

## Additional Resources

### Papers
- **SAELens:** Bloom et al. (2024) - https://github.com/decoderesearch/SAELens
- **TransformerLens:** Nanda & Bloom (2022) - https://github.com/TransformerLensOrg/TransformerLens
- **Token Erasure:** Feucht et al. (2024) - https://arxiv.org/abs/2406.20086
- **Linear Rep Geometry:** Park et al. (2023) - https://arxiv.org/abs/2311.03658
- **SAE Spelling/Absorption:** Chanin et al. (2024) - https://arxiv.org/abs/2409.14507

### Communities
- **Open Source Mech Interp Slack:** https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-375zalm04-GFd5tdBU1yLKlu_T_JSqZQ
- **Neuronpedia Slack:** https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-3m2fulfeu-0LnVnF8yCrKJYQvWLuCQaQ

### Documentation
- SAELens: https://decoderesearch.github.io/SAELens/
- TransformerLens: https://transformerlensorg.github.io/TransformerLens/
- Neuronpedia: https://docs.neuronpedia.org/
- Token Erasure: https://footprints.baulab.info/

---

## Notes

- All repositories cloned with `--depth 1` for space efficiency
- Most repos use Poetry for dependency management
- GPU recommended for all SAE and model inference tasks
- Neuronpedia requires Docker for full local setup
- Linear rep geometry repo requires manually setting MODEL_PATH in code

## Contributing

Each repository has its own contributing guidelines. See individual README files for details.

## License

Most repositories use MIT License. Check individual LICENSE files for specifics.
