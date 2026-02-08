# Where is "Washing Machine" Stored in LLMs?

An empirical investigation into how compound concepts are represented in large language models. There are far more referenceable concepts than available dimensions in an LLM's residual stream — so how does a model represent multi-token concepts like "washing machine"? Does it have a unique direction, or does it just store "washing" and let context predict "machine"?

## Key Findings

- **Next-token prediction is the primary mechanism**: After seeing "washing", GPT-2 predicts "machine" as the #1 token with P=0.827 — a 4963x boost over the control word. Median boost across 18 compounds: 20.2x (p=2.1e-4).
- **Compound directions are 94% reconstructable from constituents**: The compound representation at the word2 position is a linear combination of word1 + word2 directions (R²=0.937 ± 0.020). No dedicated "washing machine" direction exists.
- **More compositional compounds have higher R²**: Spearman r=0.669, p=0.006. Idiomatic compounds like "hot dog" (R²=0.888) retain more unique information than compositional ones like "steel bridge" (R²=0.965).
- **No token erasure in GPT-2**: Unlike Feucht et al.'s findings for named entities in Llama-2-7B, word1 identity is perfectly recoverable from the word2 position at every layer.
- **Compound contexts are still distinguishable**: Despite high R², a linear probe can tell compound from non-compound contexts with 92.2% accuracy (the ~6% unique component carries compound-specific information).
- **Findings replicate on GPT-2-medium (355M)**: Boost ratios and R² values are consistent across model sizes.

## Reproducing Results

### Environment Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add torch transformers transformer-lens matplotlib seaborn scipy scikit-learn pandas numpy tqdm einops
```

### Running Experiments
```bash
# Main experiments (GPT-2, ~3 min on GPU)
python src/experiment.py

# Statistical analysis and visualizations
python src/analysis.py

# Validation on GPT-2-medium (~2 min on GPU)
python src/validation_gpt2medium.py
```

### Hardware Requirements
- GPU recommended (RTX 3090 or similar; 8GB+ VRAM sufficient for GPT-2)
- CPU-only: experiments will run but slower (~15 min)
- Disk: ~500MB for model downloads

## File Structure
```
.
├── REPORT.md                          # Full research report with results
├── README.md                          # This file
├── planning.md                        # Research plan and methodology
├── literature_review.md               # Literature review
├── resources.md                       # Resource catalog
├── src/
│   ├── experiment.py                  # Main experiments (4 experiments)
│   ├── analysis.py                    # Statistical analysis & visualizations
│   └── validation_gpt2medium.py       # GPT-2-medium validation
├── results/
│   ├── config.json                    # Experiment configuration
│   ├── exp1_next_token.json           # Experiment 1 results
│   ├── exp2_residual_directions.json  # Experiment 2 results
│   ├── exp3_probing.json              # Experiment 3 results
│   ├── exp4_attention.json            # Experiment 4 results
│   ├── validation_gpt2medium.json     # Validation results
│   ├── statistical_analysis.json      # All statistical tests
│   └── plots/
│       ├── summary_figure.png         # Key findings summary
│       ├── exp1_*.png                 # Experiment 1 plots
│       ├── exp2_*.png                 # Experiment 2 plots
│       ├── exp3_*.png                 # Experiment 3 plots
│       └── exp4_*.png                 # Experiment 4 plots
├── papers/                            # Downloaded research papers (36)
├── datasets/                          # Compound noun test sets
└── code/                              # Cloned reference repositories
```

See [REPORT.md](REPORT.md) for the full research report with all experimental results, statistical analysis, and discussion.
