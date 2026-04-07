import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Core experiment logic
def run_context_experiment(model_sizes=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]):
    all_results = []

    # Load test cases
    with open('test_cases.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)['test_cases']

    for size in model_sizes:
        print(f"📡 Loading model: {size}...")
        tokenizer = GPT2Tokenizer.from_pretrained(size)
        model = GPT2LMHeadModel.from_pretrained(size)
        model.eval()

        # Get baseline probability distribution (after "The washing")
        baseline_text = "The washing"
        base_inputs = tokenizer(baseline_text, return_tensors="pt")
        with torch.no_grad():
            base_logits = model(**base_inputs).logits[0, -1, :]
            base_probs = F.softmax(base_logits, dim=-1).cpu().numpy()

        for case in test_cases:
            context = case['input_text']
            # Key: GPT-2 requires leading space when predicting next token
            target_str = " " + case['target_token'].strip()
            target_id = tokenizer.encode(target_str)[0]
            machine_id = tokenizer.encode(" machine")[0]

            inputs = tokenizer(context, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = F.softmax(logits, dim=-1)

            # Extract metrics
            greedy_id = torch.argmax(probs).item()
            greedy_token = tokenizer.decode([greedy_id])

            p_target = probs[target_id].item()
            p_machine = probs[machine_id].item()
            case_probs_np = probs.cpu().numpy()

            # Calculate JSD (Jensen-Shannon Divergence)
            jsd_val = jensenshannon(base_probs, case_probs_np)
            
            all_results.append({
                "Model": size,
                "Case_ID": case['id'],
                "Target": case['target_token'],
                "Greedy_Output": repr(greedy_token),  # Use repr to preserve spaces
                "Is_Machine": "machine" in greedy_token.lower(),
                "P_Target": p_target,
                "P_Machine": p_machine,
                "Confidence_Gap": p_target - p_machine,
                "JSD": jsd_val,
                "Correct": greedy_token.strip().lower() == case['target_token'].strip().lower()
            })

    return pd.DataFrame(all_results)

# 2. Run experiment and generate CSV
df = run_context_experiment()
df.to_csv("gpt2_washing_detailed_results.csv", index=False)

# 3. Summary statistics
summary = df.groupby("Model").agg({
    "Is_Machine": "mean",        # ① Frequency of greedy choosing 'machine'
    "P_Machine": "mean",         # ① Average probability assigned to ' machine'
    "JSD": "mean",               # ② Average JSD (context sensitivity)
    "Correct": "mean",           # ③ Ground truth accuracy
    "Confidence_Gap": "mean"     # ③ Confidence gap (prediction confidence indicator)
}).reindex([“gpt2”, “gpt2-medium”, “gpt2-large”, “gpt2-xl”]).reset_index()

summary.to_csv(“gpt2_washing_summary.csv”, index=False)
print(“\n✅ Experiment Complete. Files saved: 'gpt2_washing_detailed_results.csv' and 'gpt2_washing_summary.csv'”)

# 4. Plotting and save visualization (2x2 layout)
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Rate of greedy choosing 'machine' (reflects bias strength)
# Higher value indicates model more likely to default to 'machine'
sns.barplot(ax=axes[0, 0], data=df, x=”Model”, y=”Is_Machine”, palette=”Reds_r”)
axes[0, 0].set_title(“Rate of Greedy Choosing 'Machine' (Bias Rate)”)
axes[0, 0].set_ylabel(“Frequency (0.0 - 1.0)”)

# Plot 2: Confidence gap (reflects context vs. priming conflict)
# Values closer to or exceeding 0 indicate context overcoming statistical bias
sns.lineplot(ax=axes[0, 1], data=df, x=”Model”, y=”Confidence_Gap”, marker='o', errorbar='sd', color='forestgreen')
axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title(“Average Confidence Gap (P_target - P_machine)”)

# Plot 3: JSD trend (reflects context sensitivity)
sns.barplot(ax=axes[1, 0], data=df, x=”Model”, y=”JSD”, palette=”Blues_d”)
axes[1, 0].set_title(“Context Sensitivity (JSD vs Baseline)”)

# Plot 4: Ground truth accuracy (final task performance)
sns.barplot(ax=axes[1, 1], data=df, x=”Model”, y=”Correct”, palette=”magma”)
axes[1, 1].set_title(“Ground Truth (Target Token) Accuracy”)

plt.tight_layout()
plt.savefig("washing_experiment_analysis.png", dpi=300)
print("✅ Visualization saved as 'washing_experiment_analysis.png' with 4 key metrics.")