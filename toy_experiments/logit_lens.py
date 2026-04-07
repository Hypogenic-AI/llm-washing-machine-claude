import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Environment setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "gpt2-xl"
print(f"📦 Loading {model_name}...")

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

output_dir = "logit_lens_results"
os.makedirs(output_dir, exist_ok=True)

# 2. Get token ID (ensure space-prefixed)
def get_id(w):
    return tokenizer.encode(w)[0]

# 3. Core computation function
def get_case_data(case):
    text = case['input_text']
    target_word = case['target_token']
    
    target_id = get_id(" " + target_word)
    machine_id = get_id(" machine")
    ton_id = get_id("ton")
    washing_id = get_id(" washing")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    ln_f = model.transformer.ln_f
    unembedding = model.lm_head
    
    history = []
    for i, h in enumerate(hidden_states):
        last_token_v = h[0, -1, :]
        if i < len(hidden_states)-1:
            last_token_v = ln_f(last_token_v)
        logits = unembedding(last_token_v)
        probs = F.softmax(logits, dim=-1)
        
        top_id = torch.argmax(probs).item()
        history.append({
            "layer": i,
            "p_target": probs[target_id].item(),
            "p_machine": probs[machine_id].item(),
            "p_ton": probs[ton_id].item(),
            "p_washing": probs[washing_id].item(),
            "top_token": tokenizer.decode([top_id])
        })
    return pd.DataFrame(history), target_word, history[-1]["top_token"]

# 4. Plotting logic
def run_experiment(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)['test_cases'][:15]

    # Prepare summary plot (5 rows x 3 cols)
    fig_all, axes = plt.subplots(5, 3, figsize=(24, 30))
    axes = axes.flatten()

    all_data = []

    for idx, case in enumerate(cases):
        df, target_str, final_pred = get_case_data(case)
        all_data.append((df, target_str, final_pred, case['input_text']))

        # Plot individual case
        plot_single(idx+1, case['input_text'], target_str, final_pred, df)

        # Plot in summary grid
        plot_subplot(axes[idx], idx+1, target_str, final_pred, df)

    # Save summary plot
    fig_all.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_all.suptitle("GPT2-XL Logit Lens: 15 Cases Overview", fontsize=30, fontweight='bold')
    fig_all.savefig(os.path.join(output_dir, "summary_5x3_all_cases.png"), dpi=150)
    plt.close(fig_all)
    print("✅ Summary chart (5x3) saved.")

def plot_single(c_id, text, target, pred, df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer'], df['p_target'], label=f"Target: '{target}'", color='green', lw=3)
    plt.plot(df['layer'], df['p_machine'], label="Bias: ' machine'", color='red', ls='--')
    plt.plot(df['layer'], df['p_ton'], label="Bias: ' ton'", color='blue', ls='-.')
    plt.plot(df['layer'], df['p_washing'], label="Self: ' washing'", color='gray', alpha=0.3)
    
    plt.title(f"Case {c_id} | Final Pred: '{pred.strip()}'\nContext: {text[:60]}...")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"case_{c_id}.png"))
    plt.close()

def plot_subplot(ax, c_id, target, pred, df):
    ax.plot(df['layer'], df['p_target'], color='green', lw=2, label='Target')
    ax.plot(df['layer'], df['p_machine'], color='red', ls='--', label='Machine')
    ax.plot(df['layer'], df['p_ton'], color='blue', ls='-.', label='Ton')
    ax.set_title(f"#{c_id} GT:'{target.strip()}' | Pred:'{pred.strip()}'", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    if c_id == 1: ax.legend()

if __name__ == "__main__":
    run_experiment('test_cases.json')