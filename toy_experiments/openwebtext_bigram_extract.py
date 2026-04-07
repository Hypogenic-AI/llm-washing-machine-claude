import json
from collections import Counter
from pathlib import Path
import pickle

print("="*70)
print("OpenWebText 'Washing' Bigram Extraction Only")
print("="*70)

try:
    from datasets import load_dataset

    print("\nLoading OpenWebText dataset...")
    # Use streaming mode to handle large datasets
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    dataset = dataset.take(1000000)
    print("Loaded (streaming mode)\n")

    washing_bigrams = Counter()
    doc_count = 0

    print("Processing documents (filtering for 'washing *')...\n")

    for doc in dataset:
        text = doc["text"].lower()
        words = text.split()

        # Iterate through words to find bigrams starting with 'washing'
        for i in range(len(words) - 1):
            w1 = words[i].strip()

            # Clean punctuation and convert to pure token
            w1_clean = ''.join(c for c in w1 if c.isalnum() or c in ['-', "'"])

            if w1_clean == "washing":
                w2 = words[i+1].strip()
                w2_clean = ''.join(c for c in w2 if c.isalnum() or c in ['-', "'"])

                if w2_clean:
                    washing_bigrams[w2_clean] += 1

        doc_count += 1
        if doc_count % 1000 == 0:
            print(f"  Processed {doc_count:,} documents... (unique 'washing' followers: {len(washing_bigrams)})")

    print(f"\n✓ Processed {doc_count:,} documents total")
    print(f"✓ Total unique tokens following 'washing': {len(washing_bigrams)}\n")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# Output results
# ============================================================

print("="*70)
print("BIGRAMS STARTING WITH 'WASHING' (Ranked by Frequency)")
print("="*70 + "\n")

washing_sorted = washing_bigrams.most_common()

for rank, (word, count) in enumerate(washing_sorted, 1):
    # Print top 100 most common; rest are usually low-frequency tail
    print(f"{rank:3d}. washing {word:20s} | count: {count:6d}")
    if rank >= 100:
        print("...")
        break

# ============================================================
# Save results
# ============================================================

results = {
    "corpus": "OpenWebText train split (first 1M docs)",
    "stats": {
        "total_documents_scanned": doc_count,
        "unique_washing_followers": len(washing_bigrams)
    },
    "washing_bigrams": [
        {"word": w, "count": count}
        for w, count in washing_sorted
    ]
}

# Ensure output path exists or modify based on environment
output_dir = Path("/home/jiahel19/washing_machine/toy/")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = output_dir / "washing_only_bigrams.json"
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {json_path}")