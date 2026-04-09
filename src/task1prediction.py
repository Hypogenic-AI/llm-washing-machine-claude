from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch # Loading GPT2 


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

model.eval()


# Compound word pairs
pairs = [
    ("washing", "machine"),
    ("ice", "cube"),
    ("credit", "card"),
    ("fire", "truck"),
    ("sun", "flower"),
    ("rain", "bow"),
    ("black", "board"),
    ("book", "store"),
    ("door", "step"),
    ("cooking", "oil"),
]

correct = 0
total = len(pairs)


for first, second in pairs:
    inputs = tokenizer(f"The {first}", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    next_token_logits = outputs.logits[0, -1, :]
    top_token_id = torch.argmax(next_token_logits).item()
    predicted = tokenizer.decode(top_token_id).strip().lower()
    
    match = second.lower() == predicted
    if match:
        correct += 1
    print(f"'{first}' → predicted: '{predicted}' | {'✅' if match else '❌'}")

print(f"Score: {correct}/{total} = {correct/total*100:.1f}%")