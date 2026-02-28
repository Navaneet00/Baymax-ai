# Simple Validation Harness for Baymax AI

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
from tqdm import tqdm
from collections import defaultdict
import re
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_dir = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/baymax-phi2-chunked"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Ensure pad token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load validation samples
validation_path = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/processed/medmcqa/validation.jsonl"
samples_by_subject = defaultdict(list)

with open(validation_path, 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line.strip())
        if sample.get("response"):
            subject = sample["response"].split("Subject:")[-1].strip() if "Subject:" in sample["response"] else "Unknown"
            samples_by_subject[subject].append(sample)

# Sample max 10 questions per subject
eval_samples = []
for subject, samples in samples_by_subject.items():
    eval_samples.extend(random.sample(samples, min(5, len(samples))))

# Answer extraction logic
def extract_answer(text):
    patterns = [
        r'\bAnswer\s*[:\-]?\s*([A-D])\b',
        r'The correct answer is\s*([A-D])\b',
        r'The correct option is\s*([A-D])\b',
        r'Option\s*([A-D])\s*(is correct|is the answer)?\b',
        r'^([A-D])\.',
        r'\n([A-D])\.'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None

def extract_subject(text):
    match = re.search(r'Subject\s*[:\-]?\s*(.+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "Unknown"

def extract_response_only(decoded_output):
    if "Response:" in decoded_output:
        return decoded_output.split("Response:", 1)[1].strip()
    return decoded_output.strip()

# Evaluation loop
overall_correct = 0
overall_total = 0
subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
logs = []

model.eval()
for sample in tqdm(eval_samples, desc="Evaluating"):
    instruction = sample["instruction"]
    reference_response = sample["response"]
    expected_answer = extract_answer(reference_response)
    subject = extract_subject(reference_response)

    # ✅ Proper f-string usage for instruction
    prompt = (
        f"Properly and accurately answer for the given question with proper explanation.\n\n"
        f"Instruction:\n{instruction}\n\nResponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5,
                top_p=1.0,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = extract_response_only(decoded_output)
        predicted_answer = extract_answer(response_only)
        is_correct = predicted_answer == expected_answer

        overall_total += 1
        subject_stats[subject]["total"] += 1
        if is_correct:
            overall_correct += 1
            subject_stats[subject]["correct"] += 1
            print(f"\n✅ Correct!")
        else:
            print(f"\n❌ Mismatch!")
        
        print(f"Instruction: {instruction}")
        print(f"Expected: {expected_answer} | Predicted: {predicted_answer}")
        print(f"Response:\n{response_only}\n")

        logs.append({
            "instruction": instruction,
            "subject": subject,
            "expected": expected_answer,
            "predicted": predicted_answer,
            "correct": is_correct,
            "full_response": response_only
        })

    except Exception as e:
        print(f"⚠️ Error processing sample: {e}")
        continue

# Final Stats
accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0
print(f"\n✅ Overall Accuracy: {accuracy:.2f}% ({overall_correct}/{overall_total})")

print("\n📚 Subject-wise Accuracy:")
for subject, stats in sorted(subject_stats.items()):
    total = stats["total"]
    correct = stats["correct"]
    acc = correct / total * 100 if total > 0 else 0
    print(f"{subject:<25} {acc:5.2f}% ({correct}/{total})")

# Save logs
output_path = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/baymax_validation_results_no_rag.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for entry in logs:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")