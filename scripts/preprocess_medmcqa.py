import pandas as pd
import json
import os

def convert_row(row, split):
    try:
        question = str(row['question']).strip()
        opa = str(row['opa']).strip()
        opb = str(row['opb']).strip()
        opc = str(row['opc']).strip()
        opd = str(row['opd']).strip()
        cop = str(row['cop']).strip().lower()
        explanation = str(row.get('exp', '')).strip()
        subject = str(row.get('subject_name', '')).strip()
        topic = str(row.get('topic_name', '')).strip()
    except Exception as e:
        print("Data conversion error:", e)
        return None

    if not question or not opa or not opb or not opc or not opd or not cop:
        return None

    options = {
        "opa": ("A", opa),
        "opb": ("B", opb),
        "opc": ("C", opc),
        "opd": ("D", opd)
    }

    cop_map_text = {'a': 'opa', 'b': 'opb', 'c': 'opc', 'd': 'opd'}
    cop_map_0 = {'0': 'opa', '1': 'opb', '2': 'opc', '3': 'opd'}
    cop_map_1 = {'1': 'opa', '2': 'opb', '3': 'opc', '4': 'opd'}

    if cop in options:
        correct_label, correct_text = options[cop]
    elif cop in cop_map_text:
        correct_label, correct_text = options[cop_map_text[cop]]
    elif cop in cop_map_0:
        correct_label, correct_text = options[cop_map_0[cop]]
    elif cop in cop_map_1:
        correct_label, correct_text = options[cop_map_1[cop]]
    else:
        return None

    instruction = f"{question}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"
    
    if not explanation or explanation.lower() == "none":
        explanation = "No explanation provided."

    response_parts = [
        f"Answer: {correct_label}. {correct_text}",
        f"Explanation: {explanation}"
    ]

    if split == "validation":
        if subject and subject.lower() != "none":
            response_parts.append(f"Subject: {subject}")
        if topic and topic.lower() != "none":
            response_parts.append(f"Topic: {topic}")

    response = "\n".join(response_parts)

    return {
        "instruction": instruction,
        "response": response
    }

def process_parquet_file(file_path, is_test):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Print first few cop values to understand the format
    print("Sample cop values:", df['cop'].head(10).tolist())
    print("Unique cop values:", sorted(df['cop'].unique()))
    print("Value counts for cop:")
    print(df['cop'].value_counts())
    
    examples = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            if is_test:
                example = convert_row_inference_only(row)
            else:
                example = example = convert_row(row, split=os.path.basename(file_path).split('.')[0])
                
            if example:
                examples.append(example)
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped += 1
    
    print(f"Processed: {len(examples)} examples, Skipped: {skipped}")
    return examples

def save_jsonl(examples, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write('\n')

# For test.parquet dataset
def convert_row_inference_only(row):
    # For test set - create examples without correct answers
    try:
        question = str(row['question']).strip()
        opa = str(row['opa']).strip()
        opb = str(row['opb']).strip()
        opc = str(row['opc']).strip()
        opd = str(row['opd']).strip()
    except Exception as e:
        return None

    if not question or not opa or not opb or not opc or not opd:
        return None

    instruction = f"{question}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}"

    return {
        "instruction": instruction.strip(),
        "response": ""  # No answer provided
    }

# Paths — change if needed
input_dir = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/medmcqa/data"  # where your .parquet files are
output_dir = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/processed/medmcqa"
os.makedirs(output_dir, exist_ok=True)

# Process train and validation splits
for split in ['train', 'validation']:
    print(f"\nProcessing {split} split...")
    in_file = os.path.join(input_dir, f"{split}.parquet")
    out_file = os.path.join(output_dir, f"{split}.jsonl")
    
    examples = process_parquet_file(in_file, False)
    
    if examples:
        save_jsonl(examples, out_file)
        print(f"{split}: {len(examples)} examples saved to {out_file}")
        
        # Print a sample example
        if examples:
            print("Sample example:")
            print(json.dumps(examples[0], indent=2))
    else:
        print(f"{split}: No examples to save")

# Process test splits
for split in ['test']:
    print(f"\nProcessing {split} split...")
    in_file = os.path.join(input_dir, f"{split}.parquet")
    out_file = os.path.join(output_dir, f"{split}.jsonl")
    
    examples = process_parquet_file(in_file, True)
    
    if examples:
        save_jsonl(examples, out_file)
        print(f"{split}: {len(examples)} examples saved to {out_file}")
        
        # Print a sample example
        if examples:
            print("Sample example:")
            print(json.dumps(examples[0], indent=2))
    else:
        print(f"{split}: No examples to save")

print("\nProcessing complete!")