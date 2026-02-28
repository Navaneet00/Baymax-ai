from datasets import load_dataset
import os
import json

# Method 1: Load with download_mode="force_redownload" to bypass cache
try:
    dataset = load_dataset(
        "json", 
        data_files="drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/processed/medmcqa/train.jsonl", 
        split="train",
        download_mode="force_redownload"
    )
    print(f"Dataset loaded successfully with {len(dataset)} examples")
    
except Exception as e:
    print(f"Method 1 failed: {e}")
    print("Trying alternative method...")
    
    # Method 2: Manual chunking without using datasets library
    chunk_size = 500
    chunk_counter = 0
    current_chunk = []
    
    os.makedirs("drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/chunks", exist_ok=True)
    
    with open("drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/processed/medmcqa/train.jsonl", 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                current_chunk.append(data)
                
                if len(current_chunk) >= chunk_size:
                    # Save current chunk
                    output_file = f"drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/chunks/train_chunk_{chunk_counter}.jsonl"
                    with open(output_file, 'w') as out_f:
                        for item in current_chunk:
                            json.dump(item, out_f)
                            out_f.write('\n')
                    
                    print(f"Saved chunk {chunk_counter} with {len(current_chunk)} examples")
                    current_chunk = []
                    chunk_counter += 1
                    
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {line_num + 1}: {e}")
                continue
    
    # Save remaining items in the last chunk
    if current_chunk:
        output_file = f"drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/chunks/train_chunk_{chunk_counter}.jsonl"
        with open(output_file, 'w') as out_f:
            for item in current_chunk:
                json.dump(item, out_f)
                out_f.write('\n')
        print(f"Saved final chunk {chunk_counter} with {len(current_chunk)} examples")
    
    print(f"Total chunks created: {chunk_counter + 1}")
    exit()

# If Method 1 succeeded, continue with original chunking logic
chunk_size = 500
total_chunks = len(dataset) // chunk_size + 1

os.makedirs("drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/chunks", exist_ok=True)

for i in range(total_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(dataset))
    
    chunk = dataset.select(range(start_idx, end_idx))
    output_file = f"drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/chunks/train_chunk_{i}.jsonl"
    
    # Convert to list of dictionaries and save as JSONL
    chunk_data = [item for item in chunk]
    with open(output_file, 'w') as f:
        for item in chunk_data:
            json.dump(item, f)
            f.write('\n')
    
    print(f"Saved chunk {i} with {len(chunk_data)} examples")

print(f"Successfully created {total_chunks} chunks")