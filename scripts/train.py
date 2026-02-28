from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import json

# Load limited dataset
def load_jsonl(path, limit=500):
    # Read JSONL file manually to avoid caching issues
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data.append(json.loads(line.strip()))
    
    # Create dataset from the loaded data
    return Dataset.from_list(data)

x = 500  # <-- Number of samples to use

print("Loading datasets...")
train_dataset = load_jsonl("drive/MyDrive/ColabNotebooks/Baymax-ai/data/processed/medmcqa/train.jsonl", limit=x)
val_dataset = load_jsonl("drive/MyDrive/ColabNotebooks/Baymax-ai/data/processed/medmcqa/validation.jsonl", limit=x)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Load tokenizer and model
print("Loading model and tokenizer...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Add pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Apply LoRA
print("Applying LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "dense"]  # Added more target modules
)
model = get_peft_model(model, peft_config)

# Print trainable parameters
model.print_trainable_parameters()

# Tokenization function
def tokenize(example):
    prompt = f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['response']}"
    tokenized = tokenizer(
        prompt, 
        truncation=True, 
        padding="max_length",  # Pad to max_length to ensure consistent sizes
        max_length=512
    )
    # Add labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Data collator - simpler version since we're pre-padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results-phi2-medmcqa-mini",
    per_device_train_batch_size=2,  # Reduced batch size to avoid OOM
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,  # Effective batch size = 2 * 2 = 4
    eval_steps=50,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",  # Changed from evaluation_strategy
    save_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    dataloader_pin_memory=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    remove_unused_columns=False,
    warmup_steps=50,
    learning_rate=5e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine"
)

# Trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
model.save_pretrained("./baymax-phi2-mini-lora")
tokenizer.save_pretrained("./baymax-phi2-mini-lora")

print("Training completed!")