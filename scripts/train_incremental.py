import os
import json
import gc
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
from sklearn.metrics import accuracy_score
import logging
from collections import defaultdict
import time

# 🔧 Enhanced Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
torch.backends.cudnn.benchmark = True

# 📊 Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalTrainingTracker:
    """Track subject-wise performance and training metrics"""
    
    def __init__(self):
        self.subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.training_history = []
        
    def update_subject_stats(self, subjects, predictions, labels):
        """Update subject-wise accuracy tracking"""
        for subject, pred, label in zip(subjects, predictions, labels):
            self.subject_stats[subject]['total'] += 1
            if pred == label:
                self.subject_stats[subject]['correct'] += 1
    
    def get_subject_accuracies(self):
        """Get current subject-wise accuracies"""
        accuracies = {}
        for subject, stats in self.subject_stats.items():
            if stats['total'] > 0:
                accuracies[subject] = stats['correct'] / stats['total']
            else:
                accuracies[subject] = 0.0
        return accuracies
    
    def log_progress(self, chunk_number, overall_accuracy):
        """Log training progress"""
        subject_accs = self.get_subject_accuracies()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING PROGRESS - Chunk {chunk_number}")
        logger.info(f"{'='*50}")
        logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")
        logger.info(f"\nTop Performing Subjects:")
        
        # Sort subjects by accuracy
        sorted_subjects = sorted(subject_accs.items(), key=lambda x: x[1], reverse=True)
        
        for subject, acc in sorted_subjects[:5]:
            logger.info(f"  {subject}: {acc:.2%}")
        
        logger.info(f"\nNeeds Improvement:")
        for subject, acc in sorted_subjects[-5:]:
            logger.info(f"  {subject}: {acc:.2%}")

# 🚀 Enhanced Config
chunk_number = 204  # <-- CHANGE this for each run
model_dir = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/baymax-phi2-chunked"
model_checkpoint = model_dir if os.path.exists(model_dir) else "microsoft/phi-2"

# 📈 Training configuration based on Med-PaLM 2 principles
ENHANCED_CONFIG = {
    'max_length': 1024,           # Increased from 512 for better context
    'learning_rate': 5e-5,        # Reduced from 2e-4 for stability
    'batch_size': 4,              # Increased from 2
    'gradient_accumulation': 8,    # Reduced from 16 for more frequent updates
    'num_epochs': 5,              # Increased from 3
    'warmup_ratio': 0.1,          # Better warmup strategy
    'lora_r': 32,                 # Increased from 16
    'lora_alpha': 64,             # Increased from 32
    'lora_dropout': 0.05,         # Reduced from 0.1
}

print(f"🚀 Enhanced Training Configuration:")
print(f"Model: {model_checkpoint}")
print(f"Max Length: {ENHANCED_CONFIG['max_length']}")
print(f"Learning Rate: {ENHANCED_CONFIG['learning_rate']}")
print(f"Effective Batch Size: {ENHANCED_CONFIG['batch_size'] * ENHANCED_CONFIG['gradient_accumulation']}")

# 📦 Load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)

# 📁 Enhanced dataset loading with better validation
def load_jsonl(path):
    """Load JSONL with error handling"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"✅ Loaded {len(data)} examples from {path}")
        return Dataset.from_list(data)
    except Exception as e:
        logger.error(f"❌ Error loading {path}: {e}")
        return Dataset.from_list([])

def create_better_validation_set(val_dataset, min_samples_per_subject=5):
    """Create a more representative validation set"""
    
    # Group by subject
    subject_groups = defaultdict(list)
    for i, example in enumerate(val_dataset):
        subject = example.get('subject', 'Unknown')
        subject_groups[subject].append(i)
    
    # Select balanced samples
    selected_indices = []
    for subject, indices in subject_groups.items():
        # Take at least min_samples_per_subject or all available
        n_samples = min(len(indices), max(min_samples_per_subject, len(indices) // 2))
        selected_indices.extend(indices[:n_samples])
    
    logger.info(f"📊 Validation set: {len(selected_indices)} samples across {len(subject_groups)} subjects")
    return val_dataset.select(selected_indices)

# Load datasets
train_file = f"drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/chunks/train_chunk_{chunk_number}.jsonl"
train_dataset = load_jsonl(train_file)

val_file = f"drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/data/processed/medmcqa/validation.jsonl"
val_dataset = load_jsonl(val_file)

# Create better validation set
val_dataset = create_better_validation_set(val_dataset)

print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

# 🧠 Enhanced tokenization with better prompting
def enhanced_tokenize(example):
    """Enhanced tokenization with improved medical prompting"""
    
    # Extract subject information if available
    subject = example.get('subject', 'General Medicine')
    
    # Enhanced prompt with chain-of-thought reasoning
    prompt = (
        f"You are Baymax, an expert medical AI assistant specialized in {subject}.\n\n"
        f"Medical Question: {example['instruction']}\n\n"
        f"Let me analyze this step by step:\n"
        f"1. This question is about {subject}\n"
        f"2. I need to consider the relevant medical principles\n"
        f"3. Let me think through the options systematically\n\n"
        f"Analysis and Answer: {example['response']}\n\n"
    )
    
    # Add final answer if available
    if 'answer' in example:
        answer = example['answer'].strip().upper()
        prompt += f"The correct option is: {answer}"
    
    prompt += tokenizer.eos_token
    
    # Tokenize with increased max length
    tokenized = tokenizer(
        prompt, 
        truncation=True, 
        padding="max_length", 
        max_length=ENHANCED_CONFIG['max_length']
    )
    
    # Add subject information for tracking
    tokenized['subject'] = subject
    
    return tokenized

# Apply enhanced tokenization
print("🔄 Applying enhanced tokenization...")
train_dataset = train_dataset.map(enhanced_tokenize, desc="Tokenizing training data")
val_dataset = val_dataset.map(enhanced_tokenize, desc="Tokenizing validation data")

# 🔧 Enhanced LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=ENHANCED_CONFIG['lora_r'],
    lora_alpha=ENHANCED_CONFIG['lora_alpha'],
    lora_dropout=ENHANCED_CONFIG['lora_dropout'],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
        "gate_proj", "up_proj", "down_proj"       # MLP modules
    ],
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 🧹 Memory cleanup
torch.cuda.empty_cache()
gc.collect()

# 📊 Enhanced metrics computation with subject tracking
tracker = MedicalTrainingTracker()

def compute_enhanced_metrics(eval_pred):
    """Enhanced metrics with subject-wise tracking"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Mask padding tokens
    mask = labels != -100
    predictions_masked = predictions[mask]
    labels_masked = labels[mask]
    
    # Calculate overall accuracy
    overall_accuracy = (predictions_masked == labels_masked).mean()
    
    # Calculate perplexity
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten for loss calculation
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Calculate cross entropy loss manually for perplexity
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))
    perplexity = torch.exp(loss).item()
    
    return {
        "accuracy": overall_accuracy.item(),
        "perplexity": perplexity,
        "token_accuracy": overall_accuracy.item(),  # Same as accuracy for compatibility
    }

# 🛠️ Enhanced training arguments
training_args = TrainingArguments(
    output_dir=model_dir,
    
    # Batch configuration
    per_device_train_batch_size=ENHANCED_CONFIG['batch_size'],
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=ENHANCED_CONFIG['gradient_accumulation'],
    
    # Training configuration
    num_train_epochs=ENHANCED_CONFIG['num_epochs'],
    learning_rate=ENHANCED_CONFIG['learning_rate'],
    weight_decay=0.01,
    warmup_ratio=ENHANCED_CONFIG['warmup_ratio'],
    
    # Optimization
    fp16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=100,              # More frequent evaluation
    save_strategy="steps",
    save_steps=100,
    logging_steps=50,            # More frequent logging
    
    # Model selection
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    
    # Learning rate scheduling
    lr_scheduler_type="cosine_with_restarts",
    
    # Other settings
    overwrite_output_dir=True,
    report_to=[],  # Disable wandb for now
    
    # Evaluation settings
    eval_accumulation_steps=1,
)

# 🧠 Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 🎓 Enhanced Trainer with callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_enhanced_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    ]
)

# 📈 Training progress callback
class ProgressCallback:
    def __init__(self, tracker, chunk_number):
        self.tracker = tracker
        self.chunk_number = chunk_number
        self.start_time = time.time()
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        # Log progress after each evaluation
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            if 'eval_accuracy' in latest_log:
                self.tracker.log_progress(self.chunk_number, latest_log['eval_accuracy'])

# Add progress tracking
progress_callback = ProgressCallback(tracker, chunk_number)

# 🚀 Start enhanced training
print(f"\n🚀 Starting enhanced training for chunk {chunk_number}")
print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(val_dataset)}")
print(f"🎯 Target: Improve from baseline ~32% accuracy")

try:
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Get final metrics
    final_metrics = trainer.evaluate()
    
    print(f"\n✅ Training complete for chunk {chunk_number}!")
    print(f"📊 Final validation accuracy: {final_metrics.get('eval_accuracy', 0):.2%}")
    print(f"📊 Final perplexity: {final_metrics.get('eval_perplexity', 0):.2f}")
    
    # Log final progress
    tracker.log_progress(chunk_number, final_metrics.get('eval_accuracy', 0))
    
    # Save training statistics
    stats_file = f"{model_dir}/training_stats_chunk_{chunk_number}.json"
    training_stats = {
        'chunk_number': chunk_number,
        'final_accuracy': final_metrics.get('eval_accuracy', 0),
        'final_perplexity': final_metrics.get('eval_perplexity', 0),
        'subject_accuracies': tracker.get_subject_accuracies(),
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'config': ENHANCED_CONFIG
    }
    
    with open(stats_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print(f"📊 Training statistics saved to: {stats_file}")

except Exception as e:
    logger.error(f"❌ Training failed: {e}")
    raise

finally:
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Memory cleanup completed")

print(f"\n{'='*60}")
print(f"🎉 ENHANCED TRAINING PHASE 1 COMPLETED FOR CHUNK {chunk_number}")
print(f"{'='*60}")
print(f"Next steps:")
print(f"1. Continue with next chunk: {chunk_number + 1}")
print(f"2. Monitor subject-wise performance trends")
print(f"3. Consider Phase 2 improvements after completing current batch")
print(f"{'='*60}")