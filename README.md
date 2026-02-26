# 🤖 Baymax — AI Medical Assistant
> *"I am Baymax, your personal healthcare companion."*

A domain-specific medical question-answering assistant built by fine-tuning **Microsoft Phi-2 (2.7B parameters)** on the **MedMCQA dataset** using **LoRA (Low-Rank Adaptation)** — with a full RAG pipeline and Gradio interface. Fully self-directed project, built after studying LLM internals from scratch.

---

## 🎯 Why I Built This

After studying how LLMs work at a foundational level — tokenization, transformer blocks, attention mechanisms, loss functions, decoding strategies — I wanted to go beyond theory and build something real. I chose the medical domain because it's high-stakes, has a great open dataset (MedMCQA), and honestly, because Baymax from Big Hero 6 is a great vision for what helpful AI should feel like.

The goal was not to build a perfect system — it was to own the full pipeline: data → fine-tuning → retrieval → inference → UI. Every component was built and understood by me.

---

## 🏗️ Architecture

```
MedMCQA Dataset (187k medical QA)
          │
          ▼
  Data Preprocessing
  (preprocess_medmcqa.py)
  Convert to instruction-response format
          │
          ▼
  Corpus Chunking
  (convert_to_chunks.py)
  Split corpus into 500-token segments
          │
          ├─────────────────────┐
          ▼                     ▼
  LoRA Fine-tuning         FAISS Index
  (train.py)               (medmcqa_faiss/)
  Phi-2 + LoRA adapters    Vector embeddings
  on 500 train samples     for retrieval
          │                     │
          └──────────┬──────────┘
                     ▼
            Inference Pipeline
            (inference.py)
            RAG: retrieve → prompt → generate
                     │
                     ▼
            Gradio Interface
            (baymax_gradio.py)
            Single text box → medical answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | Microsoft Phi-2 (2.7B parameters) |
| Fine-tuning Method | LoRA via HuggingFace PEFT |
| Dataset | MedMCQA (187k Indian medical entrance QA) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | Sentence Transformers |
| Training Framework | HuggingFace Transformers + Trainer API |
| UI | Gradio |
| Precision | float16 (FP16) |
| Training Infrastructure | Google Colab T4 → RunPod GPU (48 hrs) |

---

## 🔬 LoRA Configuration

Rather than full fine-tuning (which would require updating all 2.7B parameters), I used **LoRA** — injecting trainable low-rank matrices into the attention layers. This reduced trainable parameters dramatically while preserving the base model's general knowledge.

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,                    # Rank — controls adapter size
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "dense"]
)
```

**Why these target modules?**
- `q_proj`, `k_proj`, `v_proj` — query, key, value projections in multi-head attention
- `dense` — output projection layer
- Targeting all attention projections gives the adapter maximum influence over how the model attends to medical context

**Trainable parameters:** LoRA with r=8 on these modules = ~3.4M trainable params out of 2.7B total (~0.1% of model)

---

## ⚙️ Training Details

```python
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,   # Effective batch size = 4
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=True,                        # float16 for memory efficiency
    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",       # Cosine decay
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

**Training data:** 500 samples from MedMCQA train split
**Validation data:** 500 samples from MedMCQA validation split
**Infrastructure path:**
- Started on **Google Colab T4** (free tier) — hit memory limits quickly
- Moved to **RunPod GPU** for 48 hours of paid compute to complete training

**Memory optimizations applied:**
- `torch.float16` — halves memory vs float32
- `device_map="auto"` — automatic layer distribution across available GPU/CPU
- `gradient_checkpointing_enable()` — trades compute for memory during backprop
- Reduced batch size to 2 to avoid OOM errors

---

## 📊 Results & Honest Assessment

**What worked:**
- Full pipeline runs end to end — data in, medical answer out
- LoRA fine-tuning successfully adapted Phi-2 to medical instruction format
- FAISS retrieval correctly surfaces relevant medical context for queries
- Gradio interface works cleanly for single-turn medical Q&A

**What didn't work perfectly:**
- Accuracy on MedMCQA validation: **not state-of-the-art** — the model sometimes gives wrong answers and occasionally hallucinates plausible-sounding but incorrect medical information
- Root causes:
  - Only 500 training samples (full MedMCQA has 187k) — severely limited by GPU cost
  - 48 hours of RunPod compute was not enough to converge well on medical reasoning
  - Phi-2 at 2.7B is relatively small for complex multi-choice medical reasoning
- **This is expected and honest** — the goal was to own and understand the full pipeline, not to compete with Med-PaLM

**What I learned from the failures:**
- Training data volume matters enormously — 500 samples vs 187k is a ~374x gap
- Medical reasoning requires more than instruction fine-tuning — RLHF or DPO would help
- Eval loss going down ≠ better answers — I built a validation harness (`validation_harness.py`) to measure actual answer quality separately from training loss

---

## 📁 Project Structure

```
Baymax.ai-v1.0.0/
├── data/
│   ├── processed/
│   │   ├── medmcqa/           ← Preprocessed train/val JSONL
│   │   └── medmcqa_faiss/     ← FAISS vector index
│   └── medmcqa_corpus.txt     ← Full corpus for retrieval
├── scripts/
│   ├── train.py               ← LoRA fine-tuning pipeline
│   ├── inference.py           ← RAG inference pipeline
│   ├── baymax_gradio.py       ← Gradio UI
│   ├── convert_to_chunks.py   ← Corpus chunking
│   ├── preprocess_medmcqa.py  ← Dataset preprocessing
│   ├── load_model.py          ← Model loading utility
│   ├── train_incremental.py   ← Incremental training script
│   └── validation_harness.py  ← Validation & accuracy measurement
├── baymax_validation_results.json        ← Validation output
├── baymax_validation_results_no_rag.json ← Ablation: without RAG
├── web_cache.json
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

> ⚠️ **Note:** This project requires a GPU to run inference at reasonable speed. A T4 (Google Colab free tier) is the minimum. The trained LoRA weights are stored on Google Drive.

**1. Clone and install dependencies:**
```bash
git clone https://github.com/Navaneet00/baymax-ai
cd baymax-ai
pip install -r requirements.txt
```

**2. Load model weights:**
- Download LoRA adapter weights from Google Drive (link below)
- Place in `baymax-phi2-chunked/` directory

**3. Run Gradio interface:**
```bash
python scripts/baymax_gradio.py
```

**4. Run inference directly:**
```bash
python scripts/inference.py
```

**5. Run validation harness:**
```bash
python scripts/validation_harness.py
```

---

## 🧠 What I Learned Building This

**Technically:**
- How LoRA works mathematically — injecting `A × B` low-rank matrices into weight updates instead of updating `W` directly
- Why `gradient_checkpointing` trades compute for memory — recomputes activations during backprop instead of storing them
- The difference between training loss and actual output quality — why you need a separate evaluation harness
- How FAISS indexes work — approximate nearest neighbor search using IVF (Inverted File Index)
- Why `device_map="auto"` is powerful — HuggingFace automatically decides which layers go on GPU vs CPU based on available VRAM

**About LLMs in practice:**
- Small training sets don't generalize — 500 samples teaches format, not domain reasoning
- Hallucination is a fundamental challenge, not a bug to fix — the model generates plausible text, not verified facts
- GPU cost is a real engineering constraint — optimizing for compute efficiency is as important as model architecture

---

## 🔮 What I Would Do With More Compute

1. Train on full 187k MedMCQA dataset instead of 500 samples
2. Use QLoRA (4-bit quantization) to fit larger batch sizes in same VRAM
3. Implement DPO (Direct Preference Optimization) to reduce hallucinations
4. Add a confidence score to answers — refuse to answer when uncertain
5. Expand to multi-turn conversation instead of single-turn Q&A

---

## 📚 Dataset

**MedMCQA** — a large-scale multi-subject multi-choice dataset for solving medical entrance exam questions (AIIMS & USMLE style).
- 187,599 questions
- 4 choices per question with explanation
- 21 medical subjects
- Source: [MedMCQA Paper](https://proceedings.mlr.press/v174/pal22a.html)

---

## 👤 Built By

**Navaneet Bammanni** — NITK Surathkal, 2x Co-founder
Self-directed project built after studying GPT-2 architecture, transformer internals, and LLM training from first principles.

[LinkedIn](https://www.linkedin.com/in/navaneetbammanni/) • [GitHub](https://github.com/Navaneet00)

---

*"There, there. It is alright to not be perfect."* — Baymax
