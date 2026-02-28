import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# 🚀 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 📦 Load model and tokenizer
model_dir = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/baymax-phi2-chunked"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 🧠 Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 🤖 Response generation function
def generate_answer(instruction_text):
    if not instruction_text.strip():
        return "⚠️ Please enter a valid medical question."

    # Prompt template
    prompt = (
        "You are Baymax, a friendly and knowledgeable medical assistant. "
        "Answer with accurate, clear, and concise medical advice.\n\n"
        f"Instruction:\n{instruction_text.strip()}\n\nResponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    try:
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract response after 'Response:'
        return generated_text.split("Response:\n", 1)[-1].strip()
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# 🧠 Chatbot handler
def chat_with_baymax(message, history):
    response = generate_answer(message)
    history.append((message, response))
    return "", history

# 🎨 Gradio UI
with gr.Blocks(title="Baymax Medical Chatbot") as demo:
    gr.HTML("""
    <div style='text-align: center; font-family:Arial;'>
        <h1>🩺 Baymax AI - Medical Assistant</h1>
        <p>Ask your medical questions with confidence and receive smart, compassionate responses.</p>
    </div>
    """)

    chatbot = gr.Chatbot(label="Baymax Medical Chat")
    msg = gr.Textbox(label="Enter your medical query", placeholder="E.g., I have mild fever, what should I do?", lines=2)
    clear = gr.Button("🧹 Clear Chat")
    state = gr.State([])

    msg.submit(chat_with_baymax, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.launch(share=True)