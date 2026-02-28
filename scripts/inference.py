from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model directory
model_dir = "drive/MyDrive/ColabNotebooks/Baymax.ai-v1.0.0/baymax-phi2-chunked"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_answer(instruction_text):
    # Format prompt like your training data
    prompt = f"Properly and accurately answer for the given question with proper explanation. Instruction:\n{instruction_text}\n\nResponse:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
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

    # Decode and extract only the relevant response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Response:\n" in generated_text:
        return generated_text.split("Response:\n", 1)[1].strip()
    return generated_text.strip()

# List of MCQ-style questions
test_questions = [
    "Which of the following is derived from fibroblast cells?\nA. TGF-13\nB. MMP2\nC. Collagen\nD. Angiopoietin",
    "In alleged history of gun shot injury, there is burning, blackening, tattooing around the wound collar. The injury is:\nA. Close shot entry wound\nB. Close shot exit wound\nC. Distant shot entry wound\nD. Distant shot exit wound",
    "Which macrolide is active against Mycobacterium leprae?\nA. Azithromycin\nB. Roxithromycin\nC. Clarithromycin\nD. Framycetin",
    "Xanthenuric acid is produced in metabolism of?\nA. Tyrosine\nB. Glycine\nC. Methionine\nD. Tryptophan",
    "Most common site of direct hernia?\nA. Hesselbach's triangle\nB. Femoral canal\nC. No site predilection\nD. None",
    "A 10-year-old male presents a smooth swelling near superficial inguinal ring, which moves downward when the testicle is pulled. Diagnosis:\nA. Inguinal hernia\nB. Congenital hydrocele\nC. Encysted hydrocele of the cord\nD. Varicocele",
    "All are derived from ectoderm except?\nA. Lens\nB. Eustachian tube\nC. Brain\nD. Retina",
    "Abnormal vascular patterns seen with colposcopy in case of cervical intraepithelial neoplasia  are all except\nA. Punctation\nB. Mosaicism\nC. Satellite lesions\nD. Atypical vessels",
    "True of umbilical hernia:\nA. Most common content is large intestine\nB. Most of the umbilical hernias disappear spontaneously\nC. Males are affected more than females\nD. Uncomplicated hernias are repaired at 1 year of age through an infraumbilical incision.",
    "False about MgSO₄ is?\nA. Not used as antihypertensive\nB. Its dose is different for eclampsia and preeclampsia\nC. Deep tendon reflexes is monitored for toxicity\nD. It acts as a membrane stabilizer and neuroprotector",
    "Prognosis of treatment in case of Class II malocclusion is favorable when?\nA. Class II malocclusion without sliding of mandible\nB. Class II malocclusion with anterior sliding of mandible\nC. Class II malocclusion with posterior sliding of mandible\nD. Prognosis does not depend on sliding of mandible",
    "Maximum increase in prolactin level is caused by?\nA. Risperidone\nB. Clozapine\nC. Olanzapine\nD. Aripiprazole",
    "What is the age of routine screening mammography?\nA. 20 years\nB. 30 years\nC. 40 years\nD. 50 years",
    "A 14-year-old girl is brought to the office by her mother because of a 3-month history of red bumps on her skin. The patient says the bumps are not itchy or painful but that she finds them embarrassing. She has no history of major medical illness and takes no medications. Her vital signs are within normal limits. Physical examination shows the findings in the photograph. Which of the following is the most likely diagnosis?\nA. Eczema\nB. Folliculitis\nC. Hidradenitis\nD. Keratosis pilaris\nE. Urticaria"
]

print("=== BAYMAX Medical AI Responses ===\n")

for i, q in enumerate(test_questions, 1):
    print(f"Question {i}:")
    print(q)
    try:
        answer = generate_answer(q)
        print(answer)
    except Exception as e:
        print(f"Error generating answer: {e}")
    print("-" * 50)