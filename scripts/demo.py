from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# config
BASE_MODEL      = "mistralai/Mistral-7B-v0.1"
SFT_CHECKPOINT  = str(PROJECT_ROOT / "checkpoints" / "exp_001" / "sft" / "final")
DPO_CHECKPOINT  = str(PROJECT_ROOT / "checkpoints" / "exp_001" / "dpo" / "final")
MAX_NEW_TOKENS  = 768
TEMPERATURE     = 0.7


# load model
def load_model(checkpoint_path: str | None, label: str):
    """
    Loads model for inference
    """
    print(f"Loading {label}...")
    compute_dtype = torch.bfloat16

    # 4bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # initiate base mmodel
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )

    # load lora adapter
    if checkpoint_path:
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=False,
        )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    print(f"{label} loaded.")
    return model, tokenizer


# load all models at startup
print("=" * 50)
print("Loading models...")
print("=" * 50)
base_model, base_tok = load_model(None, "Base model")
sft_model,  sft_tok  = load_model(SFT_CHECKPOINT, "SFT model")
dpo_model,  dpo_tok  = load_model(DPO_CHECKPOINT, "DPO model")
print("All models loaded. Starting demo...")


# inference
def generate(model, tokenizer, prompt: str) -> str:
    """
    Runs inference for the given model
    """
    # format prompt
    chat_prompt = (
        f"<|im_start|>user\n{prompt}\nKeep the answer concise and complete.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # tokenize the input
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(model.device)

    stop_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|im_start|>"])
    stop_ids = [i for i in stop_ids if i is not None]

    # generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id] + stop_ids,
        )

    # converts token id to text
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    for stop_str in ["<|im_start|>", "<|im_end|>"]:
        if stop_str in response:
            response = response[:response.index(stop_str)]

    return response.replace("<p></p>", "\n\n").replace("<p>", "\n\n").replace("</p>", "").strip()

def respond(prompt: str):
    """
    Runs generate function on 3 models
    """
    if not prompt.strip():
        return "Please enter a prompt.", "", ""

    base_response = generate(base_model, base_tok, prompt)   # mistralai/Mistral-7B-v0.1
    sft_response  = generate(sft_model,  sft_tok,  prompt)
    dpo_response  = generate(dpo_model,  dpo_tok,  prompt)

    return base_response, sft_response, dpo_response


# sample prompts
EXAMPLES = [
    "Explain recursion to a 10-year-old.",
    "Write a Python function to check if a number is prime.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain what a REST API is to someone who has never programmed.",
    "What are the pros and cons of using Docker?",
]


# gradio ui
with gr.Blocks(title="LLM Alignment with DPO", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # LLM Alignment with DPO
    **Mistral-7B** fine-tuned end-to-end with a DPO alignment pipeline

    Compare responses from three model checkpoints:
    - **Base** — raw Mistral-7B, no fine-tuning
    - **SFT** — after supervised fine-tuning on UltraFeedback examples
    - **DPO** — after preference alignment on human preference pairs
        """)

    with gr.Row():
        prompt_box = gr.Textbox(
            label="Your prompt",
            placeholder="Ask anything — coding, explanation, reasoning...",
            lines=3,
        )

    with gr.Row():
        submit_btn = gr.Button("Generate", variant="primary", scale=2)
        clear_btn  = gr.Button("Clear", scale=1)

    with gr.Row():
        base_out = gr.Textbox(
            label="🔴 Base Mistral-7B (no fine-tuning)",
            lines=20,
            interactive=False,
        )
        sft_out = gr.Textbox(
            label="🟡 SFT model (supervised fine-tuning)",
            lines=20,
            interactive=False,
        )
        dpo_out = gr.Textbox(
            label="🟢 DPO model (preference aligned)",
            lines=20,
            interactive=False,
        )

    gr.Examples(
        examples=EXAMPLES,
        inputs=prompt_box,
        label="Example prompts — click to load",
    )


    gr.Markdown("""
    ---
    **Model:** `mistralai/Mistral-7B-v0.1` + QLoRA (4-bit NF4) + LoRA r=16
    **Data:** `openbmb/UltraFeedback` — 64k GPT-4 rated preference pairs
        """)

    # Wire up buttons
    submit_btn.click(
        fn=respond,
        inputs=prompt_box,
        outputs=[base_out, sft_out, dpo_out],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[prompt_box, base_out, sft_out, dpo_out],
    )
    prompt_box.submit(
        fn=respond,
        inputs=prompt_box,
        outputs=[base_out, sft_out, dpo_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,   
    )