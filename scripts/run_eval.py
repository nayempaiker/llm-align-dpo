"""
Evaluates 3 model checkpoints side by side
    - Base Mistral-7B
    - SFT cehckpoint
    - DPO checkpoint
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# prompts
EVAL_PROMPTS = [
    "Explain the difference between supervised learning and unsupervised learning.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the main causes of climate change?",
    "Explain recursion to a 10-year-old.",
    "What is the difference between a list and a tuple in Python?",
    "How does the immune system fight viruses?",
    "Write a SQL query to find the top 5 customers by total purchase amount.",
    "Explain what a neural network is in simple terms.",
    "What are the pros and cons of remote work?",
    "How do I center a div in CSS?",
    "Explain the concept of overfitting in machine learning.",
    "What is the difference between RAM and ROM?",
    "Write a function to reverse a linked list in Python.",
    "What causes inflation and how can it be controlled?",
    "Explain what an API is to someone who has never programmed.",
    "What is the difference between TCP and UDP?",
    "How does gradient descent work?",
    "What are the SOLID principles in software engineering?",
    "Explain the difference between a stack and a queue.",
    "What is transfer learning and why is it useful?",
    "How do vaccines work?",
    "What is the difference between HTTP and HTTPS?",
    "Write a Python function to find all prime numbers up to n.",
    "What is the CAP theorem in distributed systems?",
    "Explain what blockchain technology is.",
    "What is the difference between a process and a thread?",
    "How does a hash table work?",
    "What are the main differences between Python 2 and Python 3?",
    "Explain the concept of time complexity in algorithms.",
    "What is the difference between machine learning and deep learning?",
    "How does DNS work?",
    "Write a function to implement binary search.",
    "What is the difference between Git merge and Git rebase?",
    "Explain what a REST API is.",
    "What is the difference between authentication and authorization?",
    "How does garbage collection work in Python?",
    "What is the difference between a mutex and a semaphore?",
    "Explain what Docker is and why it is useful.",
    "What is the bias-variance tradeoff?",
    "How does backpropagation work in neural networks?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain what Kubernetes is.",
    "What is a closure in Python?",
    "How does public key cryptography work?",
    "What is the difference between horizontal and vertical scaling?",
    "Explain what attention mechanism is in transformers.",
    "What is the difference between precision and recall?",
    "How does a convolutional neural network work?",
    "What is the difference between a compiler and an interpreter?",
    "Explain what a deadlock is and how to prevent it.",
]



# model
def load_model_and_tokenizer(checkpoint_path: str | None, base_model: str):
    """
    Load the model
    """

    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bitcompute_dtype = compute_dtype,
        bnb_4bit_use_double_quant = True,
    )

    # initialize
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )

    # load adapter or base model
    if checkpoint_path:
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            is_trainable=False,
        )
        logger.info(f"Loaded adapter from {checkpoint_path}")
    else:
        logger.info("Loaded base model (no adapter)")
 
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
 
    model.eval()

    return model, tokenizer



# inference
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Generate a response from a model given a prompt
    """
    chat_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
 
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    # generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
 
    # decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()



def generate_all_responses(
    model,
    tokenizer,
    prompts: list[str],
    model_name: str,
) -> list[dict]:
    """
    Run inference on all prompts, return list of {prompt, response}
    """
    
    results = []

    # generate response for all prompts
    for i, prompt in enumerate(prompts):
        logger.info(f"[{model_name}] Generating {i+1}/{len(prompts)}...")
        response = generate_response(model, tokenizer, prompt)
        results.append({"prompt": prompt, "response": response})
    
    return results


# LLM as a Judge scoring
# prompt
JUDGE_SCORE_PROMPT = """You are an expert AI evaluator. Score the following response on a scale of 1-10.
 
Criteria:
- Helpfulness: Does it directly address the question? (1-10)
- Accuracy: Is the information correct? (1-10)
- Clarity: Is it well-explained and easy to understand? (1-10)
- Completeness: Does it cover the key points? (1-10)
 
Question: {prompt}
 
Response: {response}
 
Respond with a JSON object only:
{{"helpfulness": <1-10>, "accuracy": <1-10>, "clarity": <1-10>, "completeness": <1-10>, "overall": <1-10>, "reasoning": "<one sentence>"}}"""
 
 
JUDGE_WINRATE_PROMPT = """You are an expert AI evaluator. Compare these two responses to the same question and decide which is better.
 
Question: {prompt}
 
Response A:
{response_a}
 
Response B:
{response_b}
 
Which response is better overall? Consider helpfulness, accuracy, clarity, and completeness.
 
Respond with a JSON object only:
{{"winner": "A" or "B", "reasoning": "<one sentence explaining why>"}}"""

def score_response(client: OpenAI, prompt: str, response: str) -> dict:
    """
    Ask GPT-4 mini to score a single response
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{
                "role": "user",
                "content": JUDGE_SCORE_PROMPT.format(
                    prompt=prompt,
                    response=response,
                )
            }],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        return json.loads(raw)
    
    except Exception as e:
        logger.warning(f"Scoring failed: {e}")
        return {"overall": 0, "error": str(e)}
 
 
def judge_winrate(client: OpenAI, prompt: str, response_a: str, response_b: str) -> dict:
    """
    Ask GPT-4 mini which response is better — A (SFT) or B (DPO).
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": JUDGE_WINRATE_PROMPT.format(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                )
            }],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Win rate judgment failed: {e}")
        return {"winner": "unknown", "error": str(e)}
    

    # eval pipeline
def run_eval(
    exp: str,
    n_prompts: int = 100,
    base_model: str = "mistralai/Mistral-7B-v0.1",
    dry_run: bool = False,
):
    # paths
    sft_checkpoint = PROJECT_ROOT / "checkpoints" / exp / "sft" / "final"
    dpo_checkpoint = PROJECT_ROOT / "checkpoints" / exp / "dpo" / "final"
    results_dir    = PROJECT_ROOT / "results" / exp
    results_dir.mkdir(parents=True, exist_ok=True)
 
    prompts = EVAL_PROMPTS[:n_prompts]
 
    if dry_run:
        print(f"\n--- DRY RUN ---\n")
        print(f"exp: {exp}")
        print(f"n_prompts: {n_prompts}")
        print(f"base_model: {base_model}")
        print(f"sft_checkpoint: {sft_checkpoint}")
        print(f"dpo_checkpoint: {dpo_checkpoint}")
        print(f"results_dir: {results_dir}")
        print(f"judge_model: gpt-4o-mini")
        print(f"\nPrompts (first 3):")
        for p in prompts[:3]:
            print(f"- {p}")
        print("\nNo evaluation run. Remove --dry-run to execute.\n")
        return
 
    # OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in .env")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
 
    # generate response from all 3 models
    logger.info("=" * 50)
    logger.info("STEP 1: Generating responses")
    logger.info("=" * 50)
 
    logger.info("Loading base model...")
    base_m, base_tok = load_model_and_tokenizer(None, base_model)
    base_responses = generate_all_responses(base_m, base_tok, prompts, "base")
    del base_m, base_tok
    torch.cuda.empty_cache()
 
    logger.info("Loading SFT model...")
    sft_m, sft_tok = load_model_and_tokenizer(str(sft_checkpoint), base_model)
    sft_responses = generate_all_responses(sft_m, sft_tok, prompts, "sft")
    del sft_m, sft_tok
    torch.cuda.empty_cache()
 
    logger.info("Loading DPO model...")
    dpo_m, dpo_tok = load_model_and_tokenizer(str(dpo_checkpoint), base_model)
    dpo_responses = generate_all_responses(dpo_m, dpo_tok, prompts, "dpo")
    del dpo_m, dpo_tok
    torch.cuda.empty_cache()
 
    # save raw response
    raw_path = results_dir / "raw_responses.json"
    with open(raw_path, "w") as f:
        json.dump({
            "base": base_responses,
            "sft":  sft_responses,
            "dpo":  dpo_responses,
        }, f, indent=2)
    logger.info(f"Raw responses saved -> {raw_path}")
 
    # LLM as judge scoring
    logger.info("=" * 50)
    logger.info("STEP 2: LLM-as-judge scoring")
    logger.info("=" * 50)
 
    scores = {"base": [], "sft": [], "dpo": []}
 
    for i, prompt in enumerate(prompts):
        logger.info(f"Scoring prompt {i+1}/{len(prompts)}...")
        scores["base"].append(score_response(client, prompt, base_responses[i]["response"]))
        scores["sft"].append(score_response(client, prompt, sft_responses[i]["response"]))
        scores["dpo"].append(score_response(client, prompt, dpo_responses[i]["response"]))
        time.sleep(0.5)   # rate limit buffer
 
    # compute averages
    def avg_score(score_list: list[dict]) -> float:
        vals = [s.get("overall", 0) for s in score_list if s.get("overall", 0) > 0]
        return round(sum(vals) / len(vals), 2) if vals else 0.0
 
    avg_scores = {
        "base": avg_score(scores["base"]),
        "sft":  avg_score(scores["sft"]),
        "dpo":  avg_score(scores["dpo"]),
    }
 
    logger.info(f"Average scores — Base: {avg_scores['base']} | SFT: {avg_scores['sft']} | DPO: {avg_scores['dpo']}")
 
    scores_path = results_dir / "llm_judge_scores.json"
    with open(scores_path, "w") as f:
        json.dump({"scores": scores, "averages": avg_scores}, f, indent=2)
    logger.info(f"Scores saved -> {scores_path}")
 
    # win rate: SFT vs DPO
    logger.info("=" * 50)
    logger.info("STEP 3: Win rate — SFT vs DPO")
    logger.info("=" * 50)
 
    win_results = []
    dpo_wins = 0
 
    for i, prompt in enumerate(prompts):
        logger.info(f"Win rate judgment {i+1}/{len(prompts)}...")
        result = judge_winrate(
            client,
            prompt,
            sft_responses[i]["response"],   # A = SFT
            dpo_responses[i]["response"],   # B = DPO
        )
        win_results.append({
            "prompt":    prompt,
            "winner":    result.get("winner"),
            "reasoning": result.get("reasoning", ""),
        })
        if result.get("winner") == "B":
            dpo_wins += 1
        time.sleep(0.5)
 
    win_rate = round(dpo_wins / len(prompts) * 100, 1)
    logger.info(f"DPO win rate vs SFT: {win_rate}% ({dpo_wins}/{len(prompts)})")
 
    winrate_path = results_dir / "win_rate.json"
    with open(winrate_path, "w") as f:
        json.dump({
            "dpo_wins":    dpo_wins,
            "total":       len(prompts),
            "dpo_win_rate_pct": win_rate,
            "results":     win_results,
        }, f, indent=2)
    logger.info(f"Win rate saved -> {winrate_path}")
 

    # qualitative examples
    logger.info("=" * 50)
    logger.info("STEP 4: Qualitative examples")
    logger.info("=" * 50)
 
    # Pick 5 interesting examples — first 5 prompts
    examples = []
    for i in range(min(5, len(prompts))):
        examples.append({
            "prompt":   prompts[i],
            "base":     base_responses[i]["response"],
            "sft":      sft_responses[i]["response"],
            "dpo":      dpo_responses[i]["response"],
            "base_score": scores["base"][i].get("overall", 0),
            "sft_score":  scores["sft"][i].get("overall", 0),
            "dpo_score":  scores["dpo"][i].get("overall", 0),
        })
 
    examples_path = results_dir / "qualitative_examples.json"
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Qualitative examples saved -> {examples_path}")
 
    # final summary
    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE")
    print("=" * 55)
    print(f"Prompts evaluated: {len(prompts)}")
    print(f"Base model avg score: {avg_scores['base']} / 10")
    print(f"SFT model avg score: {avg_scores['sft']} / 10")
    print(f"DPO model avg score: {avg_scores['dpo']} / 10")
    print(f"DPO win rate vs SFT: {win_rate}%")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - raw_responses.json")
    print(f"  - llm_judge_scores.json")
    print(f"  - win_rate.json")
    print(f"  - qualitative_examples.json")
    print("=" * 55)
 
    return {
        "avg_scores": avg_scores,
        "dpo_win_rate": win_rate,
    }
 
#  cli
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate base, SFT, and DPO models with GPT-4 as judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp", type=str, required=True, help="Experiment name, e.g. exp_001.")
    parser.add_argument("--n-prompts", type=int, default=50, help="Number of prompts to evaluate.")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running.")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    run_eval(
        exp=args.exp,
        n_prompts=args.n_prompts,
        base_model=args.base_model,
        dry_run=args.dry_run,
    )