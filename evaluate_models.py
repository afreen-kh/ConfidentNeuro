from dotenv import load_dotenv
load_dotenv()
import os
import time
import json
import pandas as pd
import torch
from openai import OpenAI
import anthropic
from huggingface_hub import InferenceClient
from bert_score import score as bert_score_func
import numpy as np
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Configuration
INPUT_FILE = "benchmark_150.csv"
OUTPUT_FILE = "evaluation_results.csv"
BERT_MODEL = "distilbert-base-uncased"
MAX_TOKENS = 300
SMOKE_TEST_LIMIT = None # Set to None for full run

# Initialize Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
hf_client = InferenceClient(token=HF_TOKEN)

def get_gpt4o_response(prompt):
    start_time = time.time()
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            logprobs=True,
            top_logprobs=1
        )
        duration = time.time() - start_time
        answer = response.choices[0].message.content
        
        # Calculate mean logprob
        logprobs = [lp.logprob for lp in response.choices[0].logprobs.content]
        mean_logprob = np.mean(logprobs) if logprobs else 0.0
        
        return answer, duration, mean_logprob
    except Exception as e:
        print(f"Error GPT-4o: {e}")
        return None, 0, 0

def get_claude_response(prompt):
    start_time = time.time()
    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001", 
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        duration = time.time() - start_time
        answer = response.content[0].text
        return answer, duration, 0.0
    except Exception as e:
        print(f"\nError Claude: {e}")
        return None, 0, 0

def get_hf_model_response(prompt):
    start_time = time.time()
    # Primary model: Mixtral-8x7B as reported in paper
    # Qwen2.5-7B included as fallback only — not used in published results
    models_to_try = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-7B-Instruct"   
    ] 
    
    for model_id in models_to_try:
        try:
            response = hf_client.chat_completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS
            )
            duration = time.time() - start_time
            answer = response.choices[0].message.content
            return answer, duration, 0.0
        except Exception as e:
            print(f"\nWarning: Model {model_id} failed: {e}")
            continue
            
    return None, 0, 0

def calculate_bertscore(cands, refs):
    # bert_score returns (P, R, F)
    # We pass lists to take advantage of batching if needed
    P, R, F = bert_score_func(cands, refs, model_type=BERT_MODEL, lang="en", verbose=False)
    return P.tolist()[0], R.tolist()[0], F.tolist()[0]

def main():
    # Load benchmark
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    
    df_benchmark = pd.read_csv(INPUT_FILE)
    
    # Load or initialize results
    if os.path.exists(OUTPUT_FILE):
        df_results = pd.read_csv(OUTPUT_FILE)
        print(f"Resuming from {len(df_results)} existing records.")
    else:
        df_results = pd.DataFrame(columns=[
            "question_id", "category", "difficulty", "question", 
            "ground_truth_answer", "model_name", "generated_answer", 
            "mean_logprob", "bertscore_precision", "bertscore_recall", 
            "bertscore_f1", "response_time_seconds"
        ])

    models = [
        ("GPT-4o", get_gpt4o_response, 1),
        ("Claude Haiku", get_claude_response, 1),
        ("Mixtral-8x7B", get_hf_model_response, 2)
    ]

    total_questions = len(df_benchmark)
    
    for idx, row in df_benchmark.iterrows():
        # Respect smoke test limit
        if SMOKE_TEST_LIMIT is not None and idx >= SMOKE_TEST_LIMIT:
            break

        q_id = row['question_id']
        question = row['question']
        ground_truth = row['ground_truth_answer']
        category = row['category']
        difficulty = row['difficulty']

        for model_name, model_func, delay in models:
            # Check if already evaluated
            if not df_results.empty and ((df_results['question_id'] == q_id) & (df_results['model_name'] == model_name)).any():
                continue

            print(f"Evaluating {model_name} | Question {idx+1}/{total_questions}...", end="\r")
            
            # Call Model
            answer, duration, m_logprob = model_func(question)
            
            if answer:
                # Compute BERTScore
                bp, br, bf = calculate_bertscore([answer], [ground_truth])
                
                # Append result
                new_row = {
                    "question_id": q_id,
                    "category": category,
                    "difficulty": difficulty,
                    "question": question,
                    "ground_truth_answer": ground_truth,
                    "model_name": model_name,
                    "generated_answer": answer,
                    "mean_logprob": m_logprob,
                    "bertscore_precision": bp,
                    "bertscore_recall": br,
                    "bertscore_f1": bf,
                    "response_time_seconds": duration
                }
                
                df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
                
                # Save progress
                df_results.to_csv(OUTPUT_FILE, index=False)
                
                print(f"{model_name} | Question {idx+1}/{total_questions} | BERTScore F1: {bf:.4f}")
            
            # Rate limit delay
            if delay > 0:
                time.sleep(delay)

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
