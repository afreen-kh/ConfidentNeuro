from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def compute_ece(confidences, accuracies, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    n = len(confidences)
    
    for i in range(n_bins):
        # Index of items in this bin
        bin_idx = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(accuracies[bin_idx])
            bin_conf = np.mean(confidences[bin_idx])
            ece += (np.sum(bin_idx) / n) * np.abs(bin_acc - bin_conf)
            
    return ece

def run_analysis():
    # 1. Load Data
    input_file = 'evaluation_results.csv'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    
    # Compute nonconformity score
    df['nonconformity_score'] = 1 - df['bertscore_f1']
    
    models = df['model_name'].unique()
    alpha_levels = [0.05, 0.10, 0.20]
    
    all_test_results = []
    coverage_data = []
    risk_dist_data = []
    ece_data = []

    for model in models:
        model_df = df[df['model_name'] == model].copy()
        
        # Split into calibration (80) and test (70)
        # Using shuffle=True with random_state=42
        train_df, test_df = train_test_split(
            model_df, train_size=80, test_size=70, 
            random_state=RANDOM_SEED, shuffle=True
        )
        
        # Conformal Prediction Calibration
        q_hat_01 = None
        for alpha in alpha_levels:
            # q_hat using finite-sample valid formula from Angelopoulos & Bates 2023
            n_cal = len(train_df)
            level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            level = min(level, 1.0)
            q_hat = np.quantile(train_df['nonconformity_score'], level)
            
            # Apply to test set
            test_df[f'conforming_{alpha}'] = test_df['nonconformity_score'] <= q_hat
            
            # Compute coverage
            empirical_coverage = test_df[f'conforming_{alpha}'].mean()
            
            coverage_data.append({
                'model': model,
                'alpha': alpha,
                'empirical_coverage': empirical_coverage,
                'q_hat': q_hat
            })
            
            if alpha == 0.10:
                q_hat_01 = q_hat
                test_df['q_hat'] = q_hat
                test_df['conforming'] = test_df[f'conforming_{alpha}']

        # 3-Tier Risk Scoring (using alpha=0.10)
        def assign_risk(score, qh):
            if score <= qh * 0.5:
                return 'LOW'
            elif score <= qh:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        test_df['risk_tier'] = test_df['nonconformity_score'].apply(lambda x: assign_risk(x, q_hat_01))
        all_test_results.append(test_df)
        
        # Risk Distribution
        risk_counts = test_df['risk_tier'].value_counts()
        total = len(test_df)
        for tier in ['LOW', 'MEDIUM', 'HIGH']:
            count = risk_counts.get(tier, 0)
            risk_dist_data.append({
                'model': model,
                'risk_tier': tier,
                'count': count,
                'percentage': (count / total) * 100
            })

        # Calibration Quality for GPT-4o
        if model == 'GPT-4o':
            # Handle logprobs
            gpt_df = model_df.copy()
            # If logprobs are 0 but should be negative (log space), check distribution
            # Minimally, if all are 0 for non-gpt models, skip them as requested.
            
            # Drop rows with NaN logprobs if any
            gpt_df = gpt_df.dropna(subset=['mean_logprob'])
            
            if not gpt_df.empty:
                # Min-max scale logprobs to 0-1 for "confidence"
                min_lp = gpt_df['mean_logprob'].min()
                max_lp = gpt_df['mean_logprob'].max()
                if max_lp != min_lp:
                    gpt_df['confidence'] = (gpt_df['mean_logprob'] - min_lp) / (max_lp - min_lp)
                else:
                    gpt_df['confidence'] = 1.0 # fallback
                
                accuracies = gpt_df['bertscore_f1'].values
                confidences = gpt_df['confidence'].values
                
                ece = compute_ece(confidences, accuracies)
                ece_data.append({'model': model, 'ECE': ece})
                
                # Plot Reliability Diagram
                plt.figure(figsize=(8, 6))
                bin_boundaries = np.linspace(0, 1, 11)
                bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
                
                bin_accs = []
                for i in range(10):
                    idx = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
                    if np.sum(idx) > 0:
                        bin_accs.append(np.mean(accuracies[idx]))
                    else:
                        bin_accs.append(0)
                        
                plt.bar(bin_centers, bin_accs, width=0.1, alpha=0.7, edgecolor='black', label='Actual BERTScore F1')
                plt.plot([0, 1], [0, 1], '--', color='red', label='Perfect Calibration')
                plt.xlabel('Confidence (Normalized Logprobs)')
                plt.ylabel('Accuracy (BERTScore F1)')
                plt.title(f'Reliability Diagram: GPT-4o (ECE: {ece:.4f})')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig('calibration_plot.png')
                plt.close()

    # 5. Save Output Files
    final_test_df = pd.concat(all_test_results)
    # Reorder columns to match request preference
    cols_to_keep = [
        'question_id', 'model_name', 'nonconformity_score', 'q_hat', 
        'risk_tier', 'conforming', 'question', 'ground_truth_answer', 'generated_answer'
    ]
    final_test_df[cols_to_keep].to_csv('conformal_results.csv', index=False)
    
    pd.DataFrame(coverage_data).to_csv('coverage_table.csv', index=False)
    pd.DataFrame(risk_dist_data).to_csv('risk_distribution.csv', index=False)
    pd.DataFrame(ece_data).to_csv('ece_results.csv', index=False)

    # 6. Print Summary
    print("\n" + "="*30)
    print("CONFORMAL PREDICTION SUMMARY")
    print("="*30)
    
    print("\n--- Coverage Table ---")
    cov_df = pd.DataFrame(coverage_data)
    print(cov_df.to_string(index=False))
    
    print("\n--- Risk Distribution ---")
    risk_df = pd.DataFrame(risk_dist_data)
    for model in models:
        print(f"\nModel: {model}")
        print(risk_df[risk_df['model'] == model][['risk_tier', 'count', 'percentage']].to_string(index=False))
        
    print("\n--- Calibration (ECE) ---")
    for item in ece_data:
        print(f"{item['model']}: ECE = {item['ECE']:.4f}")

if __name__ == "__main__":
    run_analysis()
