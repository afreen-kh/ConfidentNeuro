"""
ablation_analysis.py
Ablation and sensitivity analysis for conformal prediction framework.
Ablation 1: Nonconformity score function comparison
Ablation 2: Calibration set size sensitivity
Ablation 3: Alpha sensitivity analysis
"""

from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
df_all = pd.read_csv('evaluation_results.csv')
gpt_df = df_all[df_all['model_name'] == 'GPT-4o'].copy().reset_index(drop=True)

# Normalize logprobs to [0, 1] for GPT-4o
lp_min = gpt_df['mean_logprob'].min()
lp_max = gpt_df['mean_logprob'].max()
if lp_max != lp_min:
    gpt_df['normalized_logprob'] = (gpt_df['mean_logprob'] - lp_min) / (lp_max - lp_min)
else:
    gpt_df['normalized_logprob'] = 1.0

# Nonconformity score functions
gpt_df['score_A'] = 1 - gpt_df['bertscore_f1']
gpt_df['score_B'] = 1 - gpt_df['bertscore_precision']
gpt_df['score_C'] = 1 - gpt_df['normalized_logprob']

print(f"  GPT-4o rows: {len(gpt_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 1: Nonconformity Score Function Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Ablation 1] Comparing nonconformity score functions...")

ALPHA = 0.10
CAL_SIZE = 80
TEST_SIZE = 70

score_functions = {
    'score_A (1 - BERTScore F1)':        'score_A',
    'score_B (1 - BERTScore Precision)':  'score_B',
    'score_C (1 - Normalized Logprob)':   'score_C',
}

ablation1_rows = []
for label, col in score_functions.items():
    cal_df, test_df = train_test_split(
        gpt_df, train_size=CAL_SIZE, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, shuffle=True
    )
    q_hat = np.quantile(cal_df[col].values, 1 - ALPHA)
    conforming = (test_df[col] <= q_hat)
    coverage = conforming.mean()
    mean_nc = gpt_df[col].mean()

    ablation1_rows.append({
        'score_function':    label,
        'q_hat':             round(q_hat, 4),
        'empirical_coverage': round(coverage, 4),
        'mean_nonconformity': round(mean_nc, 4),
    })
    print(f"  {label}: q_hat={q_hat:.4f}, coverage={coverage:.4f}, mean_nc={mean_nc:.4f}")

ablation1_df = pd.DataFrame(ablation1_rows)
ablation1_df.to_csv('ablation_score_functions.csv', index=False)
print("  Saved ablation_score_functions.csv")

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 2: Calibration Set Size Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Ablation 2] Calibration set size sensitivity...")

cal_sizes = [30, 50, 80, 100, 120]
N_RUNS = 10
score_col = 'score_A'
total_n = len(gpt_df)

ablation2_rows = []
for cal_size in cal_sizes:
    test_size = total_n - cal_size
    if test_size <= 0:
        print(f"  Skipping cal_size={cal_size}: not enough data.")
        continue

    coverages = []
    q_hats = []
    for run in range(N_RUNS):
        seed = RANDOM_SEED + run * 7
        cal_df, test_df = train_test_split(
            gpt_df, train_size=cal_size, test_size=test_size,
            random_state=seed, shuffle=True
        )
        q_hat = np.quantile(cal_df[score_col].values, 1 - ALPHA)
        coverage = (test_df[score_col] <= q_hat).mean()
        coverages.append(coverage)
        q_hats.append(q_hat)

    mean_cov = np.mean(coverages)
    std_cov  = np.std(coverages)
    mean_qh  = np.mean(q_hats)

    ablation2_rows.append({
        'cal_size':     cal_size,
        'mean_coverage': round(mean_cov, 4),
        'std_coverage':  round(std_cov, 4),
        'mean_q_hat':    round(mean_qh, 4),
    })
    print(f"  cal_size={cal_size}: mean_cov={mean_cov:.4f} ± {std_cov:.4f}, mean_q_hat={mean_qh:.4f}")

ablation2_df = pd.DataFrame(ablation2_rows)
ablation2_df.to_csv('ablation_calibration_size.csv', index=False)
print("  Saved ablation_calibration_size.csv")

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION 3: Alpha Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Ablation 3] Alpha sensitivity analysis for GPT-4o...")

alpha_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

cal_df, test_df = train_test_split(
    gpt_df, train_size=CAL_SIZE, test_size=TEST_SIZE,
    random_state=RANDOM_SEED, shuffle=True
)

ablation3_rows = []
for alpha in alpha_values:
    q_hat = np.quantile(cal_df[score_col].values, 1 - alpha)
    conforming_mask = test_df[score_col] <= q_hat
    coverage = conforming_mask.mean()
    conf_frac = conforming_mask.mean()  # Binary: same as coverage here

    ablation3_rows.append({
        'alpha':               alpha,
        'nominal_coverage':    round(1 - alpha, 2),
        'q_hat':               round(q_hat, 4),
        'empirical_coverage':  round(coverage, 4),
        'conforming_fraction': round(conf_frac, 4),
    })
    print(f"  alpha={alpha:.2f}: nominal={1-alpha:.2f}, empirical={coverage:.4f}, q_hat={q_hat:.4f}")

ablation3_df = pd.DataFrame(ablation3_rows)
ablation3_df.to_csv('ablation_alpha_sensitivity.csv', index=False)
print("  Saved ablation_alpha_sensitivity.csv")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

COLORS = ['#4A90D9', '#E05C5C', '#5CB85C']
FONT = {'family': 'sans-serif', 'size': 11}
plt.rc('font', **FONT)

# Plot 1: Score Function Comparison (bar chart)
fig, ax = plt.subplots(figsize=(9, 5))
short_labels = ['score_A\n(1 - F1)', 'score_B\n(1 - Precision)', 'score_C\n(1 - Logprob)']
coverages_plot = ablation1_df['empirical_coverage'].tolist()
bars = ax.bar(short_labels, coverages_plot, color=COLORS, edgecolor='black', alpha=0.87, width=0.5)
ax.axhline(y=1 - ALPHA, color='red', linestyle='--', linewidth=1.5, label=f'Nominal Coverage (1-α = {1-ALPHA})')
ax.set_ylabel('Empirical Coverage', fontsize=12)
ax.set_title('Ablation 1: Nonconformity Score Function Comparison\n(GPT-4o, α=0.10, Cal=80)', fontsize=12)
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, coverages_plot):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('ablation_score_functions.png', dpi=150)
plt.close()
print("  Saved ablation_score_functions.png")

# Plot 2: Calibration Size Sensitivity (line + error bars)
fig, ax = plt.subplots(figsize=(9, 5))
x = ablation2_df['cal_size'].tolist()
y = ablation2_df['mean_coverage'].tolist()
yerr = ablation2_df['std_coverage'].tolist()
ax.errorbar(x, y, yerr=yerr, fmt='-o', color='#4A90D9', ecolor='#2166AC',
            capsize=5, capthick=1.5, linewidth=2, markersize=8, label='Mean Coverage ± Std (10 runs)')
ax.axhline(y=1 - ALPHA, color='red', linestyle='--', linewidth=1.5, label=f'Nominal Coverage ({1-ALPHA})')
ax.set_xlabel('Calibration Set Size', fontsize=12)
ax.set_ylabel('Empirical Coverage', fontsize=12)
ax.set_title('Ablation 2: Calibration Set Size Sensitivity\n(GPT-4o, score_A, α=0.10, 10 random seeds)', fontsize=12)
ax.set_xticks(x)
ax.set_ylim(0.7, 1.05)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
# Annotate each point
for xi, yi, ei in zip(x, y, yerr):
    ax.annotate(f'{yi:.3f}', xy=(xi, yi), xytext=(4, 6), textcoords='offset points', fontsize=9)
plt.tight_layout()
plt.savefig('ablation_calibration_size.png', dpi=150)
plt.close()
print("  Saved ablation_calibration_size.png")

# Plot 3: Alpha Sensitivity (empirical vs nominal coverage)
fig, ax = plt.subplots(figsize=(9, 5))
nominal  = ablation3_df['nominal_coverage'].tolist()
empirical = ablation3_df['empirical_coverage'].tolist()
alphas_x = ablation3_df['alpha'].tolist()

ax.plot(nominal, nominal, '--', color='gray', linewidth=1.5, label='Perfect Calibration (diagonal)')
ax.plot(nominal, empirical, '-o', color='#E05C5C', linewidth=2, markersize=8, label='Empirical Coverage (GPT-4o)')
ax.fill_between(nominal, nominal, empirical,
                where=[e > n for e, n in zip(empirical, nominal)],
                alpha=0.15, color='green', label='Over-covered')
ax.fill_between(nominal, nominal, empirical,
                where=[e < n for e, n in zip(empirical, nominal)],
                alpha=0.15, color='red', label='Under-covered')
ax.set_xlabel('Nominal Coverage (1 - α)', fontsize=12)
ax.set_ylabel('Empirical Coverage', fontsize=12)
ax.set_title('Ablation 3: Alpha Sensitivity — Empirical vs Nominal Coverage\n(GPT-4o, score_A, Cal=80)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
# Annotate each point with alpha value
for n, e, a in zip(nominal, empirical, alphas_x):
    ax.annotate(f'α={a}', xy=(n, e), xytext=(5, -12), textcoords='offset points', fontsize=8.5, color='#333')
plt.tight_layout()
plt.savefig('ablation_alpha_sensitivity.png', dpi=150)
plt.close()
print("  Saved ablation_alpha_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print Full Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ABLATION ANALYSIS SUMMARY")
print("="*60)

print("\n[Ablation 1] Nonconformity Score Function Comparison:")
print(ablation1_df.to_string(index=False))

print("\n[Ablation 2] Calibration Set Size Sensitivity:")
print(ablation2_df.to_string(index=False))

print("\n[Ablation 3] Alpha Sensitivity:")
print(ablation3_df[['alpha','nominal_coverage','empirical_coverage','q_hat','conforming_fraction']].to_string(index=False))

print("\n=== OUTPUT FILES ===")
print("  ablation_score_functions.csv")
print("  ablation_calibration_size.csv")
print("  ablation_alpha_sensitivity.csv")
print("  ablation_score_functions.png")
print("  ablation_calibration_size.png")
print("  ablation_alpha_sensitivity.png")
