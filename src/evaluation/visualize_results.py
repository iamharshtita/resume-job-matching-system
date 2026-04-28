"""
Generate visualizations from evaluation results.

Creates:
1. Bar chart: NDCG@5 comparison
2. Metrics comparison table
3. Score distribution histograms

Usage:
    python3 src/evaluation/visualize_results.py
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import OUTPUT_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def plot_metric_comparison(df, output_dir):
    """Bar chart comparing all metrics across methods."""

    metrics = ['ndcg@5', 'prec@5', 'rec@5', 'map']
    methods = df['method'].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison Across All Methods', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = df[metric].tolist()

        bars = ax.bar(methods, values, color=['#3498db', '#e74c3c', '#2ecc71'])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel(metric.upper(), fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{metric.upper()} Comparison')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'metric_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_ndcg_comparison(df, output_dir):
    """Focused bar chart for NDCG@5 (primary metric)."""

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = df['method'].tolist()
    ndcg_values = df['ndcg@5'].tolist()

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(methods, ndcg_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('NDCG@5', fontsize=13, fontweight='bold')
    ax.set_title('NDCG@5 Comparison (Primary Metric)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'ndcg_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_score_distributions(detailed_df, output_dir):
    """Histograms of score distributions for each method."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Score Distributions', fontsize=16, fontweight='bold')

    methods = [
        ('tfidf_score', 'TF-IDF', '#3498db'),
        ('skill_idf_score', 'Skill-IDF', '#e74c3c'),
        ('multi_agent_score', 'Multi-Agent+IDF', '#2ecc71')
    ]

    for idx, (col, name, color) in enumerate(methods):
        ax = axes[idx]

        # Separate relevant vs irrelevant
        relevant = detailed_df[detailed_df['relevance'] == 1][col]
        irrelevant = detailed_df[detailed_df['relevance'] == 0][col]

        ax.hist(relevant, bins=20, alpha=0.6, label='Relevant', color='green', edgecolor='black')
        ax.hist(irrelevant, bins=20, alpha=0.6, label='Irrelevant', color='red', edgecolor='black')

        ax.set_xlabel('Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(name)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'score_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_comparison_table(df, output_dir):
    """LaTeX/Markdown table for paper."""

    # Prepare table
    table_data = []
    for _, row in df.iterrows():
        table_data.append({
            'Method': row['method'],
            'NDCG@5': f"{row['ndcg@5']:.4f}",
            'Precision@5': f"{row['prec@5']:.4f}",
            'Recall@5': f"{row['rec@5']:.4f}",
            'MAP': f"{row['map']:.4f}",
            'Time(s)': f"{row['time']:.2f}"
        })

    table_df = pd.DataFrame(table_data)

    # Save as CSV
    output_path = output_dir / 'comparison_table.csv'
    table_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Save as Markdown
    md_path = output_dir / 'comparison_table.md'
    with open(md_path, 'w') as f:
        f.write("# Evaluation Results - Method Comparison\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n\n**Best results in bold** would be added manually.\n")
    print(f"  Saved: {md_path}")

    # Print to console
    print(f"\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(table_df.to_string(index=False))

def main():
    print("=" * 70)
    print("VISUALIZATION GENERATOR")
    print("=" * 70)

    results_dir = OUTPUT_DIR / "results"
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\nLoading evaluation results...")
    comparison_path = results_dir / "comparison_results.csv"
    detailed_path = results_dir / "detailed_scores.csv"

    if not comparison_path.exists():
        print(f"Error: {comparison_path} not found.")
        print("Run src/evaluation/evaluate_all.py first.")
        sys.exit(1)

    comparison_df = pd.read_csv(comparison_path)
    print(f"  Loaded: {comparison_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    print("\n[1/4] Metric comparison chart...")
    plot_metric_comparison(comparison_df, viz_dir)

    print("\n[2/4] NDCG@5 focused chart...")
    plot_ndcg_comparison(comparison_df, viz_dir)

    if detailed_path.exists():
        print("\n[3/4] Score distribution histograms...")
        detailed_df = pd.read_csv(detailed_path)
        plot_score_distributions(detailed_df, viz_dir)
    else:
        print("\n[3/4] Skipped (detailed_scores.csv not found)")

    print("\n[4/4] Comparison table...")
    create_comparison_table(comparison_df, viz_dir)

    print("\n" + "=" * 70)
    print("DONE - All visualizations generated")
    print("=" * 70)
    print(f"\nOutput directory: {viz_dir}")
    print("\nGenerated files:")
    for f in sorted(viz_dir.glob("*")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
