"""
Fairness and bias analysis across different demographic groups.

Analyzes:
1. Score distribution by experience level (Junior/Mid/Senior)
2. Performance by keyword category (Backend/Frontend/Data/QA)
3. Statistical tests for systematic bias

Usage:
    python3 src/evaluation/fairness_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR, OUTPUT_DIR

# Keyword categories
KEYWORD_CATEGORIES = {
    "Backend": [".NET", "Golang", "Java", "Node.js", "PHP", "Python", "Ruby", "Scala"],
    "Frontend": ["JavaScript", "Flutter", "iOS", "Unity"],
    "Data": ["Data Analyst", "Data Engineer", "Data Science", "Business Analyst", "SQL"],
    "QA": ["QA", "QA Automation"],
    "Infrastructure": ["DevOps", "C++"]
}

def categorize_keyword(keyword):
    """Map keyword to category."""
    for category, keywords in KEYWORD_CATEGORIES.items():
        if keyword in keywords:
            return category
    return "Other"

def extract_experience_level(exp_years):
    """Categorize experience into Junior/Mid/Senior."""
    if pd.isna(exp_years) or exp_years is None:
        return "Unknown"
    exp = float(exp_years)
    if exp < 2:
        return "Junior (0-2y)"
    elif exp < 5:
        return "Mid (2-5y)"
    else:
        return "Senior (5+y)"

def plot_score_by_experience(df, output_dir):
    """Box plot: score distribution by experience level."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Score Distribution by Experience Level', fontsize=16, fontweight='bold')

    methods = [
        ('tfidf_score', 'TF-IDF', '#3498db'),
        ('skill_idf_score', 'Skill-IDF', '#e74c3c'),
        ('multi_agent_score', 'Multi-Agent+IDF', '#2ecc71')
    ]

    for idx, (col, name, color) in enumerate(methods):
        ax = axes[idx]

        # Filter out Unknown
        plot_df = df[df['exp_level'] != 'Unknown']

        # Box plot
        order = ['Junior (0-2y)', 'Mid (2-5y)', 'Senior (5+y)']
        sns.boxplot(data=plot_df, x='exp_level', y=col, ax=ax,
                   order=order, palette=[color])

        ax.set_xlabel('Experience Level', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(name)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'fairness_by_experience.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_score_by_category(df, output_dir):
    """Box plot: score distribution by keyword category."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Score Distribution by Keyword Category', fontsize=16, fontweight='bold')

    methods = [
        ('tfidf_score', 'TF-IDF', '#3498db'),
        ('skill_idf_score', 'Skill-IDF', '#e74c3c'),
        ('multi_agent_score', 'Multi-Agent+IDF', '#2ecc71')
    ]

    for idx, (col, name, color) in enumerate(methods):
        ax = axes[idx]

        # Box plot
        order = ['Backend', 'Frontend', 'Data', 'QA', 'Infrastructure']
        order = [o for o in order if o in df['category'].unique()]

        sns.boxplot(data=df, x='category', y=col, ax=ax,
                   order=order, palette='Set2')

        ax.set_xlabel('Keyword Category', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(name)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'fairness_by_category.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def compute_fairness_statistics(df):
    """Statistical tests for bias detection."""
    print("\n" + "=" * 70)
    print("FAIRNESS STATISTICS")
    print("=" * 70)

    methods = ['tfidf_score', 'skill_idf_score', 'multi_agent_score']

    # 1. Experience level analysis
    print("\n[1] Score by Experience Level:")
    print("-" * 70)

    exp_levels = ['Junior (0-2y)', 'Mid (2-5y)', 'Senior (5+y)']
    for method in methods:
        print(f"\n{method}:")

        groups = [df[df['exp_level'] == level][method].dropna() for level in exp_levels]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            # ANOVA test
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"  ANOVA: F={f_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"  ⚠️  Significant difference detected (p < 0.05)")
            else:
                print(f"  ✓ No significant bias (p >= 0.05)")

            # Mean by level
            for level in exp_levels:
                scores = df[df['exp_level'] == level][method]
                if len(scores) > 0:
                    print(f"  {level}: μ={scores.mean():.3f}, σ={scores.std():.3f}, n={len(scores)}")

    # 2. Keyword category analysis
    print("\n[2] Score by Keyword Category:")
    print("-" * 70)

    categories = df['category'].unique()
    for method in methods:
        print(f"\n{method}:")

        groups = [df[df['category'] == cat][method].dropna() for cat in categories]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            # ANOVA test
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"  ANOVA: F={f_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"  ⚠️  Significant difference detected (p < 0.05)")
            else:
                print(f"  ✓ No significant bias (p >= 0.05)")

            # Mean by category
            for cat in sorted(categories):
                scores = df[df['category'] == cat][method]
                if len(scores) > 0:
                    print(f"  {cat}: μ={scores.mean():.3f}, σ={scores.std():.3f}, n={len(scores)}")

def create_fairness_summary_table(df, output_dir):
    """Create summary table with fairness metrics."""

    methods = [
        ('tfidf_score', 'TF-IDF'),
        ('skill_idf_score', 'Skill-IDF'),
        ('multi_agent_score', 'Multi-Agent+IDF')
    ]

    rows = []

    for col, name in methods:
        # Experience level variance
        exp_levels = ['Junior (0-2y)', 'Mid (2-5y)', 'Senior (5+y)']
        exp_means = [df[df['exp_level'] == level][col].mean() for level in exp_levels]
        exp_means = [m for m in exp_means if not pd.isna(m)]
        exp_variance = np.var(exp_means) if exp_means else 0.0

        # Category variance
        categories = df['category'].unique()
        cat_means = [df[df['category'] == cat][col].mean() for cat in categories]
        cat_means = [m for m in cat_means if not pd.isna(m)]
        cat_variance = np.var(cat_means) if cat_means else 0.0

        rows.append({
            'Method': name,
            'Exp_Variance': f"{exp_variance:.4f}",
            'Category_Variance': f"{cat_variance:.4f}",
            'Overall_Mean': f"{df[col].mean():.4f}",
            'Overall_Std': f"{df[col].std():.4f}"
        })

    summary_df = pd.DataFrame(rows)

    # Save
    output_path = output_dir / 'fairness_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    # Markdown
    md_path = output_dir / 'fairness_summary.md'
    with open(md_path, 'w') as f:
        f.write("# Fairness Analysis Summary\n\n")
        f.write("Lower variance indicates less bias across groups.\n\n")
        f.write(summary_df.to_markdown(index=False))
    print(f"  Saved: {md_path}")

    # Print
    print("\n" + "=" * 70)
    print("FAIRNESS SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))

def main():
    print("=" * 70)
    print("FAIRNESS & BIAS ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    detailed_path = OUTPUT_DIR / "results" / "detailed_scores.csv"
    resumes_path = PROCESSED_DIR / "resumes_parsed.parquet"

    if not detailed_path.exists():
        print(f"Error: {detailed_path} not found. Run evaluate_all.py first.")
        sys.exit(1)

    scores_df = pd.read_csv(detailed_path)
    resumes_df = pd.read_parquet(resumes_path)
    print(f"  Loaded {len(scores_df):,} scored pairs")
    print(f"  Loaded {len(resumes_df):,} resumes")

    # Merge resume metadata
    print("\nMerging resume metadata...")
    resumes_df['resume_id'] = resumes_df['id'].astype(str)
    merged = scores_df.merge(
        resumes_df[['resume_id', 'experience_years', 'primary_keyword']],
        on='resume_id',
        how='left'
    )

    # Add derived columns
    merged['exp_level'] = merged['experience_years'].apply(extract_experience_level)
    merged['category'] = merged['primary_keyword'].apply(categorize_keyword)

    print(f"  Experience levels: {merged['exp_level'].value_counts().to_dict()}")
    print(f"  Categories: {merged['category'].value_counts().to_dict()}")

    # Create output directory
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating fairness visualizations...")

    print("\n[1/2] Score by experience level...")
    plot_score_by_experience(merged, viz_dir)

    print("\n[2/2] Score by keyword category...")
    plot_score_by_category(merged, viz_dir)

    # Statistical analysis
    compute_fairness_statistics(merged)

    # Summary table
    create_fairness_summary_table(merged, viz_dir)

    print("\n" + "=" * 70)
    print("FAIRNESS ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
