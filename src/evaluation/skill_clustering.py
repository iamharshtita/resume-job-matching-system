"""
Skill clustering visualization using t-SNE.

Shows that semantically similar skills cluster together in embedding space,
validating the skill canonicalization and FAISS-based matching approach.

Usage:
    python3 src/evaluation/skill_clustering.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR, OUTPUT_DIR

# Skill categories for coloring
SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "c++", "c#", "ruby", "php", "golang",
        "scala", "kotlin", "swift", "typescript", "rust"
    ],
    "Web Frameworks": [
        "django", "flask", "spring", "react", "angular", "vue.js", "node.js",
        "express.js", "fastapi", "laravel", "rails"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "cassandra", "oracle", "sql server", "dynamodb"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform",
        "ansible", "ci/cd", "git", "github"
    ],
    "Data & ML": [
        "machine learning", "deep learning", "data science", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch", "spark", "hadoop"
    ],
    "Mobile": [
        "android", "ios", "react native", "flutter", "swift", "kotlin"
    ],
    "Other": []
}

def categorize_skill(skill):
    """Map skill to category for coloring."""
    skill_lower = skill.lower().strip()
    for category, skills in SKILL_CATEGORIES.items():
        if skill_lower in skills:
            return category
    return "Other"

def plot_skill_clusters(embeddings, skill_names, output_dir, n_samples=500):
    """Generate t-SNE visualization of skill embeddings."""
    print(f"\n  Computing t-SNE projection for {len(embeddings)} skills...")

    # Sample if too many
    if len(embeddings) > n_samples:
        print(f"  Sampling {n_samples} most frequent skills...")
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        skill_names = [skill_names[i] for i in indices]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # Categorize skills
    categories = [categorize_skill(s) for s in skill_names]
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'skill': skill_names,
        'category': categories
    })

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color by category
    palette = {
        "Programming Languages": "#e74c3c",
        "Web Frameworks": "#3498db",
        "Databases": "#2ecc71",
        "Cloud & DevOps": "#f39c12",
        "Data & ML": "#9b59b6",
        "Mobile": "#1abc9c",
        "Other": "#95a5a6"
    }

    for category, color in palette.items():
        cat_df = df[df['category'] == category]
        if len(cat_df) > 0:
            ax.scatter(cat_df['x'], cat_df['y'], c=color, label=category,
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Annotate key skills
    key_skills = [
        'python', 'java', 'javascript', 'react', 'django', 'aws',
        'docker', 'kubernetes', 'sql', 'mongodb', 'machine learning'
    ]

    for _, row in df.iterrows():
        if row['skill'].lower() in key_skills:
            ax.annotate(row['skill'], (row['x'], row['y']),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=12)
    ax.set_title('Skill Embedding Clusters (t-SNE Projection)', fontweight='bold', fontsize=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'skill_clusters.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_category_heatmap(embeddings, skill_names, output_dir):
    """Heatmap showing average similarity within and between categories."""
    print("\n  Computing category similarity matrix...")

    # Categorize
    categories = [categorize_skill(s) for s in skill_names]
    df = pd.DataFrame({'skill': skill_names, 'category': categories, 'embedding': list(embeddings)})

    # Get unique categories
    unique_cats = [c for c in SKILL_CATEGORIES.keys() if c in df['category'].unique()]

    # Compute average embeddings per category
    cat_embeddings = {}
    for cat in unique_cats:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            avg_emb = np.mean(np.vstack(cat_df['embedding'].values), axis=0)
            cat_embeddings[cat] = avg_emb

    # Similarity matrix
    n = len(cat_embeddings)
    sim_matrix = np.zeros((n, n))
    cat_list = list(cat_embeddings.keys())

    for i, cat1 in enumerate(cat_list):
        for j, cat2 in enumerate(cat_list):
            emb1 = cat_embeddings[cat1]
            emb2 = cat_embeddings[cat2]
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            sim_matrix[i, j] = sim

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=cat_list, yticklabels=cat_list,
               cbar_kws={'label': 'Cosine Similarity'}, ax=ax)

    ax.set_title('Skill Category Similarity Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()

    output_path = output_dir / 'category_similarity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def load_or_generate_taxonomy():
    """Load taxonomy if exists, otherwise generate from skill_idf.json"""
    taxonomy_path = PROCESSED_DIR / "skills_taxonomy.parquet"

    if taxonomy_path.exists():
        print(f"  Loading existing taxonomy: {taxonomy_path}")
        df = pd.read_parquet(taxonomy_path)
        embeddings = np.vstack(df['embedding'].values).astype(np.float32)
        skill_names = df['skill'].tolist()
        return embeddings, skill_names

    # Generate on-the-fly
    print("  Taxonomy not found. Generating from skill_idf.json...")
    idf_path = PROCESSED_DIR / "skill_idf.json"

    if not idf_path.exists():
        print(f"Error: {idf_path} not found. Run compute_idf_weights.py first.")
        sys.exit(1)

    import json
    from sentence_transformers import SentenceTransformer

    with open(idf_path, 'r') as f:
        skill_idf = json.load(f)

    skill_names = list(skill_idf.keys())
    print(f"  Found {len(skill_names):,} skills")

    # Sample if too many
    if len(skill_names) > 2000:
        print(f"  Sampling 2000 most frequent skills...")
        # Sort by IDF (inverse frequency)
        sorted_skills = sorted(skill_idf.items(), key=lambda x: -x[1])
        skill_names = [s[0] for s in sorted_skills[:2000]]

    # Load embedding model
    print("  Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings
    print("  Computing embeddings...")
    embeddings = model.encode(skill_names, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings, skill_names

def main():
    print("=" * 70)
    print("SKILL CLUSTERING VISUALIZATION")
    print("=" * 70)

    # Load skill embeddings
    print("\nLoading skill taxonomy with embeddings...")
    embeddings, skill_names = load_or_generate_taxonomy()
    print(f"  Loaded {len(embeddings):,} skills")
    print(f"  Embedding shape: {embeddings.shape}")

    # Create output directory
    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating clustering visualizations...")

    print("\n[1/2] t-SNE skill clusters...")
    plot_skill_clusters(embeddings, skill_names, viz_dir, n_samples=500)

    print("\n[2/2] Category similarity heatmap...")
    plot_category_heatmap(embeddings, skill_names, viz_dir)

    print("\n" + "=" * 70)
    print("SKILL CLUSTERING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
