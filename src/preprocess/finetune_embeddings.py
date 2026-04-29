"""
fine-tunes all-MiniLM-L6-v2 on domain-specific resume/JD data
so the embedding model understands tech hiring context better

training pairs come from three sources:
1. skill aliases  - "py" and "python" should be near-identical
2. resume + JD texts from same keyword - should be similar
3. cross-keyword pairs - should be dissimilar (negatives)

saves fine-tuned model to data/models/skill_embedding_model/
once done, skill_miner will use it automatically if present

usage:
    python3 src/preprocess/finetune_embeddings.py
    python3 src/preprocess/finetune_embeddings.py --epochs 3 --batch-size 32 --samples 5000
"""
import sys
import random
import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PROCESSED_DIR

MODEL_SAVE_PATH = ROOT / "data" / "models" / "skill_embedding_model"

# same aliases as skill_miner
ALIASES = {
    'js': 'javascript', 'ts': 'typescript', 'py': 'python', 'rb': 'ruby',
    'reactjs': 'react', 'react.js': 'react', 'nextjs': 'next.js',
    'nodejs': 'node.js', 'vuejs': 'vue.js', 'angularjs': 'angular',
    'k8s': 'kubernetes', 'kube': 'kubernetes', 'tf': 'tensorflow',
    'pg': 'postgresql', 'postgres': 'postgresql', 'mongo': 'mongodb',
    'ml': 'machine learning', 'dl': 'deep learning',
    'nlp': 'natural language processing', 'cv': 'computer vision',
    'oop': 'object-oriented programming', 'cicd': 'ci/cd',
    'ror': 'ruby on rails', 'net': '.net', 'dotnet': '.net',
    'es6': 'javascript', 'mssql': 'sql server', 'pgsql': 'postgresql',
    'expressjs': 'express.js', 'nestjs': 'nest.js', 'nuxtjs': 'nuxt.js',
    'springboot': 'spring boot', 'scss': 'sass',
}


def split_and_save(resumes, jds, test_ratio=0.2, seed=42):
    """
    splits resumes and JDs into train/test by id.
    saves test ids to data/test/test_split.json so evaluate_all.py
    can use the exact same held-out records every time.
    """
    import json
    random.seed(seed)

    r_test_ids, r_train_ids = {}, {}
    j_test_ids, j_train_ids = {}, {}

    for kw in resumes['primary_keyword'].unique():
        ids = resumes[resumes['primary_keyword'] == kw]['id'].astype(str).tolist()
        random.shuffle(ids)
        n_test = max(1, int(len(ids) * test_ratio))
        r_test_ids[kw]  = ids[:n_test]
        r_train_ids[kw] = ids[n_test:]

    for kw in jds['primary_keyword'].unique():
        ids = jds[jds['primary_keyword'] == kw]['id'].astype(str).tolist()
        random.shuffle(ids)
        n_test = max(1, int(len(ids) * test_ratio))
        j_test_ids[kw]  = ids[:n_test]
        j_train_ids[kw] = ids[n_test:]

    split = {
        "resume_test_ids":  r_test_ids,
        "resume_train_ids": r_train_ids,
        "jd_test_ids":      j_test_ids,
        "jd_train_ids":     j_train_ids,
    }

    out = ROOT / "data" / "test" / "test_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(split, f)
    print(f"train/test split saved to {out}")

    # return train-only subsets
    all_r_train = [rid for ids in r_train_ids.values() for rid in ids]
    all_j_train = [jid for ids in j_train_ids.values() for jid in ids]
    r_train = resumes[resumes['id'].astype(str).isin(all_r_train)]
    j_train = jds[jds['id'].astype(str).isin(all_j_train)]

    total_r_test = sum(len(v) for v in r_test_ids.values())
    total_j_test = sum(len(v) for v in j_test_ids.values())
    print(f"train: {len(r_train):,} resumes, {len(j_train):,} jds")
    print(f"test (held out): {total_r_test:,} resumes, {total_j_test:,} jds")

    return r_train, j_train


def skill_overlap_label(resume_skills, jd_skills) -> float:
    # graded label based on actual skill overlap between resume and JD
    # uses raw_skills from resume and required_skills from JD
    r = {str(s).lower().strip() for s in resume_skills if s}
    j = {str(s).lower().strip() for s in jd_skills if s}
    if not j:
        return 0.7  # no JD skills info, assume moderate match for same keyword
    overlap = len(r & j) / len(j)
    # scale: 0.3 base (same domain) + up to 0.7 for perfect skill match
    return round(min(1.0, 0.3 + 0.7 * overlap), 3)


def build_training_pairs(resumes, jds, samples_per_keyword, seed=42):
    random.seed(seed)
    pairs = []

    keywords = resumes['primary_keyword'].unique().tolist()

    # alias pairs - skill strings that mean the same thing → label 1.0
    for alias, canonical in ALIASES.items():
        pairs.append(InputExample(texts=[alias, canonical], label=1.0))
        pairs.append(InputExample(texts=[alias.upper(), canonical], label=1.0))

    for kw in keywords:
        kw_resumes = resumes[resumes['primary_keyword'] == kw].dropna(subset=['raw_text'])
        kw_jds     = jds[jds['primary_keyword'] == kw].dropna(subset=['raw_text'])

        if kw_resumes.empty or kw_jds.empty:
            continue

        n = min(samples_per_keyword, len(kw_resumes), len(kw_jds))
        r_sample = kw_resumes.sample(n, random_state=seed)
        j_sample = kw_jds.sample(n, random_state=seed)

        # same-keyword pairs with graded labels based on skill overlap
        for (_, r_row), (_, j_row) in zip(r_sample.iterrows(), j_sample.iterrows()):
            r_skills = r_row.get('raw_skills', [])
            j_skills = j_row.get('required_skills', [])
            r_skills = r_skills.tolist() if hasattr(r_skills, 'tolist') else list(r_skills or [])
            j_skills = j_skills.tolist() if hasattr(j_skills, 'tolist') else list(j_skills or [])
            label = skill_overlap_label(r_skills, j_skills)
            pairs.append(InputExample(
                texts=[str(r_row['raw_text'])[:256], str(j_row['raw_text'])[:256]],
                label=label
            ))

        # cross-keyword negatives → label 0.0
        other_kws = [k for k in keywords if k != kw]
        if not other_kws:
            continue
        neg_kw  = random.choice(other_kws)
        neg_jds = jds[jds['primary_keyword'] == neg_kw].dropna(subset=['raw_text'])
        if neg_jds.empty:
            continue

        n_neg     = min(n // 2, len(neg_jds), len(kw_resumes))
        r_neg     = kw_resumes.sample(n_neg, random_state=seed)
        j_neg     = neg_jds.sample(n_neg, random_state=seed)
        for (_, r_row), (_, j_row) in zip(r_neg.iterrows(), j_neg.iterrows()):
            pairs.append(InputExample(
                texts=[str(r_row['raw_text'])[:256], str(j_row['raw_text'])[:256]],
                label=0.0
            ))

    pos = sum(1 for p in pairs if p.label > 0.5)
    neg = sum(1 for p in pairs if p.label <= 0.5)
    print(f"training pairs: {len(pairs):,} total ({pos:,} positive/graded, {neg:,} negative)")
    return pairs


def evaluate_model(model, resumes, jds):
    """Evaluate on held-out test split only to avoid leaking training data."""
    import json
    split_path = ROOT / "data" / "test" / "test_split.json"
    if split_path.exists():
        split = json.load(open(split_path))
        test_r_ids = set(rid for ids in split["resume_test_ids"].values() for rid in ids)
        test_j_ids = set(jid for ids in split["jd_test_ids"].values()     for jid in ids)
        resumes = resumes[resumes['id'].astype(str).isin(test_r_ids)]
        jds     = jds[jds['id'].astype(str).isin(test_j_ids)]

    random.seed(99)
    sentences1, sentences2, scores = [], [], []

    keywords = resumes['primary_keyword'].unique().tolist()
    for kw in random.sample(keywords, min(5, len(keywords))):
        r = resumes[resumes['primary_keyword'] == kw]['raw_text'].dropna().tolist()
        j = jds[jds['primary_keyword'] == kw]['raw_text'].dropna().tolist()
        if not r or not j:
            continue
        for _ in range(min(10, len(r), len(j))):
            sentences1.append(random.choice(r)[:256])
            sentences2.append(random.choice(j)[:256])
            scores.append(1.0)

        other_kws = [k for k in keywords if k != kw]
        neg_kw = random.choice(other_kws) if other_kws else None
        if neg_kw:
            neg_j = jds[jds['primary_keyword'] == neg_kw]['raw_text'].dropna().tolist()
            for _ in range(min(10, len(r), len(neg_j))):
                sentences1.append(random.choice(r)[:256])
                sentences2.append(random.choice(neg_j)[:256])
                scores.append(0.0)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    result = evaluator(model)
    if isinstance(result, dict):
        return list(result.values())[0]
    return float(result)


def main():
    parser = argparse.ArgumentParser(description="fine-tune skill embeddings")
    parser.add_argument("--epochs",       type=int,   default=2)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--samples",      type=int,   default=2000,
                        help="resume+jd pairs to sample per keyword")
    parser.add_argument("--warmup-steps", type=int,   default=100)
    args = parser.parse_args()

    print("loading parsed data...")
    resumes = pd.read_parquet(PROCESSED_DIR / "resumes_parsed.parquet")
    jds     = pd.read_parquet(PROCESSED_DIR / "jds_parsed.parquet")
    print(f"resumes: {len(resumes):,} | jds: {len(jds):,}")

    # load existing train/test split - created by create_test_split.py
    import json
    split_path = ROOT / "data" / "test" / "test_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"test split not found at {split_path}. "
            "Run python3 src/preprocess/create_test_split.py first."
        )
    split = json.load(open(split_path))
    train_r_ids = set(rid for ids in split["resume_train_ids"].values() for rid in ids)
    train_j_ids = set(jid for ids in split["jd_train_ids"].values() for jid in ids)
    resumes_train = resumes[resumes['id'].astype(str).isin(train_r_ids)]
    jds_train     = jds[jds['id'].astype(str).isin(train_j_ids)]
    print(f"using train split: {len(resumes_train):,} resumes, {len(jds_train):,} jds")

    print(f"\nbuilding training pairs (samples per keyword: {args.samples})...")
    train_pairs = build_training_pairs(resumes_train, jds_train, samples_per_keyword=args.samples)

    BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nloading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

    print("evaluating base model before fine-tuning...")
    base_score = evaluate_model(model, resumes, jds)
    print(f"base model score: {base_score:.4f}")

    # use cosine similarity loss - works well with positive/negative pairs
    train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    print(f"\nfine-tuning for {args.epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        show_progress_bar=True,
    )

    print("\nevaluating fine-tuned model...")
    finetuned_score = evaluate_model(model, resumes, jds)
    print(f"fine-tuned model score: {finetuned_score:.4f}")
    print(f"improvement: {finetuned_score - base_score:+.4f}")

    # save model
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_SAVE_PATH))
    print(f"\nmodel saved to {MODEL_SAVE_PATH}")
    print("skill_miner will use this model automatically on next run")

    # save summary
    out = ROOT / "outputs" / "results" / "finetuning_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "EMBEDDING MODEL FINE-TUNING SUMMARY",
        "=" * 40,
        f"base model: {BASE_MODEL_NAME}",
        f"fine-tuned model: {MODEL_SAVE_PATH}",
        f"evaluated on: held-out test split only",
        f"epochs: {args.epochs}",
        f"batch size: {args.batch_size}",
        f"training pairs: {len(train_pairs):,}",
        f"base model score: {base_score:.4f}",
        f"fine-tuned score: {finetuned_score:.4f}",
        f"improvement: {finetuned_score - base_score:+.4f}",
        f"saved to: {MODEL_SAVE_PATH}",
    ]
    out.write_text("\n".join(lines))
    print(f"summary saved to {out}")


if __name__ == "__main__":
    main()
