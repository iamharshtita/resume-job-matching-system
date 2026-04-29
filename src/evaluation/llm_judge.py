"""
LLM-as-Judge evaluation using AWS Bedrock Nova Lite.

Sends sampled JD-resume pairs to Nova Lite for independant relevance ratings (0-2),
then computes Spearman correlation between LLM rankings and each methods rankings.

This provides independant validation that the systems rankings align with
an external judgment, beyond the keyword-based proxy labels.

Rating scale:
    0 = Poor match (different domain, missing most required skills)
    1 = Partial match (some skill overlap, but significant gaps)
    2 = Strong match (same domain, most required skills present)

Usage:
    python3 src/evaluation/llm_judge.py
    python3 src/evaluation/llm_judge.py --n-pairs 300 --profile hackathon
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import OUTPUT_DIR, RESULTS_DIR

EVAL_PAIRS_PATH = ROOT / "data" / "test" / "eval_pairs.parquet"
DETAILED_SCORES_PATH = RESULTS_DIR / "detailed_scores.csv"

# bedrock config
MODEL_ID = "amazon.nova-lite-v1:0"
REGION = "us-east-1"

JUDGE_PROMPT = """You are an expert technical recruiter evaluating how well a candidate's resume matches a job description.

Rate the match on this scale:
0 = Poor match - different technical domain, missing most required skills
1 = Partial match - some skill overlap but significant gaps in key requirements
2 = Strong match - same technical domain, most required skills present

Consider: skill alignment, experience relevance, and domain fit.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Respond with ONLY a JSON object in this exact format:
{{"score": <0 or 1 or 2>, "reason": "<one sentence explanation>"}}"""


def call_bedrock(client, jd_text, resume_text, max_retries=3):
    """Call Bedrock Nova Lite and extract the relevence score."""
    prompt = JUDGE_PROMPT.format(
        jd_text=jd_text[:3000],    # truncate to stay within token limits
        resume_text=resume_text[:3000],
    )

    for attempt in range(max_retries):
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {
                        "maxTokens": 100,
                        "temperature": 0.0,
                    },
                }),
            )
            body = json.loads(response["body"].read())
            text = body["output"]["message"]["content"][0]["text"].strip()

            # parse JSON response, handle markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            score = int(result["score"])
            reason = result.get("reason", "")

            if score not in (0, 1, 2):
                score = max(0, min(2, score))

            return score, reason

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            # last resort: try to pull a number from the raw response
            try:
                for ch in text:
                    if ch in "012":
                        return int(ch), "parse fallback"
            except:
                pass
            return -1, f"parse error: {str(e)}"

        except Exception as e:
            if "ThrottlingException" in str(type(e).__name__) or "throttl" in str(e).lower():
                wait = 2 ** (attempt + 1)
                print(f"    throttled, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return -1, f"api error: {str(e)}"

    return -1, "max retries exceeded"


def sample_pairs(n_pairs, seed=42):
    """Sample pairs stratified by keyword so each domain is represented."""
    ep = pd.read_parquet(EVAL_PAIRS_PATH)
    ds = pd.read_csv(DETAILED_SCORES_PATH)

    # merge scores with eval pair texts
    merged = ep.merge(
        ds[["resume_id", "jd_id", "tfidf_score", "skill_idf_score", "multi_agent_score"]],
        on=["resume_id", "jd_id"],
        how="inner",
    )

    # stratified sample, equal from each keyword
    keywords = sorted(merged["keyword"].unique())
    per_keyword = max(n_pairs // len(keywords), 2)

    sampled = []
    for kw in keywords:
        kw_df = merged[merged["keyword"] == kw]
        n = min(per_keyword, len(kw_df))
        sampled.append(kw_df.sample(n, random_state=seed))

    result = pd.concat(sampled).reset_index(drop=True)

    # grab more if we havent hit n_pairs yet
    if len(result) < n_pairs:
        remaining = merged[~merged.index.isin(result.index)]
        extra = remaining.sample(min(n_pairs - len(result), len(remaining)), random_state=seed)
        result = pd.concat([result, extra]).reset_index(drop=True)

    return result.head(n_pairs)


def compute_spearman_per_jd(df, score_col, llm_col):
    """Compute Spearman correlation between method scores and LLM scores, per JD."""
    correlations = []
    for _, group in df.groupby("jd_id"):
        if len(group) < 3:
            continue
        method_scores = group[score_col].values
        llm_scores = group[llm_col].values

        # skip if theres no variance in either
        if np.std(method_scores) == 0 or np.std(llm_scores) == 0:
            continue

        rho, pval = stats.spearmanr(method_scores, llm_scores)
        if not np.isnan(rho):
            correlations.append(rho)

    return correlations


def compute_global_spearman(df, score_col, llm_col):
    """Compute global Spearman correlation accross all pairs."""
    method_scores = df[score_col].values
    llm_scores = df[llm_col].values

    if np.std(method_scores) == 0 or np.std(llm_scores) == 0:
        return float("nan"), float("nan")

    rho, pval = stats.spearmanr(method_scores, llm_scores)
    return rho, pval


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="number of pairs to evaluate (default 200)")
    parser.add_argument("--profile", type=str, default="hackathon",
                        help="AWS profile name (default: hackathon)")
    parser.add_argument("--region", type=str, default=REGION,
                        help="AWS region (default: us-east-1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION (AWS Bedrock Nova Lite)")
    print("=" * 70)

    if not EVAL_PAIRS_PATH.exists():
        print(f"Error: {EVAL_PAIRS_PATH} not found. Run build_eval_dataset.py first.")
        sys.exit(1)
    if not DETAILED_SCORES_PATH.exists():
        print(f"Error: {DETAILED_SCORES_PATH} not found. Run evaluate_all.py first.")
        sys.exit(1)

    # setup bedrock client
    import boto3
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("bedrock-runtime")
    print(f"  AWS profile: {args.profile}")
    print(f"  Region: {args.region}")
    print(f"  Model: {MODEL_ID}")

    # sample pairs
    print(f"\nSampling {args.n_pairs} pairs...")
    df = sample_pairs(args.n_pairs, seed=args.seed)
    print(f"  Sampled {len(df)} pairs across {df['keyword'].nunique()} keywords")
    print(f"  Difficulty distribution: {df['difficulty'].value_counts().to_dict()}")

    # run LLM evaluation
    print(f"\nRunning LLM evaluation on {len(df)} pairs...")
    print("  (this takes ~1-3 minutes depending on rate limits)")

    llm_scores = []
    llm_reasons = []
    errors = 0

    for i, (_, row) in enumerate(df.iterrows()):
        score, reason = call_bedrock(client, row["jd_text"], row["resume_text"])

        if score == -1:
            errors += 1
            score = 0  # default to 0 on error
            reason = f"ERROR: {reason}"

        llm_scores.append(score)
        llm_reasons.append(reason)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(df)} done ({errors} errors)")

        # small delay to avoid throttling
        time.sleep(0.3)

    df["llm_score"] = llm_scores
    df["llm_reason"] = llm_reasons

    print(f"\n  Completed: {len(df)} pairs, {errors} errors")
    print(f"  LLM score distribution: {df['llm_score'].value_counts().sort_index().to_dict()}")

    # compute correlations
    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION: Method Rankings vs LLM Judgments")
    print("=" * 70)

    methods = [
        ("tfidf_score", "TF-IDF"),
        ("skill_idf_score", "Skill-IDF"),
        ("multi_agent_score", "Multi-Agent+IDF"),
    ]

    correlation_rows = []

    print(f"\n{'Method':<20} {'Global rho':>10} {'p-value':>10} {'Mean per-JD rho':>15} {'n_JDs':>8}")
    print("-" * 65)

    for col, name in methods:
        global_rho, global_p = compute_global_spearman(df, col, "llm_score")
        per_jd_rhos = compute_spearman_per_jd(df, col, "llm_score")
        mean_rho = np.mean(per_jd_rhos) if per_jd_rhos else float("nan")

        sig = "***" if global_p < 0.001 else "**" if global_p < 0.01 else "*" if global_p < 0.05 else "ns"

        print(f"{name:<20} {global_rho:>10.4f} {global_p:>10.4f}{sig:>3} {mean_rho:>13.4f} {len(per_jd_rhos):>8}")

        correlation_rows.append({
            "method": name,
            "global_spearman": round(global_rho, 4),
            "global_p_value": round(global_p, 6),
            "significant": global_p < 0.05,
            "mean_per_jd_spearman": round(mean_rho, 4) if not np.isnan(mean_rho) else None,
            "n_jds_with_variance": len(per_jd_rhos),
        })

    # LLM agreement with proxy labels
    print("\n" + "=" * 70)
    print("LLM AGREEMENT WITH PROXY LABELS")
    print("=" * 70)

    relevant = df[df["relevance"] == 1]["llm_score"]
    irrelevant = df[df["relevance"] == 0]["llm_score"]

    print(f"\n  Relevant pairs (proxy=1):   LLM mean={relevant.mean():.2f}  distribution={relevant.value_counts().sort_index().to_dict()}")
    print(f"  Irrelevant pairs (proxy=0): LLM mean={irrelevant.mean():.2f}  distribution={irrelevant.value_counts().sort_index().to_dict()}")

    if len(relevant) > 0 and len(irrelevant) > 0:
        u_stat, u_p = stats.mannwhitneyu(relevant, irrelevant, alternative="greater")
        print(f"\n  Mann-Whitney U test (relevant > irrelevant): U={u_stat:.0f}, p={u_p:.6f}")
        if u_p < 0.05:
            print("  LLM scores are significantly higher for proxy-relevant pairs")
        else:
            print("  LLM scores do NOT significantly differ - proxy labels may be weak")

    # save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    viz_dir = OUTPUT_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    llm_output_path = RESULTS_DIR / "llm_judge_scores.csv"
    df[["keyword", "difficulty", "resume_id", "jd_id", "relevance",
        "tfidf_score", "skill_idf_score", "multi_agent_score",
        "llm_score", "llm_reason"]].to_csv(llm_output_path, index=False)
    print(f"  LLM scores: {llm_output_path}")

    corr_df = pd.DataFrame(correlation_rows)
    corr_path = RESULTS_DIR / "llm_judge_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Correlations: {corr_path}")

    # markdown summary
    md_path = viz_dir / "llm_judge_summary.md"
    with open(md_path, "w") as f:
        f.write("# LLM-as-Judge Evaluation (AWS Bedrock Nova Lite)\n\n")
        f.write(f"**Pairs evaluated:** {len(df)}\n")
        f.write(f"**Model:** {MODEL_ID}\n")
        f.write(f"**LLM score distribution:** {df['llm_score'].value_counts().sort_index().to_dict()}\n\n")
        f.write("## Spearman Correlation with LLM Judgments\n\n")
        f.write("Higher correlation = method rankings align better with independent LLM assessment.\n\n")
        f.write(corr_df.to_markdown(index=False))
        f.write("\n\n## LLM Agreement with Proxy Labels\n\n")
        f.write(f"- Relevant pairs: LLM mean = {relevant.mean():.2f}\n")
        f.write(f"- Irrelevant pairs: LLM mean = {irrelevant.mean():.2f}\n")
        if len(relevant) > 0 and len(irrelevant) > 0:
            f.write(f"- Mann-Whitney p-value = {u_p:.6f}\n")
    print(f"  Summary: {md_path}")

    # generate chart
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Method Scores vs LLM Judgments", fontsize=16, fontweight="bold")

        for idx, (col, name) in enumerate(methods):
            ax = axes[idx]
            for llm_val, color, label in [(0, "red", "LLM=0 (Poor)"),
                                           (1, "orange", "LLM=1 (Partial)"),
                                           (2, "green", "LLM=2 (Strong)")]:
                subset = df[df["llm_score"] == llm_val]
                if len(subset) > 0:
                    ax.hist(subset[col], bins=15, alpha=0.5, label=label, color=color, edgecolor="black")

            rho, _ = compute_global_spearman(df, col, "llm_score")
            ax.set_title(f"{name}\nrho = {rho:.3f}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_path = viz_dir / "llm_judge_distributions.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Chart: {chart_path}")
    except Exception as e:
        print(f"  Chart generation failed: {e}")

    print("\n" + "=" * 70)
    print("LLM-AS-JUDGE EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
