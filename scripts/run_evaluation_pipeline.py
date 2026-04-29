"""
End-to-end evaluation pipeline runner.

Runs all steps in order, from preprocessing through final analysis.
Each step checks if its prereqs exist and reports pass/fail.

Usage:
    # full pipeline including preprocessing and fine-tuning (~1-2 hours)
    python3 scripts/run_evaluation_pipeline.py

    # skip preprocessing if data/processed/ already exists
    python3 scripts/run_evaluation_pipeline.py --skip-preprocessing

    # skip fine-tuning (faster, uses base embedding model)
    python3 scripts/run_evaluation_pipeline.py --skip-finetune

    # skip LLM judge (no AWS credentials needed)
    python3 scripts/run_evaluation_pipeline.py --skip-llm

    # quick mode: skip preprocessing + ablation (fastest re-run)
    python3 scripts/run_evaluation_pipeline.py --skip-preprocessing --skip-ablation

    # custom AWS profile for LLM judge
    python3 scripts/run_evaluation_pipeline.py --aws-profile hackathon
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENV_PYTHON = str(ROOT / "venv" / "bin" / "python3")


def run_step(step_num, name, command, required_files=None, output_files=None):
    """Run a single pipeline step and check its output."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {name}")
    print(f"{'='*70}")

    # check prereqs before running
    if required_files:
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            print(f"  SKIPPED - missing prerequisites:")
            for f in missing:
                print(f"     {f}")
            return False

    print(f"  Running: {' '.join(command)}")
    start = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max per step
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  FAILED in {elapsed:.1f}s")
            print(f"  stderr (last 20 lines):")
            for line in result.stderr.strip().split("\n")[-20:]:
                print(f"    {line}")
            return False

        print(f"  PASSED in {elapsed:.1f}s")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (exceeded 2 hours)")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # make sure expected outputs actually got created
    if output_files:
        missing = [f for f in output_files if not Path(f).exists()]
        if missing:
            print(f"  WARNING - missing expected outputs:")
            for f in missing:
                print(f"     {f}")
            return False
        print(f"  Outputs verified: {len(output_files)} files")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="skip Step 1 (use existing parsed data)")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="skip fine-tuning in Step 1 (use base model)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="skip Step 5 ablation study (~10 min saved)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip Step 12 LLM judge (no AWS needed)")
    parser.add_argument("--aws-profile", type=str, default="hackathon",
                        help="AWS profile for LLM judge (default: hackathon)")
    parser.add_argument("--ablation-jds", type=int, default=20,
                        help="number of JDs for ablation study (default: 20)")
    parser.add_argument("--llm-pairs", type=int, default=200,
                        help="number of pairs for LLM judge (default: 200)")
    args = parser.parse_args()

    print("=" * 70)
    print("RESUME-JOB MATCHING SYSTEM - FULL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"\nOptions:")
    print(f"  Skip preprocessing: {args.skip_preprocessing}")
    print(f"  Skip fine-tuning:   {args.skip_finetune}")
    print(f"  Skip ablation:      {args.skip_ablation}")
    print(f"  Skip LLM judge:     {args.skip_llm}")
    print(f"  AWS profile:        {args.aws_profile}")

    results = {}
    pipeline_start = time.time()

    # -- Step 1: Preprocessing -----------------------------------------------
    if args.skip_preprocessing:
        print(f"\n{'='*70}")
        print("STEP 1: Preprocessing - SKIPPED (--skip-preprocessing)")
        print(f"{'='*70}")
        # still need to verify existing data is there
        required = [
            "data/processed/resumes_parsed.parquet",
            "data/processed/jds_parsed.parquet",
            "data/processed/skill_idf.json",
            "data/taxonomy/skills_master.csv",
            "data/test/test_split.json",
        ]
        missing = [f for f in required if not Path(ROOT / f).exists()]
        if missing:
            print(f"  Cannot skip - missing files: {missing}")
            print(f"  Run without --skip-preprocessing first.")
            sys.exit(1)
        print(f"  Existing data verified ({len(required)} files)")
        results["Step 1"] = True
    else:
        preprocess_cmd = [VENV_PYTHON, "scripts/run_preprocessing.py"]
        if args.skip_finetune:
            preprocess_cmd.append("--skip-finetune")
        results["Step 1"] = run_step(
            1, "Preprocessing (parse, taxonomy, IDF, split, fine-tune)",
            preprocess_cmd,
            required_files=[str(ROOT / "data/raw/resumes.parquet"), str(ROOT / "data/raw/jds.parquet")],
            output_files=[
                str(ROOT / "data/processed/resumes_parsed.parquet"),
                str(ROOT / "data/processed/jds_parsed.parquet"),
                str(ROOT / "data/processed/skill_idf.json"),
                str(ROOT / "data/taxonomy/skills_master.csv"),
                str(ROOT / "data/test/test_split.json"),
            ],
        )
        if not results["Step 1"]:
            print("\nPreprocessing failed. Cannot continue.")
            sys.exit(1)

    # -- Step 2: Build eval dataset ------------------------------------------
    results["Step 2"] = run_step(
        2, "Build evaluation dataset (5,000 pairs)",
        [VENV_PYTHON, "scripts/build_eval_dataset.py", "--force"],
        required_files=[
            str(ROOT / "data/processed/resumes_parsed.parquet"),
            str(ROOT / "data/test/test_split.json"),
        ],
        output_files=[str(ROOT / "data/test/eval_pairs.parquet")],
    )
    if not results["Step 2"]:
        print("\nEval dataset build failed. Cannot continue.")
        sys.exit(1)

    # -- Step 3: TF-IDF benchmark --------------------------------------------
    results["Step 3"] = run_step(
        3, "TF-IDF benchmark",
        [VENV_PYTHON, "scripts/run_tfidf_baseline.py", "--benchmark", "--eval-pairs"],
        required_files=[str(ROOT / "data/test/eval_pairs.parquet")],
        output_files=[str(ROOT / "outputs/results/tfidf_benchmark.csv")],
    )

    # -- Step 4: Full 3-way evaluation ---------------------------------------
    results["Step 4"] = run_step(
        4, "Full evaluation (TF-IDF vs Skill-IDF vs Multi-Agent)",
        [VENV_PYTHON, "src/evaluation/evaluate_all.py"],
        required_files=[str(ROOT / "data/test/eval_pairs.parquet")],
        output_files=[
            str(ROOT / "outputs/results/comparison_results.csv"),
            str(ROOT / "outputs/results/detailed_scores.csv"),
            str(ROOT / "outputs/results/avg_comparison_results.csv"),
        ],
    )
    if not results["Step 4"]:
        print("\nMain evaluation failed. Downstream steps will be skipped.")

    # -- Steps 5-12: Analysis (all depend on Step 4) -------------------------
    # these can run in any order after step 4

    if args.skip_ablation:
        print(f"\n{'='*70}")
        print("STEP 5: Ablation study - SKIPPED (--skip-ablation)")
        print(f"{'='*70}")
        results["Step 5"] = None
    else:
        results["Step 5"] = run_step(
            5, f"Ablation study ({args.ablation_jds} JDs)",
            [VENV_PYTHON, "src/evaluation/ablation_study.py", "--n-jds", str(args.ablation_jds)],
            required_files=[str(ROOT / "data/test/eval_pairs.parquet")],
            output_files=[
                str(ROOT / "outputs/results/ablation_results.csv"),
                str(ROOT / "outputs/visualizations/ablation_study.png"),
            ],
        )

    results["Step 6"] = run_step(
        6, "Fairness analysis",
        [VENV_PYTHON, "src/evaluation/fairness_analysis.py"],
        required_files=[
            str(ROOT / "outputs/results/detailed_scores.csv"),
            str(ROOT / "data/test/eval_pairs.parquet"),
        ],
        output_files=[
            str(ROOT / "outputs/visualizations/fairness_by_experience.png"),
            str(ROOT / "outputs/visualizations/fairness_by_category.png"),
            str(ROOT / "outputs/visualizations/fairness_summary.csv"),
        ],
    )

    results["Step 7"] = run_step(
        7, "Skill miner evaluation",
        [VENV_PYTHON, "src/evaluation/evaluate_skill_miner.py"],
        required_files=[str(ROOT / "data/processed/resumes_parsed.parquet")],
        output_files=[str(ROOT / "outputs/results/skill_miner_evaluation.txt")],
    )

    results["Step 8"] = run_step(
        8, "Visualizations",
        [VENV_PYTHON, "src/evaluation/visualize_results.py"],
        required_files=[
            str(ROOT / "outputs/results/comparison_results.csv"),
            str(ROOT / "outputs/results/detailed_scores.csv"),
        ],
        output_files=[
            str(ROOT / "outputs/visualizations/metric_comparison.png"),
            str(ROOT / "outputs/visualizations/ndcg_comparison.png"),
            str(ROOT / "outputs/visualizations/score_distributions.png"),
        ],
    )

    results["Step 9"] = run_step(
        9, "Skill clustering",
        [VENV_PYTHON, "src/evaluation/skill_clustering.py"],
        required_files=[str(ROOT / "data/processed/skill_idf.json")],
        output_files=[
            str(ROOT / "outputs/visualizations/skill_clusters.png"),
            str(ROOT / "outputs/visualizations/category_similarity.png"),
        ],
    )

    results["Step 10"] = run_step(
        10, "Stress test (domain discrimination)",
        [VENV_PYTHON, "src/evaluation/stress_test.py"],
        required_files=[str(ROOT / "outputs/results/detailed_scores.csv")],
        output_files=[
            str(ROOT / "outputs/results/stress_test_separation.csv"),
            str(ROOT / "outputs/visualizations/stress_test_auc.png"),
        ],
    )

    results["Step 11"] = run_step(
        11, "Statistical significance tests",
        [VENV_PYTHON, "src/evaluation/statistical_tests.py"],
        required_files=[str(ROOT / "outputs/results/detailed_scores.csv")],
        output_files=[
            str(ROOT / "outputs/results/significance_tests.csv"),
            str(ROOT / "outputs/results/improvement_report.csv"),
            str(ROOT / "outputs/results/confidence_intervals.csv"),
        ],
    )

    if args.skip_llm:
        print(f"\n{'='*70}")
        print("STEP 12: LLM-as-Judge - SKIPPED (--skip-llm)")
        print(f"{'='*70}")
        results["Step 12"] = None
    else:
        results["Step 12"] = run_step(
            12, f"LLM-as-Judge ({args.llm_pairs} pairs, Bedrock Nova Lite)",
            [VENV_PYTHON, "src/evaluation/llm_judge.py",
             "--n-pairs", str(args.llm_pairs),
             "--profile", args.aws_profile],
            required_files=[
                str(ROOT / "outputs/results/detailed_scores.csv"),
                str(ROOT / "data/test/eval_pairs.parquet"),
            ],
            output_files=[
                str(ROOT / "outputs/results/llm_judge_scores.csv"),
                str(ROOT / "outputs/results/llm_judge_correlations.csv"),
            ],
        )

    # -- Summary -------------------------------------------------------------
    total_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal time: {total_time/60:.1f} minutes\n")

    for step, status in results.items():
        if status is True:
            icon = "PASS"
        elif status is False:
            icon = "FAIL"
        else:
            icon = "SKIP"
        print(f"  [{icon}] {step}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")

    if failed > 0:
        print(f"\n{failed} step(s) failed. Check output above for details.")
        sys.exit(1)
    else:
        print(f"\nAll steps completed successfully!")
        print(f"\nResults:  outputs/results/")
        print(f"Charts:   outputs/visualizations/")


if __name__ == "__main__":
    main()
