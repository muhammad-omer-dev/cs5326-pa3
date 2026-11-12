"""PA3 benchmark and analysis runner. All paths are relative to the starter repository.

python run_all.py                   # full run, including model loading
python run_all.py --benchmark-only  # Kaggle Notebook 1, generates per_problem.jsonl
python run_all.py --analysis-only   # Kaggle Notebook 2, from attached Notebook 1 output
"""

import argparse
from pathlib import Path


def run_benchmark():
    import time
    import torch
    import transformers
    import bitsandbytes
    from datetime import datetime, timezone
    from src.utils import load_models

    if not torch.cuda.is_available():
        raise RuntimeError("Notebook 1 needs Kaggle GPU T4x2 and Internet enabled")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print('Available GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    gen_model, gen_tok, judge_model, judge_tok = load_models()

    import json
    import time
    from pathlib import Path

    from tqdm.auto import tqdm
    from transformers import set_seed

    from src.generation import run_greedy, run_self_consistency, compute_majority_vote
    from src.prm_judge import score_path
    from src.search import run_compute_matched_self_consistency, select_best_path

    OUTPUTS = Path("outputs")
    OUTPUTS.mkdir(exist_ok=True)
    DATA_PATH = Path("data/gsm8k_test_50.jsonl")
    PER_PROBLEM_PATH = OUTPUTS / "per_problem.jsonl"

    with DATA_PATH.open(encoding="utf-8") as file:
        problems = [json.loads(line) for line in file if line.strip()]
    assert len(problems) == 50, f"Expected 50 GSM8K questions; got {len(problems)}"
    assert len({p['id'] for p in problems}) == 50, "Duplicate problem ids in benchmark"

    # Load only completed records so an interrupted Kaggle run can continue safely.
    completed = {}
    if PER_PROBLEM_PATH.exists():
        with PER_PROBLEM_PATH.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if len(row.get("candidates", [])) == 15 and all(
                    key in row.get("predictions", {})
                    for key in ("greedy", "sc5", "sc15", "min", "product", "step_penalty")
                ):
                    completed[row["id"]] = row

    # First check the strict Final Answer delimiter across all 50 greedy completions.
    GREEDY_CHECKPOINT = OUTPUTS / "greedy_checkpoint.jsonl"
    greedy_by_id = {pid: row["greedy"] for pid, row in completed.items()}
    if GREEDY_CHECKPOINT.exists():
        with GREEDY_CHECKPOINT.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    entry = json.loads(line)
                    greedy_by_id[entry["id"]] = entry["greedy"]
    for problem_index, problem in enumerate(tqdm(problems, desc="Greedy delimiter check")):
        pid = problem["id"]
        if pid in greedy_by_id:
            continue
        set_seed(5326 + problem_index)
        greedy = run_greedy(gen_model, gen_tok, problem["question"])
        greedy_by_id[pid] = greedy
        with GREEDY_CHECKPOINT.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"id": pid, "greedy": greedy}, ensure_ascii=False) + "\n")
            file.flush()
    compliant = sum(greedy_by_id[p["id"]]["answer"] is not None for p in problems)
    print(f"Greedy final-answer delimiter parsed: {compliant}/50")
    if compliant < 50:
        print("Some greedy completions lack a parsable Final Answer; inspect these in the report.")

    for problem_index, problem in enumerate(tqdm(problems, desc="50 GSM8K problems")):
        pid = problem["id"]
        if pid in completed:
            continue
        set_seed(5326 + problem_index)
        question = problem["question"]

        greedy = greedy_by_id[pid]
        sc5 = run_self_consistency(gen_model, gen_tok, question, n=5, temperature=0.7)

        # All three aggregations use the same five paths and the same judge responses.
        judge_start = time.perf_counter()
        scored_first5 = []
        judge_tokens = 0
        judge_calls = 0
        for candidate in sc5["candidates"]:
            judged = score_path(judge_model, judge_tok, question, candidate)
            scored_first5.append({**candidate, "step_scores": judged["step_scores"],
                                  "rationales": judged["rationales"]})
            judge_tokens += judged["judge_tokens"]
            judge_calls += judged["judge_calls"]
        judge_time = time.perf_counter() - judge_start

        selections = {}
        selection_times = {}
        for strategy in ("min", "product", "step_penalty"):
            t0 = time.perf_counter()
            selections[strategy] = select_best_path(
                scored_first5, aggregation_strategy=strategy, penalty_lambda=2.0
            )
            selection_times[strategy] = time.perf_counter() - t0

        # Keep samples 0..4, then add exactly 10 new samples (5..14).
        extra = run_compute_matched_self_consistency(
            gen_model, gen_tok, question, target_calls=15,
            temperature=0.7, existing_candidates=sc5["candidates"]
        )
        assert extra["generation_calls"] == 10 and extra["total_samples"] == 15
        candidates = scored_first5 + extra["candidates"][5:]
        assert len(candidates) == 15
        predictions = {
            "greedy": greedy["answer"],
            "sc5": sc5["answer"],
            "sc15": extra["answer"],
            "min": selections["min"].get("answer"),
            "product": selections["product"].get("answer"),
            "step_penalty": selections["step_penalty"].get("answer"),
        }
        assert compute_majority_vote([c["answer"] for c in candidates[:5]]) == predictions["sc5"]
        assert compute_majority_vote([c["answer"] for c in candidates]) == predictions["sc15"]
        row = {
            "id": pid,
            "question": question,
            "ground_truth_answer": int(problem["ground_truth_answer"]),
            "greedy": greedy,
            "candidates": candidates,
            "predictions": predictions,
            "selections": selections,
            "judge_calls": judge_calls,
            "judge_tokens": judge_tokens,
            "timing": {
                "greedy": greedy["time_seconds"],
                "sc5_sampling": sc5["time_seconds"],
                "extra10_sampling": extra["time_seconds"],
                "judging": judge_time,
                "min_selection": selection_times["min"],
                "product_selection": selection_times["product"],
                "step_penalty_selection": selection_times["step_penalty"],
            },
        }
        # Append after the entire problem has finished; earlier lines survive disconnections.
        with PER_PROBLEM_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
        completed[pid] = row

    records = [completed[p["id"]] for p in problems]
    print(f"Completed {len(records)}/50 problems. Saved to {PER_PROBLEM_PATH}")


    OUTPUTS.joinpath("run_environment.json").write_text(json.dumps({
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "start_utc": started_utc,
        "finish_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": round(time.perf_counter() - started, 2),
    }, indent=2), encoding="utf-8")
    print("Stage 1 complete. Save Version > Save & Run All so Kaggle publishes these output files.")


def run_analysis():
    import json
    from pathlib import Path
    with Path("data/gsm8k_test_50.jsonl").open(encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    assert len(problems) == 50, "The official 50-problem file is required"

    import json
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from IPython.display import display
    import importlib.metadata

    def package_version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    from src.generation import compute_majority_vote
    from src.search import select_best_path

    output_dir = Path("outputs")
    method_definitions = [
        ("Greedy (N=1)", "greedy"),
        ("Self-Consistency (N=5)", "sc5"),
        ("Compute-Matched SC (N=15)", "sc15"),
        ("PRM Best-of-5 (min)", "min"),
        ("PRM Best-of-5 (product)", "product"),
        ("PRM Best-of-5 (step_penalty, lambda=2.0)", "step_penalty"),
    ]

    # Keep exactly one complete record per question and restore official dataset order.
    with (output_dir / "per_problem.jsonl").open(encoding="utf-8") as file:
        full_records = {row["id"]: row for line in file if line.strip()
                        for row in [json.loads(line)]}
    records = [full_records[p["id"]] for p in problems]
    assert len(records) == 50
    n_problems = len(records)

    def correct(row, answer):
        return answer is not None and int(answer) == row["ground_truth_answer"]

    def percent(num, den=n_problems):
        return round(100.0 * num / den, 2)

    rows = []
    for display_name, key in method_definitions:
        successes = sum(correct(r, r["predictions"][key]) for r in records)
        if key == "greedy":
            gen_calls = n_problems
            judge_calls = 0
            gen_tokens = sum(r["greedy"]["generated_tokens"] for r in records)
            judge_tokens = 0
            duration = sum(r["timing"]["greedy"] for r in records)
        elif key in ("sc5", "sc15"):
            samples = 5 if key == "sc5" else 15
            gen_calls = samples * n_problems
            judge_calls = 0
            gen_tokens = sum(sum(c["num_tokens"] for c in r["candidates"][:samples])
                             for r in records)
            judge_tokens = 0
            duration = sum(r["timing"]["sc5_sampling"] +
                           (r["timing"]["extra10_sampling"] if samples == 15 else 0.0)
                           for r in records)
        else:
            gen_calls = 5 * n_problems
            judge_calls = sum(r["judge_calls"] for r in records)
            gen_tokens = sum(sum(c["num_tokens"] for c in r["candidates"][:5])
                             for r in records)
            judge_tokens = sum(r["judge_tokens"] for r in records)
            duration = sum(r["timing"]["sc5_sampling"] + r["timing"]["judging"] +
                           r["timing"][key + "_selection"] for r in records)
        rows.append({
            "Method": display_name,
            "Accuracy (%)": percent(successes),
            "Generation Calls": gen_calls,
            "Judge Calls": judge_calls,
            "Total LLM Calls": gen_calls + judge_calls,
            "Avg Gen Tokens / Problem": round(gen_tokens / n_problems, 2),
            "Avg Judge Tokens / Problem": round(judge_tokens / n_problems, 2),
            "Total Time (s)": round(duration, 2),
        })
    with (output_dir / "results_summary.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    ablation = []
    for strategy, lambdas in (("min", [None]), ("product", [None]),
                               ("step_penalty", [0.0, 0.5, 2.0])):
        for lam in lambdas:
            selected = [select_best_path(r["candidates"][:5], strategy,
                                         penalty_lambda=2.0 if lam is None else lam)
                        for r in records]
            ablation.append({
                "Strategy": strategy,
                "Lambda": lam,
                "Accuracy (%)": percent(sum(correct(r, s.get("answer"))
                                            for r, s in zip(records, selected))),
                "Tie Rate (%)": percent(sum(s["num_tied"] > 1 for s in selected)),
            })
    with (output_dir / "aggregation_ablation.json").open("w", encoding="utf-8") as file:
        json.dump(ablation, file, indent=2)

    scaling = []
    for n in range(1, 16):
        predictions = [compute_majority_vote([c["answer"] for c in r["candidates"][:n]])
                       for r in records]
        gen_tokens = sum(sum(c["num_tokens"] for c in r["candidates"][:n])
                         for r in records)
        scaling.append({
            "N": n,
            "Accuracy (%)": percent(sum(correct(r, a) for r, a in zip(records, predictions))),
            "Avg Gen Tokens / Problem": round(gen_tokens / n_problems, 2),
        })
    with (output_dir / "sc_scaling_curve.json").open("w", encoding="utf-8") as file:
        json.dump(scaling, file, indent=2)
    assert scaling[4]["Accuracy (%)"] == rows[1]["Accuracy (%)"]
    assert scaling[14]["Accuracy (%)"] == rows[2]["Accuracy (%)"]
    assert all(rows[i][k] == rows[3][k] for i in (4, 5) for k in (
        "Generation Calls", "Judge Calls", "Total LLM Calls",
        "Avg Gen Tokens / Problem", "Avg Judge Tokens / Problem"
    ))

    # Report section 1: 1,000 paired-problem bootstrap resamples, seed 5326.
    rng = np.random.default_rng(5326)
    bootstrap_indices = rng.integers(0, n_problems, size=(1000, n_problems))
    bootstrap = []
    for display_name, key in method_definitions:
        values = np.array([int(correct(r, r["predictions"][key])) for r in records])
        sampled = values[bootstrap_indices].mean(axis=1) * 100
        bootstrap.append({
            "Method": display_name,
            "Accuracy (%)": percent(values.sum()),
            "CI 95% Low": round(float(np.percentile(sampled, 2.5)), 2),
            "CI 95% High": round(float(np.percentile(sampled, 97.5)), 2),
        })

    paired = []
    paired_diff_intervals = []
    for prm_key in ("min", "product", "step_penalty"):
        for baseline_key in ("sc5", "sc15"):
            prm_correct = [correct(r, r["predictions"][prm_key]) for r in records]
            baseline_correct = [correct(r, r["predictions"][baseline_key]) for r in records]
            wins = sum(p and not b for p, b in zip(prm_correct, baseline_correct))
            losses = sum(b and not p for p, b in zip(prm_correct, baseline_correct))
            difference = (np.array(prm_correct, dtype=int) -
                          np.array(baseline_correct, dtype=int))
            boot_diff = difference[bootstrap_indices].mean(axis=1) * 100
            paired_diff_intervals.append({
                "PRM": prm_key, "Baseline": baseline_key,
                "Accuracy Gap (percentage points)": round(float(difference.mean() * 100), 2),
                "Paired Bootstrap CI Low": round(float(np.percentile(boot_diff, 2.5)), 2),
                "Paired Bootstrap CI High": round(float(np.percentile(boot_diff, 97.5)), 2),
            })
            paired.append({
                "PRM": prm_key, "Baseline": baseline_key,
                "Wins": wins, "Losses": losses, "Ties": n_problems - wins - losses,
            })

    # Report section 4: fixed token budget, including BOTH generator and judge outputs.
    greedy_tokens = sum(r["greedy"]["generated_tokens"] for r in records)
    budget = 10 * greedy_tokens
    sc_under_budget = [entry for entry in scaling if
                       sum(c["num_tokens"] for r in records
                           for c in r["candidates"][:entry["N"]]) <= budget]
    prm_token_spend = sum(sum(c["num_tokens"] for c in r["candidates"][:5]) +
                          r["judge_tokens"] for r in records)
    prm_under_budget = [r["Method"] for r in rows[3:] if prm_token_spend <= budget]
    compute_budget = {
        "Greedy Total Output Tokens": greedy_tokens,
        "Ten Times Greedy Output Token Budget": budget,
        "Largest SC N Within Budget": max((s["N"] for s in sc_under_budget), default=None),
        "Best SC Under Budget": max(sc_under_budget,
                                    key=lambda entry: entry["Accuracy (%)"], default=None),
        "PRM Methods Within Budget": prm_under_budget,
        "PRM Total Output Tokens": prm_token_spend,
        "PRM Total Judge Calls": sum(r["judge_calls"] for r in records),
        "PRM Avg Output Tokens Per Judge Call": round(
            sum(r["judge_tokens"] for r in records) /
            max(1, sum(r["judge_calls"] for r in records)), 2),
        "PRM Avg Output Tokens Per Generation Call": round(
            sum(c["num_tokens"] for r in records for c in r["candidates"][:5]) /
            (5 * n_problems), 2),
    }

    environment_path = output_dir / "run_environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    metadata = {
        "Seed Protocol": "5326 + problem_index",
        "Torch Version": environment.get("torch_version", package_version("torch")),
        "Transformers Version": environment.get("transformers_version", package_version("transformers")),
        "Bitsandbytes Version": environment.get("bitsandbytes_version", package_version("bitsandbytes")),
        "GPUs": environment.get("gpu_names", []),
        "Benchmark Wall Time (s)": environment.get("wall_time_seconds"),
        "Sample Size": n_problems,
    }
    report_statistics = {
        "bootstrap_95_ci_1000_resamples": bootstrap,
        "paired_win_loss": paired,
        "paired_bootstrap_accuracy_difference": paired_diff_intervals,
        "ten_x_greedy_budget": compute_budget,
        "reproducibility": metadata,
    }
    with (output_dir / "report_statistics.json").open("w", encoding="utf-8") as file:
        json.dump(report_statistics, file, indent=2)

    print("Results summary")
    display(pd.DataFrame(rows))
    print("Aggregation ablation")
    display(pd.DataFrame(ablation))
    print("95% bootstrap confidence intervals")
    display(pd.DataFrame(bootstrap))
    print("Paired win/loss counts")
    display(pd.DataFrame(paired))
    print("Paired bootstrap accuracy differences")
    display(pd.DataFrame(paired_diff_intervals))
    print("10x greedy token budget:", compute_budget)

    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    prm_rows = rows[3:]
    plot_markers = ["o", "s", "^", "D"]

    # Figure 1: Self-consistency scaling and PRM at their full LLM-call costs.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([50 * item["N"] for item in scaling],
            [item["Accuracy (%)"] for item in scaling], "o-", label="Self-Consistency N=1..15")
    for row, marker in zip(prm_rows, plot_markers):
        ax.scatter(row["Total LLM Calls"], row["Accuracy (%)"],
                   marker=marker, s=90, label=row["Method"], zorder=3)
    ax.set(xlabel="Total LLM Calls (50 problems)", ylabel="Exact-match accuracy (%)",
           title="Accuracy vs. Total LLM Calls")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure1_accuracy_vs_llm_calls.png", dpi=180)
    plt.close(fig)

    # Figure 2: Both generator and judge output tokens count as compute.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sc_tokens = [sum(c["num_tokens"] for row in records
                     for c in row["candidates"][:point["N"]]) for point in scaling]
    ax.plot(sc_tokens, [point["Accuracy (%)"] for point in scaling],
            "o-", label="Self-Consistency N=1..15")
    prm_total_tokens = sum(sum(c["num_tokens"] for c in row["candidates"][:5]) +
                           row["judge_tokens"] for row in records)
    for row, marker in zip(prm_rows, plot_markers):
        ax.scatter(prm_total_tokens, row["Accuracy (%)"],
                   marker=marker, s=90, label=row["Method"], zorder=3)
    ax.set(xlabel="Total Output Tokens (generator + judge, 50 problems)",
           ylabel="Exact-match accuracy (%)", title="Accuracy vs. Total Output Tokens")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure2_accuracy_vs_output_tokens.png", dpi=180)
    plt.close(fig)

    # Figure 3: Path aggregation ablation accuracy.
    labels = [a["Strategy"] if a["Lambda"] is None else
              f"step_penalty\nλ={a['Lambda']:g}" for a in ablation]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(ablation)), [a["Accuracy (%)"] for a in ablation])
    ax.set_xticks(range(len(ablation)), labels)
    ax.set(ylabel="Exact-match accuracy (%)", title="Aggregation Ablation: Accuracy",
           ylim=(0, 100))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure3_aggregation_accuracy.png", dpi=180)
    plt.close(fig)

    # Figure 4: Rates of scores tied at the maximum eligible path score.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(ablation)), [a["Tie Rate (%)"] for a in ablation])
    ax.set_xticks(range(len(ablation)), labels)
    ax.set(ylabel="Problems with a top-score tie (%)", title="Aggregation Ablation: Tie Rate",
           ylim=(0, 100))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure4_aggregation_tie_rate.png", dpi=180)
    plt.close(fig)

    # Figure 5: 95% bootstrap confidence intervals (1,000 resamples).
    method_names = [b["Method"] for b in bootstrap]
    values = np.array([b["Accuracy (%)"] for b in bootstrap])
    lows = np.array([b["CI 95% Low"] for b in bootstrap])
    highs = np.array([b["CI 95% High"] for b in bootstrap])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_positions = np.arange(len(bootstrap))
    ax.errorbar(values, y_positions, xerr=[values - lows, highs - values],
                fmt="o", capsize=5)
    ax.set_yticks(y_positions, method_names)
    ax.invert_yaxis()
    ax.set(xlabel="Exact-match accuracy (%)", title="95% Bootstrap Confidence Intervals",
           xlim=(0, 100))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure5_bootstrap_confidence_intervals.png", dpi=180)
    plt.close(fig)

    print("Saved report graphs to", fig_dir.resolve())

    import csv
    import random
    from collections import Counter
    from pathlib import Path

    AUDIT_PATH = Path("outputs/judge_audit.csv")
    if not AUDIT_PATH.exists():
        step_rows = []
        for row in records:
            for candidate_index, candidate in enumerate(row["candidates"][:5]):
                for step_index, (step, score, rationale) in enumerate(zip(
                    candidate["steps"], candidate["step_scores"], candidate["rationales"]
                )):
                    step_rows.append({
                        "problem_id": row["id"],
                        "question": row["question"],
                        "candidate_index": candidate_index,
                        "step_index": step_index + 1,
                        "preceding_steps": " | ".join(candidate["steps"][:step_index]),
                        "current_step": step,
                        "judge_label": "INVALID" if score == -1.0 else "VALID",
                        "judge_rationale": rationale,
                        "human_label": "",
                    })
        invalid = [row for row in step_rows if row["judge_label"] == "INVALID"]
        valid = [row for row in step_rows if row["judge_label"] == "VALID"]
        rand = random.Random(5326)
        rand.shuffle(invalid)
        rand.shuffle(valid)
        audit = invalid[:10] + valid[:20]
        if len(audit) < 30:
            remaining = [row for row in invalid[10:] + valid[20:]]
            rand.shuffle(remaining)
            audit += remaining[:30 - len(audit)]
        rand.shuffle(audit)
        fields = ["problem_id", "question", "candidate_index", "step_index",
                  "preceding_steps", "current_step", "judge_label",
                  "judge_rationale", "human_label"]
        with AUDIT_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(audit)
        print(f"Created {AUDIT_PATH} with {len(audit)} steps, including "
              f"{sum(r['judge_label'] == 'INVALID' for r in audit)} judge-INVALID steps.")
        if len(invalid) < 10:
            print("Important: Fewer than 10 INVALID judge decisions exist in this run. "
                  "Explain this in the report; do not invent labels.")
    else:
        print(f"Kept existing {AUDIT_PATH} so your human labels are not overwritten.")

    print("Open the CSV and read each question, prefix, and current step. "
          "Fill human_label with VALID or INVALID using local validity (assuming preceding steps are correct).")

    import csv
    import json
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    AUDIT_PATH = Path("outputs/judge_audit.csv")
    with AUDIT_PATH.open(encoding="utf-8", newline="") as file:
        audit_rows = list(csv.DictReader(file))
    judged = [r for r in audit_rows if r["human_label"].strip().upper() in ("VALID", "INVALID")]
    invalid_count = sum(r["judge_label"] == "INVALID" for r in judged)
    if len(judged) < 30 or invalid_count < 10:
        print(f"Audit incomplete: {len(judged)}/30 human labels entered; "
              f"{invalid_count}/10 judge-INVALID examples labeled.")
        print("Fill human_label in outputs/judge_audit.csv and rerun this cell. "
              "Do not invent the ground-truth step labels.")
    else:
        # Positive class is INVALID, as required by the assignment.
        tp = sum(r["judge_label"] == "INVALID" and r["human_label"].strip().upper() == "INVALID"
                 for r in judged)
        fp = sum(r["judge_label"] == "INVALID" and r["human_label"].strip().upper() == "VALID"
                 for r in judged)
        tn = sum(r["judge_label"] == "VALID" and r["human_label"].strip().upper() == "VALID"
                 for r in judged)
        fn = sum(r["judge_label"] == "VALID" and r["human_label"].strip().upper() == "INVALID"
                 for r in judged)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        audit_metrics = {
            "audited_steps": len(judged), "judge_invalid_steps": invalid_count,
            "positive_class": "INVALID", "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "invalid_precision": precision, "invalid_recall": recall,
        }
        Path("outputs/judge_audit_metrics.json").write_text(
            json.dumps(audit_metrics, indent=2), encoding="utf-8"
        )
        print("Judge audit: ", audit_metrics)
        cm = np.array([[tp, fn], [fp, tn]])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Predicted INVALID", "Predicted VALID"])
        ax.set_yticks([0, 1], ["Actually INVALID", "Actually VALID"])
        for (r, c), value in np.ndenumerate(cm):
            ax.text(c, r, str(value), ha="center", va="center", fontsize=16)
        ax.set_title("Judge Confusion Matrix (positive = INVALID)")
        fig.tight_layout()
        fig.savefig("outputs/figures/figure6_judge_confusion_matrix.png", dpi=180)
        plt.close(fig)
        false_positive_valid = [r for r in judged if r["judge_label"] == "VALID"
                                and r["human_label"].strip().upper() == "INVALID"]
        false_negative_valid = [r for r in judged if r["judge_label"] == "INVALID"
                                and r["human_label"].strip().upper() == "VALID"]
        print("Flawed step judged VALID (judge false positive):")
        display(pd.DataFrame(false_positive_valid[:3]))
        print("Valid step judged INVALID (judge false negative):")
        display(pd.DataFrame(false_negative_valid[:3]))

    # Third report case: first five paths share a frequent wrong answer.
    collapse_examples = []
    for row in records:
        answers = [c["answer"] for c in row["candidates"][:5] if c["answer"] is not None]
        frequencies = Counter(answers)
        if frequencies:
            most_common_answer, count = frequencies.most_common(1)[0]
            if count >= 4 and most_common_answer != row["ground_truth_answer"]:
                collapse_examples.append({
                    "id": row["id"], "question": row["question"],
                    "ground_truth_answer": row["ground_truth_answer"],
                    "shared_wrong_answer": most_common_answer, "count_out_of_5": count,
                    "first_candidate_steps": row["candidates"][0]["steps"],
                })
    print(f"Shared wrong answers in at least 4 of 5 paths: {len(collapse_examples)} problems")
    display(pd.DataFrame(collapse_examples[:3]))
    if not collapse_examples:
        print("No strong shared-answer collapse was observed; discuss the closest shared-premise case manually.")


    for required_file in ("per_problem.jsonl", "results_summary.json", "aggregation_ablation.json", "sc_scaling_curve.json"):
        target = Path("outputs") / required_file
        assert target.is_file(), f"Missing required output: {target}"
    print("Stage 2 complete: all four required output files and report graphs are saved.")


def main():
    parser = argparse.ArgumentParser(description="PA3 local benchmark and report statistics")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark-only", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.benchmark_only:
        run_benchmark()
    elif args.analysis_only:
        run_analysis()
    else:
        run_benchmark()
        run_analysis()


if __name__ == "__main__":
    main()
