import math
import time
from typing import Any

from .generation import (
    _generate, _make_candidate, _sampling_kwargs, compute_majority_vote,
    format_step_prompt, run_self_consistency,
)
from .prm_judge import score_path


def aggregate_path_score(step_scores: list[float], strategy: str = "min",
                         penalty_lambda: float = 2.0) -> float:
    if strategy not in ("min", "product", "step_penalty"):
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    if not step_scores:
        return -1.0
    if strategy == "min":
        return float(min(step_scores))
    if strategy == "product":
        probabilities = [max(0.01, min(0.99, (score + 1.0) / 2.0))
                         for score in step_scores]
        return float(math.prod(probabilities))
    invalid_count = sum(score == -1.0 for score in step_scores)
    return float(sum(step_scores) / len(step_scores) - penalty_lambda * invalid_count)


def select_best_path(candidate_paths: list[dict[str, Any]],
                     aggregation_strategy: str = "min",
                     penalty_lambda: float = 2.0) -> dict[str, Any]:
    if not candidate_paths:
        raise ValueError("candidate_paths must contain at least one candidate")
    scores = [aggregate_path_score(c.get("step_scores", []),
                                   aggregation_strategy, penalty_lambda)
              for c in candidate_paths]
    eligible = [i for i, c in enumerate(candidate_paths) if c.get("answer") is not None]
    if not eligible:
        eligible = list(range(len(candidate_paths)))
    best_score = max(scores[i] for i in eligible)
    tied = [i for i in eligible if math.isclose(scores[i], best_score, rel_tol=0.0, abs_tol=1e-9)]
    vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
    chosen = next((i for i in tied if candidate_paths[i].get("answer") == vote), tied[0])
    return {**candidate_paths[chosen], "path_score": float(scores[chosen]),
            "num_tied": len(tied)}


def run_prm_best_of_n(model: Any = None, judge_model: Any = None,
                      gen_tokenizer: Any = None, judge_tokenizer: Any = None,
                      question: str = "", n: int = 5,
                      aggregation_strategy: str = "min", penalty_lambda: float = 2.0,
                      candidates: list[dict[str, Any]] | None = None,
                      generate_fn=None, judge_fn=None) -> dict[str, Any]:
    start = time.perf_counter()
    if candidates is None:
        sampled = run_self_consistency(model, gen_tokenizer, question, n=n,
                                       generate_fn=generate_fn)
        current = sampled["candidates"]
        generation_calls = sampled["generation_calls"]
        generated_tokens = sampled["generated_tokens"]
    else:
        current = candidates
        generation_calls = 0
        generated_tokens = 0
    scored = []
    judge_calls = 0
    judge_tokens = 0
    for candidate in current:
        verification = score_path(judge_model, judge_tokenizer, question,
                                  candidate, judge_fn=judge_fn)
        scored.append({**candidate, **verification})
        judge_calls += verification["judge_calls"]
        judge_tokens += verification["judge_tokens"]
    selected = select_best_path(scored, aggregation_strategy, penalty_lambda)
    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": scored,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(model: Any = None, tokenizer: Any = None,
                                         question: str = "", target_calls: int = 15,
                                         temperature: float = 0.7,
                                         existing_candidates: list[dict[str, Any]] | None = None,
                                         generate_fn=None) -> dict[str, Any]:
    start = time.perf_counter()
    existing = list(existing_candidates) if existing_candidates is not None else []
    deficit = max(0, target_calls - len(existing))
    new = _generate(model, tokenizer, format_step_prompt(question), deficit,
                    _sampling_kwargs(temperature), generate_fn)
    new_candidates = [_make_candidate(item) for item in new]
    combined = existing + new_candidates
    return {
        "candidates": combined,
        "answer": compute_majority_vote([c.get("answer") for c in combined]),
        "generation_calls": len(new_candidates),
        "total_samples": len(combined),
        "generated_tokens": sum(c["num_tokens"] for c in new_candidates),
        "time_seconds": time.perf_counter() - start,
    }
