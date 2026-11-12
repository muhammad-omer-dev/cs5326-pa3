"""Public PA3 grading interface. All functions use the implementations in src/."""

from typing import Any, Callable

from src.extraction import extract_answer as _extract_answer, parse_steps as _parse_steps
from src.generation import (
    compute_majority_vote as _compute_majority_vote,
    format_step_prompt as _format_step_prompt,
    run_greedy as _run_greedy,
    run_self_consistency as _run_self_consistency,
)
from src.prm_judge import (
    build_prm_judge_prompt as _build_prm_judge_prompt,
    parse_judge_decision as _parse_judge_decision,
    score_path as _score_path,
)
from src.search import (
    aggregate_path_score as _aggregate_path_score,
    select_best_path as _select_best_path,
    run_prm_best_of_n as _run_prm_best_of_n,
    run_compute_matched_self_consistency as _run_compute_matched_self_consistency,
)

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def format_step_prompt(question: str) -> str:
    return _format_step_prompt(question)


def parse_steps(completion_text: str) -> list[str]:
    return _parse_steps(completion_text)


def extract_answer(completion_text: str) -> int | None:
    return _extract_answer(completion_text)


def compute_majority_vote(answers: list[int | None]) -> int | None:
    return _compute_majority_vote(answers)


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    return _build_prm_judge_prompt(question, preceding_steps, current_step)


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    return _parse_judge_decision(judge_output)


def aggregate_path_score(step_scores: list[float], strategy: str = "min",
                         penalty_lambda: float = 2.0) -> float:
    return _aggregate_path_score(step_scores, strategy, penalty_lambda)


def select_best_path(candidate_paths: list[dict[str, Any]],
                     aggregation_strategy: str = "min", penalty_lambda: float = 2.0) -> dict[str, Any]:
    return _select_best_path(candidate_paths, aggregation_strategy, penalty_lambda)


def run_greedy(model: Any = None, tokenizer: Any = None, question: str = "",
               generate_fn: GenerateFn | None = None) -> dict[str, Any]:
    return _run_greedy(model, tokenizer, question, generate_fn)


def run_self_consistency(model: Any = None, tokenizer: Any = None, question: str = "",
                         n: int = 5, temperature: float = 0.7,
                         generate_fn: GenerateFn | None = None) -> dict[str, Any]:
    return _run_self_consistency(model, tokenizer, question, n, temperature, generate_fn)


def score_path(judge_model: Any = None, judge_tokenizer: Any = None, question: str = "",
               candidate: dict[str, Any] | None = None,
               judge_fn: JudgeFn | None = None) -> dict[str, Any]:
    return _score_path(judge_model, judge_tokenizer, question, candidate, judge_fn)


def run_prm_best_of_n(model: Any = None, judge_model: Any = None,
                      gen_tokenizer: Any = None, judge_tokenizer: Any = None,
                      question: str = "", n: int = 5,
                      aggregation_strategy: str = "min", penalty_lambda: float = 2.0,
                      candidates: list[dict[str, Any]] | None = None,
                      generate_fn: GenerateFn | None = None,
                      judge_fn: JudgeFn | None = None) -> dict[str, Any]:
    return _run_prm_best_of_n(model, judge_model, gen_tokenizer, judge_tokenizer,
                              question, n, aggregation_strategy, penalty_lambda,
                              candidates, generate_fn, judge_fn)


def run_compute_matched_self_consistency(
    model: Any = None, tokenizer: Any = None, question: str = "",
    target_calls: int = 15, temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    return _run_compute_matched_self_consistency(
        model, tokenizer, question, target_calls, temperature,
        existing_candidates, generate_fn
    )
