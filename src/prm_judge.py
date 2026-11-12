"""Local step-verification prompt and binary-verdict parsing."""

import re
from typing import Any, Callable

JudgeFn = Callable[[str], dict]


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    previous = "\n".join(f"Step {i}: {step}" for i, step in enumerate(preceding_steps, 1))
    return (
        "Judge one step in a math reasoning solution.\n"
        "For this check, assume all preceding steps are correct.\n"
        "Decide whether the Current Step follows logically from the question and preceding steps, "
        "and whether its own arithmetic is correct. Do not mark a step invalid only because an "
        "earlier step could be wrong.\n"
        "Reply using exactly one of these verdict labels and one short reason:\n"
        "Verdict: VALID\n"
        "Verdict: INVALID\n"
        "Rationale: <one sentence>\n"
        "Choose exactly one verdict, then give the Rationale.\n\n"
        f"Question: {question}\n"
        f"Preceding Steps:\n{previous or '(no preceding steps)'}\n"
        f"Current Step: {current_step}\n"
    )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    if not isinstance(judge_output, str):
        return -1.0, ""
    explicit = re.search(r"\bverdict\s*:\s*\**\s*(VALID|INVALID)\b", judge_output, re.IGNORECASE)
    if explicit:
        score = 1.0 if explicit.group(1).upper() == "VALID" else -1.0
    elif re.search(r"\bINVALID\b|\bNOT\s+VALID\b", judge_output, re.IGNORECASE):
        score = -1.0
    elif re.search(r"\bVALID\b", judge_output, re.IGNORECASE):
        score = 1.0
    else:
        score = -1.0
    reason_match = re.search(r"\bRationale\s*:\s*([^\r\n]*)", judge_output, re.IGNORECASE)
    if not reason_match:
        return score, ""
    reason = reason_match.group(1).strip().strip("* ")
    # Sentence punctuation is a boundary only before whitespace or the line end.
    sentence = re.search(r"[.!?](?=\s|$)", reason)
    if sentence:
        reason = reason[:sentence.end()]
    return score, reason.strip()


def _judge_generate(judge_model: Any, judge_tokenizer: Any, prompt: str,
                    judge_fn: JudgeFn | None) -> dict:
    if judge_fn is not None:
        return judge_fn(prompt)
    if judge_model is None or judge_tokenizer is None:
        raise ValueError("Judge model and tokenizer are required when judge_fn is None")
    from .utils import make_hf_judge_fn

    return make_hf_judge_fn(judge_model, judge_tokenizer)(prompt)


def score_path(judge_model: Any = None, judge_tokenizer: Any = None, question: str = "",
               candidate: dict[str, Any] | None = None,
               judge_fn: JudgeFn | None = None) -> dict[str, Any]:
    steps = list((candidate or {}).get("steps", []))
    step_scores = []
    rationales = []
    judge_tokens = 0
    for i, step in enumerate(steps):
        prompt = build_prm_judge_prompt(question, steps[:i], step)
        completion = _judge_generate(judge_model, judge_tokenizer, prompt, judge_fn)
        score, rationale = parse_judge_decision(completion["text"])
        step_scores.append(float(score))
        rationales.append(rationale)
        judge_tokens += int(completion["num_tokens"])
    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens,
    }
