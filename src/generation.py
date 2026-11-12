"""Step-based reasoning prompts, greedy decoding, and self-consistency."""

import time
from collections import Counter
from typing import Any, Callable

from .extraction import extract_answer, parse_steps

GenerateFn = Callable[[str, int, dict], list[dict]]


def format_step_prompt(question: str) -> str:
    return (
        "Solve the math word problem carefully. Show only numbered steps and one final answer.\n"
        "Required format:\n"
        "Step 1: <first calculation or deduction>\n"
        "Step 2: <next calculation or deduction>\n"
        "Continue with Step 3, Step 4, and so on if needed.\n"
        "Final Answer: <integer>\n\n"
        "Use plain text only. Do not use markdown, LaTeX, boxes, headings, or an extra explanation.\n"
        "Write exactly one 'Final Answer:' line as the final line, containing just an integer.\n\n"
        f"Question: {question}"
    )


def compute_majority_vote(answers: list[int | None]) -> int | None:
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    most = max(counts.values())
    return next(a for a in answers if a is not None and counts[a] == most)


def _generate(model: Any, tokenizer: Any, prompt: str, n: int, kwargs: dict,
              generate_fn: GenerateFn | None) -> list[dict]:
    if n <= 0:
        return []
    if generate_fn is not None:
        return generate_fn(prompt, n, kwargs)
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer are required when generate_fn is None")
    from .utils import make_hf_generate_fn

    hf_generate = make_hf_generate_fn(model, tokenizer)
    results = []
    for offset in range(0, n, 5):
        count = min(5, n - offset)
        results.extend(hf_generate(prompt, count, kwargs))
    return results


def _sampling_kwargs(temperature: float) -> dict:
    return {
        "do_sample": True, "temperature": temperature, "top_p": 1.0,
        "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": 512,
    }


def _make_candidate(completion: dict) -> dict:
    text = completion["text"]
    return {
        "completion": text,
        "steps": parse_steps(text),
        "answer": extract_answer(text),
        "num_tokens": int(completion["num_tokens"]),
    }


def run_greedy(model: Any = None, tokenizer: Any = None, question: str = "",
               generate_fn: GenerateFn | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    prompt = format_step_prompt(question)
    kwargs = {
        "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
        "repetition_penalty": 1.0, "max_new_tokens": 512,
    }
    completion = _generate(model, tokenizer, prompt, 1, kwargs, generate_fn)[0]
    parsed = _make_candidate(completion)
    return {
        "prompt": prompt,
        "completion": parsed["completion"],
        "steps": parsed["steps"],
        "answer": parsed["answer"],
        "generation_calls": 1,
        "generated_tokens": parsed["num_tokens"],
        "time_seconds": time.perf_counter() - start,
    }


def run_self_consistency(model: Any = None, tokenizer: Any = None, question: str = "",
                         n: int = 5, temperature: float = 0.7,
                         generate_fn: GenerateFn | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    prompt = format_step_prompt(question)
    completions = _generate(model, tokenizer, prompt, n, _sampling_kwargs(temperature), generate_fn)
    candidates = [_make_candidate(item) for item in completions]
    return {
        "candidates": candidates,
        "answer": compute_majority_vote([c["answer"] for c in candidates]),
        "generation_calls": len(candidates),
        "generated_tokens": sum(c["num_tokens"] for c in candidates),
        "time_seconds": time.perf_counter() - start,
    }
