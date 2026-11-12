import re
from decimal import Decimal, InvalidOperation

_STEP_HEADER = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__|`)?[ \t]*"
    r"Step[ \t]*\d+[ \t]*[:.]?[ \t]*(?:\*\*|__|`)?[ \t]*"
)
_FINAL_HEADER = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__|`)?[ \t]*"
    r"Final[ \t]+Answer[ \t]*(?:\*\*|__|`)?[ \t]*:?[ \t]*(?:\*\*|__|`)?[ \t]*"
)
_NUMBER = re.compile(r"^\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_SUFFIX = re.compile(r"^\s*(?:[A-Za-z]+(?:\s+[A-Za-z]+)*)?\s*[.!?]?\s*$")


def parse_steps(completion_text: str) -> list[str]:
    """Extract numbered reasoning steps, allowing common markdown formatting."""
    if not isinstance(completion_text, str) or not completion_text:
        return []
    matches = list(_STEP_HEADER.finditer(completion_text))
    if not matches:
        return []
    steps = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(completion_text)
        answer_heading = _FINAL_HEADER.search(completion_text, match.end(), end)
        if answer_heading:
            end = answer_heading.start()
        content = completion_text[match.end():end].strip().strip("*#`_ ").strip()
        if content:
            steps.append(content)
    return steps


def extract_answer(completion_text: str) -> int | None:
    """Read the last Final Answer line; accept integers and integer-valued decimals only."""
    if not isinstance(completion_text, str) or not completion_text:
        return None
    headings = list(_FINAL_HEADER.finditer(completion_text))
    if not headings:
        return None
    last = headings[-1]
    line = completion_text[last.end():].splitlines()[0].strip().strip("*`_ ").strip()
    found = _NUMBER.match(line)
    if not found or not _SUFFIX.fullmatch(line[found.end():]):
        return None
    try:
        number = Decimal(found.group(1).replace(",", ""))
        return int(number) if number == number.to_integral_value() else None
    except (InvalidOperation, ValueError, OverflowError):
        return None
