from functools import partial
from typing import Any, Mapping


LETTERS = tuple("ABCDEFGHIJ")


def _options(doc: Mapping[str, Any]) -> list[str]:
    options = doc["options"]
    if (
        not isinstance(options, list)
        or not 2 <= len(options) <= len(LETTERS)
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError("MMLU-Pro MC requires two to ten nonempty answer options")
    return [option.strip() for option in options]


def doc_to_text(doc: Mapping[str, Any]) -> str:
    question = doc["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("MMLU-Pro MC requires a nonempty question")
    options = _options(doc)
    lines = ["Question:", question, "Options:"]
    lines.extend(f"{LETTERS[index]}. {option}" for index, option in enumerate(options))
    lines.append("Answer:")
    return "\n".join(lines)


def doc_to_choice(doc: Mapping[str, Any]) -> list[str]:
    return list(LETTERS[:len(_options(doc))])


def doc_to_target(doc: Mapping[str, Any]) -> int:
    choices = doc_to_choice(doc)
    index = doc["answer_index"]
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < len(choices)
        or doc["answer"] != choices[index]
    ):
        raise ValueError("MMLU-Pro answer index and letter disagree")
    return index


def _subject_docs(dataset: Any, *, subject: str) -> Any:
    # Reuse the pinned upstream partition without making prompt tests import the harness.
    from lm_eval.tasks.mmlu_pro.utils import process_docs

    return process_docs(dataset, subject)


process_biology = partial(_subject_docs, subject="biology")
process_business = partial(_subject_docs, subject="business")
process_chemistry = partial(_subject_docs, subject="chemistry")
process_computer_science = partial(_subject_docs, subject="computer science")
process_economics = partial(_subject_docs, subject="economics")
process_engineering = partial(_subject_docs, subject="engineering")
process_health = partial(_subject_docs, subject="health")
process_history = partial(_subject_docs, subject="history")
process_law = partial(_subject_docs, subject="law")
process_math = partial(_subject_docs, subject="math")
process_other = partial(_subject_docs, subject="other")
process_philosophy = partial(_subject_docs, subject="philosophy")
process_physics = partial(_subject_docs, subject="physics")
process_psychology = partial(_subject_docs, subject="psychology")
