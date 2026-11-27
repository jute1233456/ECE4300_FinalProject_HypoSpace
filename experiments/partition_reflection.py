"""Classify the complete observed answer history without reference-answer access."""
import json


def review_prompt_for_partitions(problem, records):
    answers = []
    for index, record in enumerate(records):
        answer = record.get("answer") or record.get("text")
        if answer:
            answers.append(dict(attempt=index + 1, answer=answer,
                                valid=record.get("valid"), reward=record.get("reward")))
    return (
        problem + "\n\nAll recorded candidate answers and observed outcomes:\n"
        + json.dumps(answers, ensure_ascii=False)
        + "\n\nReview the entire answer history by partitioning the legal answer space. "
        "Use explicit structural properties derived from the problem. Define disjoint classes "
        "where possible, and explain whether the classes cover all legal possibilities. "
        "Include plausible classes with no observed answers. Map observed valid answers to "
        "classes using attempt IDs; invalid answers do not establish coverage. "
        "For every class report its definition, found distinct answers, untried choices, "
        "and status OPEN, UNKNOWN or EXHAUSTED. Deduplicate repeated answers. "
        "EXHAUSTED requires a concrete argument enumerating or bounding every legal choice "
        "within that class and accounting for each; consistency of one answer, repeated outputs, "
        "or inability to think of another answer is insufficient. State uncertainty explicitly. "
        "Then choose one OPEN or UNKNOWN class and an untried condition for the next search. "
        "Output sections Partition rule, Class coverage, Exhaustion evidence, Next target. "
        "Do not submit a candidate. Your class labels and exhaustion claims are advisory "
        "and must not close tree nodes automatically."
    )
