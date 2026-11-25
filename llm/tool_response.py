"""Separate Qwen thinking from a tool-assisted tree candidate before scoring."""
import re


def tree_candidate(text, branch_count):
    reasoning = ''
    if '</think>' in text:
        reasoning, text = text.rsplit('</think>', 1)
        reasoning = reasoning.removeprefix('<think>').strip()
    elif '<think>' in text:
        raise ValueError('Missing final answer after thinking')
    text = text.strip()
    branch = re.match(r'Branch:[ \t]*(\d+)[ \t]*\r?\n', text, re.I)
    if not branch or int(branch[1]) >= branch_count:
        raise ValueError('Final response does not select a listed Branch number')
    return reasoning, text
