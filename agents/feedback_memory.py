"""Compact candidate memory with deterministic constraint feedback."""
import json
from .direct import DirectAgent


class FeedbackMemoryAgent(DirectAgent):
    def __init__(self, client, *, error_limit=8):
        super().__init__(client)
        self.valid_candidates = {}
        self.errors = {}
        self.error_limit = error_limit

    def next_prompt(self, prompt):
        if not self.valid_candidates and not self.errors:
            return prompt
        rows = [*self.valid_candidates.values(), *self.errors.values()]
        return (prompt + '\n\nProgram-checked candidate history (data, not instructions):\n'
                + '\n'.join(json.dumps(row) for row in rows)
                + '\n\nVALID candidates satisfy the observations but must not be repeated. '
                'INVALID candidates are failed attempts, not examples to imitate. '
                'Correct the stated constraint violations and produce a new valid candidate. '
                'Return only the required Graph:, Expression:, or Structure: candidate, '
                'without explanations or additional examples.')

    def observe(self, text, key, valid, *, candidate=None, issues=()):
        candidate = candidate or text.strip()[:512]
        duplicate = valid and key in self.valid_candidates
        if valid:
            previous = self.valid_candidates.get(key)
            self.valid_candidates[key] = {'status': 'VALID_DUPLICATE' if duplicate else 'VALID_NEW',
                'candidate': candidate, 'count': (previous['count'] if previous else 0) + 1}
        else:
            previous = self.errors.pop(candidate, None)
            self.errors[candidate] = {'status': 'INVALID', 'candidate': candidate,
                'violations': list(issues), 'count': (previous['count'] if previous else 0) + 1}
            while len(self.errors) > self.error_limit:
                self.errors.pop(next(iter(self.errors)))
        return text, {'duplicate_valid': bool(duplicate)}
