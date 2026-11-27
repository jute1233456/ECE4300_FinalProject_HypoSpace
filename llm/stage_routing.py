"""Route reflection to a separate adapter without mutating the search client."""
from copy import copy
from types import MethodType

from .base import LLMError


def route_reflection(client, model):
    if not model or model == client.model:
        raise ValueError('Reflection routing requires a distinct, explicit model')
    if getattr(client, 'stage_models', None):
        raise ValueError('Stage routing is already configured')
    primary = client.model
    reviewer = copy(client)
    reviewer.model = model
    original_review = type(client).review

    def review(self, prompt, *, max_tokens):
        response = original_review(reviewer, prompt, max_tokens=max_tokens)
        if response.model != model:
            raise LLMError('Reflection backend returned a different model; no fallback allowed')
        assert self.model == primary
        return response

    client.review = MethodType(review, client)
    client.stage_models = dict(reflection=model, read_tree=primary, selection=primary,
                               expansion=primary, candidate=primary)
    return client.stage_models
