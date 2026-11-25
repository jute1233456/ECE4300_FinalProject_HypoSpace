"""Pure helpers for balancing shared vLLM engines across gateway generations."""
import re


def engine_jobs(text):
    values=[]
    for name in ('running','waiting'):
        matches=re.findall(r'^vllm:num_requests_'+name+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)\s*$',text,re.M)
        if not matches:raise ValueError('Missing engine load metric')
        values.append(sum(float(x) for x in matches))
    return sum(values)


def choose_backend(observed,inflight,token_load):
    return min(range(len(observed)),key=lambda i:(observed[i] is None,
        max(observed[i] if observed[i] is not None else 0,inflight[i]),token_load[i],i))
