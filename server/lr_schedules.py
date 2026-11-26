"""Learning-rate multipliers; steps count optimizer updates, not microbatches."""


def warmup_reciprocal(step, total_steps, warmup_steps, shape=2.0):
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps or shape < 0:
        raise ValueError("Invalid schedule dimensions")
    if warmup_steps and step < warmup_steps:
        return max(0.0, step / warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    return (1.0 - progress) / (1.0 + shape * progress)
