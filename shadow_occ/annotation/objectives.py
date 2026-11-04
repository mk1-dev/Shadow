from __future__ import annotations

import re


def assistant_token_groups(tokenizer, full_text, answer, supervised_ids):
    """Align text offsets to the processor's supervised suffix, not image tokens."""
    import torch

    start = full_text.rfind(answer)
    if start < 0:
        raise ValueError("Assistant answer absent from chat template")
    raw = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets, ids = raw["offset_mapping"], raw["input_ids"]
    first = next((i for i, (a, b) in enumerate(offsets) if b > start and a < start+len(answer)), None)
    if first is None or ids[first:] != supervised_ids:
        raise ValueError("Token offset alignment failed; check model/processor versions")
    match = re.search(r'"shadow"\s*:\s*"(YES|NO|UNCERTAIN)"', answer)
    if match is None:
        raise ValueError("Component loss/scoring requires a JSON shadow decision")
    decision_start, decision_end = start+match.start(1), start+match.end(1)+1
    caster = re.search(r'"caster"\s*:', answer)
    caster_start = start+caster.end() if caster else start+len(answer)
    groups = []
    for a, b in offsets[first:]:
        if b > decision_start and a < decision_end:
            groups.append(1)
        elif b > caster_start and a < start+len(answer):
            groups.append(2)
        else:
            groups.append(3)
    if 1 not in groups:
        raise ValueError("No shadow-decision tokens found")
    return torch.tensor(groups, dtype=torch.long)


def component_loss(logits, labels, groups, decision_weight, weights):
    import torch
    from torch.nn import functional as F

    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise ValueError("No supervised response tokens")
    # Upcast only response tokens, not the potentially long visual prompt.
    losses = F.cross_entropy(logits[:, :-1][valid].float(), labels[:, 1:][valid], reduction="none")
    selected_groups = groups[:, 1:][valid]
    components, total, normalizer = {}, losses.sum()*0, 0.0
    for index, name in ((1, "decision"), (2, "association"), (3, "format")):
        selected = selected_groups == index
        if bool(selected.any()) and weights[name] > 0:
            value = losses[selected].mean()
            components[name] = float(value.detach())
            multiplier = decision_weight if name == "decision" else 1.0
            total = total+weights[name]*multiplier*value
            normalizer += weights[name]
    if normalizer <= 0:
        raise ValueError("No active component loss")
    return total/normalizer, components
