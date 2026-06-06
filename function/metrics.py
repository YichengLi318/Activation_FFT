from typing import Dict, List, Tuple

import torch


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def jaccard(set_a, set_b) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def error_breakdown(
    pred: List[Tuple[int, int, str]],
    gold: List[Tuple[int, int, str]],
) -> Dict[str, int]:
    pred_set = set(pred)
    gold_set = set(gold)
    gold_pairs = {(h, t) for h, t, _r in gold_set}

    wrong_rel = 0
    wrong_head = 0
    wrong_tail = 0
    wrong_both = 0

    for h, t, r in pred_set - gold_set:
        if (h, t) in gold_pairs:
            wrong_rel += 1
        else:
            head_in_gold = any(h == gh for gh, _gt, _gr in gold_set)
            tail_in_gold = any(t == gt for _gh, gt, _gr in gold_set)
            if head_in_gold and not tail_in_gold:
                wrong_tail += 1
            elif tail_in_gold and not head_in_gold:
                wrong_head += 1
            else:
                wrong_both += 1

    return {
        "wrong_relation_same_pair": wrong_rel,
        "wrong_head_only": wrong_head,
        "wrong_tail_only": wrong_tail,
        "wrong_head_tail": wrong_both,
    }


def attention_js_distance(
    base_attn: torch.Tensor,
    pert_attn: torch.Tensor,
    valid_len: int,
    eps: float = 1e-9,
) -> float:
    if base_attn is None or pert_attn is None or valid_len <= 0:
        return 0.0
    # attention shape: (1, heads, seq, seq)
    a = base_attn[0, :, :valid_len, :valid_len].float()
    b = pert_attn[0, :, :valid_len, :valid_len].float()
    a = a.clamp_min(eps)
    b = b.clamp_min(eps)
    m = 0.5 * (a + b)
    js = 0.5 * (a * (a / m).log()).sum(dim=-1) + 0.5 * (b * (b / m).log()).sum(dim=-1)
    return float(js.mean().item())

