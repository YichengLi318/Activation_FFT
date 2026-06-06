from typing import List, Tuple

import torch

from function import activation
from function.agreement_tasks import build_prompt_and_gold, evaluate_prediction
from function.perturb import _get_decoder_layers, _run_single_text


def _max_new_tokens_for_task(task_name: str, generation_cfg: dict) -> int:
    if task_name == "mawps":
        return int(generation_cfg.get("max_new_tokens_mawps", 64))
    if task_name == "boolq":
        return int(generation_cfg.get("max_new_tokens_boolq", generation_cfg.get("max_new_tokens_choice", 8)))
    if task_name == "ethics_commonsense":
        return int(generation_cfg.get("max_new_tokens_ethics", generation_cfg.get("max_new_tokens_choice", 8)))
    return int(generation_cfg.get("max_new_tokens_choice", 8))


def _extract_layer_vector(
    model,
    tokenizer,
    device: str,
    text: str,
    max_length: int,
    layer_index: int,
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
    layers = _get_decoder_layers(model)
    resolved = layer_index if layer_index >= 0 else len(layers) + layer_index
    resolved = max(0, min(resolved, len(layers) - 1))
    hs = outputs.hidden_states[resolved + 1][0]
    valid_len = int(encoded["attention_mask"][0].sum().item())
    valid_len = max(valid_len, 1)
    return hs[:valid_len].mean(dim=0).detach().cpu()


def collect_calibration_vectors_and_labels(
    model,
    tokenizer,
    device: str,
    task_name: str,
    items: List[dict],
    layer_index: int,
    generation_cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompts = []
    golds = []
    for item in items:
        prompt, gold = build_prompt_and_gold(task_name, item)
        prompts.append(prompt)
        golds.append(gold)
    target_len = activation._compute_uniform_token_length(tokenizer, prompts, fast_mode=True)
    target_len = min(target_len, 256)

    vecs = []
    labels = []
    for i in range(len(items)):
        prompt = prompts[i]
        gold = golds[i]
        max_new = _max_new_tokens_for_task(task_name, generation_cfg)
        out = _run_single_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            text=prompt,
            max_length=target_len,
            layer_index=layer_index,
            hook=None,
            topk=1,
            max_new_tokens=max_new,
            temperature=generation_cfg.get("temperature", 0.0),
            max_time=generation_cfg.get("max_time", None),
        )
        pred = evaluate_prediction(task_name, (out.get("generated") or "").strip(), gold)
        vecs.append(_extract_layer_vector(model, tokenizer, device, prompt, target_len, layer_index))
        labels.append(1 if pred["correct"] else 0)

    if not vecs:
        return torch.empty(0), torch.empty(0)
    x = torch.stack(vecs, dim=0).float()
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def _binary_split_indices(y: torch.Tensor, n: int):
    pos = (y == 1).nonzero(as_tuple=True)[0]
    neg = (y == 0).nonzero(as_tuple=True)[0]
    if len(pos) == 0 or len(neg) == 0:
        half = max(1, n // 2)
        pos = torch.arange(0, half, dtype=torch.long)
        neg = torch.arange(half, n, dtype=torch.long)
        if len(neg) == 0:
            neg = torch.arange(0, half, dtype=torch.long)
    return pos, neg


def select_diffmean_dims_from_calibration(x: torch.Tensor, y: torch.Tensor, topk: int) -> List[int]:
    if x.numel() == 0:
        return []
    n, h = x.shape
    pos_idx, neg_idx = _binary_split_indices(y, n)
    pos_mean = x[pos_idx].mean(dim=0)
    neg_mean = x[neg_idx].mean(dim=0)
    score = (pos_mean - neg_mean).abs()
    k = min(int(topk), int(h))
    return [int(d) for d in torch.topk(score, k=k).indices.tolist()]


def select_probing_dims_from_calibration(
    x: torch.Tensor,
    y: torch.Tensor,
    topk: int,
    steps: int = 200,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
) -> List[int]:
    if x.numel() == 0:
        return []
    n, h = x.shape
    pos_idx, neg_idx = _binary_split_indices(y, n)
    y_bin = torch.zeros(n, dtype=torch.float32)
    y_bin[pos_idx] = 1.0

    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_norm = (x - x_mean) / x_std

    clf = torch.nn.Linear(h, 1, bias=True)
    opt = torch.optim.Adam(clf.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(max(1, int(steps))):
        logits = clf(x_norm).squeeze(-1)
        loss = loss_fn(logits, y_bin)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        # map back to original feature scale
        w = clf.weight[0].abs() / x_std.squeeze(0)
        k = min(int(topk), int(h))
        return [int(d) for d in torch.topk(w, k=k).indices.tolist()]
