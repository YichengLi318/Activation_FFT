import random
import re
from typing import Dict, List, Tuple

from datasets import load_from_disk


def _sample_items(ds, limit: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: min(limit, len(ds))]
    return [ds[int(i)] for i in idxs]


def load_task_items(dataset_path: str, split: str, limit: int, seed: int) -> List[dict]:
    ds = load_from_disk(dataset_path)[split]
    return _sample_items(ds, limit, seed)


def build_prompt_and_gold(task_name: str, item: dict) -> Tuple[str, str]:
    if task_name == "ioi":
        choices = item.get("choices", [])
        a = choices[0] if len(choices) > 0 else "A"
        b = choices[1] if len(choices) > 1 else "B"
        prompt = (
            "Choose the correct indirect object.\n"
            "Keep reasoning internal and output exactly one line.\n"
            "Output format: Answer: A or Answer: B\n"
            f"Sentence: {item.get('prompt', '').strip()}\n"
            f"A. {a}\n"
            f"B. {b}\n"
            "Answer:"
        )
        ans_idx = int(item.get("answerKey", 0))
        gold = "A" if ans_idx == 0 else "B"
        return prompt, gold

    if task_name == "commonsense_qa":
        q = item.get("question", "").strip()
        labels = item.get("choices", {}).get("label", [])
        texts = item.get("choices", {}).get("text", [])
        options = "\n".join([f"{lab}. {txt}" for lab, txt in zip(labels, texts)])
        prompt = (
            "Answer the multiple-choice question. Reply with only one option letter.\n"
            f"Question: {q}\n"
            f"{options}\n"
            "Answer:"
        )
        gold = str(item.get("answerKey", "")).strip().upper()
        return prompt, gold

    if task_name == "boolq":
        q = (item.get("question") or "").strip()
        passage = (item.get("passage") or "").strip()
        prompt = (
            "Read the passage and answer the question.\n"
            "Keep reasoning internal and output exactly one line.\n"
            "Output format: Answer: Yes or Answer: No\n"
            f"Passage: {passage}\n"
            f"Question: {q}\n"
            "Answer:"
        )
        label = item.get("label")
        if label is None:
            label = item.get("answer")
        if isinstance(label, bool):
            gold = "YES" if label else "NO"
        else:
            gold = "YES" if int(label) == 1 else "NO"
        return prompt, gold

    if task_name == "blimp":
        sent_good = (item.get("sentence_good") or "").strip()
        sent_bad = (item.get("sentence_bad") or "").strip()
        prompt = (
            "Choose the grammatically correct sentence.\n"
            "Output exactly one single letter: A or B.\n"
            "Do not output any other words or symbols.\n"
            f"A. {sent_good}\n"
            f"B. {sent_bad}\n"
            "Answer:"
        )
        return prompt, "A"

    if task_name == "mawps":
        q = (item.get("question") or item.get("Question") or "").strip()
        prompt = (
            "You are a math solver.\n"
            "Keep reasoning internal and output exactly one line.\n"
            "Output format: Answer: <number>\n"
            "Examples:\n"
            "Problem: Calculate 7 + 5.\n"
            "Answer: 12\n"
            "Problem: Divide -84 by 7.\n"
            "Answer: -12\n"
            "Problem: Calculate the greatest common divisor of 45 and 12.\n"
            "Answer: 3\n"
            "Now solve:\n"
            f"Problem: {q}\n"
            "Answer:"
        )
        gold = extract_math_gold_number(item.get("answer", item.get("Answer", "")))
        return prompt, gold

    if task_name == "copa":
        premise = (item.get("premise") or "").strip()
        choice1 = (item.get("choice1") or "").strip()
        choice2 = (item.get("choice2") or "").strip()
        question = (item.get("question") or "effect").strip().lower()
        q_text = "the most plausible cause" if question == "cause" else "the most plausible effect"
        prompt = (
            "Choose the better option.\n"
            "Keep reasoning internal and output exactly one line.\n"
            "Output format: Answer: A or Answer: B\n"
            f"Premise: {premise}\n"
            f"Question: What is {q_text}?\n"
            f"A. {choice1}\n"
            f"B. {choice2}\n"
            "Answer:"
        )
        label = int(item.get("label", 0))
        gold = "A" if label == 0 else "B"
        return prompt, gold

    if task_name == "ethics_commonsense":
        text = " ".join((item.get("input", "") or "").strip().split())
        # Keep the answer cue inside max_length=256 token cap used in eval.
        max_chars = 420
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        prompt = (
            "Binary classification task.\n"
            "Return exactly one token: 0 or 1.\n"
            "0 means ACCEPTABLE. 1 means UNACCEPTABLE.\n"
            f"Action: {text}\n"
            "Label:"
        )
        label = int(item.get("label", 0))
        # ETHICS Commonsense convention: 0 = acceptable, 1 = unacceptable
        gold = "0" if label == 0 else "1"
        return prompt, gold

    raise ValueError(f"Unknown task: {task_name}")


def extract_choice_label(text: str, valid: str) -> str:
    t = (text or "").strip().upper()
    m = re.search(rf"ANSWER\s*:\s*([{re.escape(valid)}])\b", t)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{re.escape(valid)}])\b", t)
    if m:
        return m.group(1)
    for ch in t:
        if ch in valid:
            return ch
    return ""


def extract_math_gold_number(answer: str) -> str:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer or "")
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer or "")
    return nums[-1].replace(",", "") if nums else ""


def extract_number(text: str) -> str:
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or "")
    return nums[-1].replace(",", "") if nums else ""


def extract_math_pred_number(text: str) -> str:
    t = text or ""
    m_answer = re.search(r"answer\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)", t, flags=re.IGNORECASE)
    if m_answer:
        return m_answer.group(1).replace(",", "")
    patterns = [
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"final answer\s*(?:is|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"answer\s*(?:is|:)?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    ]
    for p in patterns:
        ms = list(re.finditer(p, t, flags=re.IGNORECASE))
        if ms:
            return ms[-1].group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", t)
    if not nums:
        return ""
    # Prefer the last generated number to match "final answer at the end".
    return nums[-1].replace(",", "")


def evaluate_prediction(task_name: str, generated_text: str, gold: str) -> Dict[str, object]:
    pred = ""
    if task_name == "ioi":
        pred = extract_choice_label(generated_text, "AB")
    elif task_name == "copa":
        pred = extract_choice_label(generated_text, "AB")
    elif task_name == "commonsense_qa":
        pred = extract_choice_label(generated_text, "ABCDE")
    elif task_name == "boolq":
        pred = extract_yes_no(generated_text)
    elif task_name == "blimp":
        pred = extract_choice_label(generated_text, "AB")
    elif task_name == "mawps":
        pred = extract_math_pred_number(generated_text)
    elif task_name == "ethics_commonsense":
        t = (generated_text or "").strip().upper()
        m = re.search(r"\b([01])\b", t)
        if m:
            pred = m.group(1)
        elif "UNACCEPTABLE" in t:
            pred = "1"
        elif "ACCEPTABLE" in t:
            pred = "0"
        else:
            pred = ""
    else:
        raise ValueError(f"Unknown task: {task_name}")
    correct = pred == gold
    return {"pred": pred, "gold": gold, "correct": bool(correct)}


def extract_yes_no(text: str) -> str:
    t = (text or "").strip().upper()
    m = re.search(r"ANSWER\s*:\s*(YES|NO)\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b(YES|NO)\b", t)
    if m:
        return m.group(1)
    if "TRUE" in t:
        return "YES"
    if "FALSE" in t:
        return "NO"
    if t.startswith("Y"):
        return "YES"
    if t.startswith("N"):
        return "NO"
    return ""
