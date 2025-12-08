import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reduce thread contention and tokenizer warnings
torch.set_num_threads(1)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_model(model_dir_name: str = 'Qwen3-1.7B'):
    """
    简化版模型加载，仅支持当前单一模型（Qwen3-1.7B 或对应 HF 仓库名）。
    - 优先从本地 'model/<model_dir_name>' 目录加载；若不存在则按 HF 仓库名加载。
    - 统一设置右侧填充；如无 pad_token 则使用 eos_token 作为 pad_token（不新增词表）。
    - 根据设备选择 dtype：CUDA 用 float16，CPU 用 float32。
    返回 (tokenizer, model, device)。
    """
    model_path = os.path.join(os.path.dirname(__file__), 'model', model_dir_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    print(f"========================================")
    print(f"           USING DEVICE: {device.upper()}       ")
    print(f"========================================")

    def _prepare_tokenizer_model(tokenizer, model):
        # 统一右侧填充；如无 pad_token，直接复用 eos_token，避免新增词表与 resize
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token is None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass
        model.eval()
        return tokenizer, model

    # 优先本地加载，其次按 HF 仓库名加载；无多余回退分支
    if os.path.exists(model_path):
        print(f"Loading local model from '{model_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map='auto' if device == 'cuda' else None,
        )
        tokenizer, model = _prepare_tokenizer_model(tokenizer, model)
        print("Model loaded successfully.")
        return tokenizer, model, device
    else:
        print(f"Loading HF model '{model_dir_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir_name,
            torch_dtype=dtype,
            device_map='auto' if device == 'cuda' else None,
        )
        tokenizer, model = _prepare_tokenizer_model(tokenizer, model)
        print("HF model loaded successfully.")
        return tokenizer, model, device


def _compute_uniform_token_length(tokenizer, texts, fast_mode=False) -> int:
    """
    Compute the uniform target token length for a list of texts:
    - Tokenize without padding/truncation to get lengths.
    - Clamp to tokenizer.model_max_length.
    - In fast_mode, cap to 256 to speed up.
    - Align to multiple of 8 for performance.
    """
    model_cap = getattr(tokenizer, 'model_max_length', 512) or 512
    try:
        token_lengths = [len(tokenizer(t, padding=False, truncation=False)['input_ids']) for t in texts]
        max_len = max(token_lengths) if token_lengths else 1
    except Exception:
        max_len = model_cap
    target_len = min(max_len, model_cap)
    if fast_mode:
        target_len = min(target_len, 256)
    if target_len % 8 != 0:
        target_len += (8 - target_len % 8)
    return target_len


def get_all_activations(model, tokenizer, text, device, max_length=None):
    """
    Extract hidden states for a single text with uniform padding to max_length.
    Returns a list of layer activations (excluding embedding layer), each shape (seq_len, hidden_size).
    """
    effective_max = max_length or getattr(tokenizer, 'model_max_length', 512) or 512
    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=effective_max,
        )
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # list length: 1 + num_layers
        # Skip embedding output (index 0) to align with num_layers
        layer_activations = [hs.squeeze(0).cpu().numpy() for hs in hidden_states[1:]]
    return layer_activations


def _read_corpus_texts(corpus_path):
    """
    Read dataset JSON and normalize to {category: [texts]} for activation extraction.
    - For list of dicts with 'paragraphs', join by '\n'.
    - For dict of categories, each value may be list of dicts with 'paragraphs'.
    """
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, dict):
                txt = item.get('paragraphs') or item.get('content') or ''
                if isinstance(txt, list):
                    txt = '\n'.join(map(str, txt))
                texts.append(str(txt))
            else:
                texts.append(str(item))
        items_by_category = {corpus_name: texts}
    elif isinstance(data, dict):
        items_by_category = {}
        for category, items in data.items():
            cat_texts = []
            for item in items:
                if isinstance(item, dict):
                    txt = item.get('paragraphs') or item.get('content') or ''
                    if isinstance(txt, list):
                        txt = '\n'.join(map(str, txt))
                    cat_texts.append(str(txt))
                else:
                    cat_texts.append(str(item))
            items_by_category[category] = cat_texts
    else:
        raise ValueError(f"Unknown corpus format in {corpus_path}")

    return corpus_name, items_by_category


def process_corpus(model, tokenizer, corpus_path, num_layers, device, fast_mode=False):
    """
    Processes a corpus file, extracts activations, and saves them to JSON files.
    - Saves under 'output/data/<corpus_name>/<category>/layer_<idx>/sample_<i>.json'.
    - Pads all samples in each category to a uniform token length computed from texts.
    """
    corpus_name, items_by_category = _read_corpus_texts(corpus_path)
    print(f"Starting activation extraction for corpus: {corpus_name}")

    # In fast mode, limit items to speed up
    if fast_mode:
        print("--- FAST MODE: Processing only the first 5 items per category ---")
        for category in items_by_category:
            items_by_category[category] = items_by_category[category][:5]

    base_output_dir = os.path.join('output', 'data', corpus_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # Compute a global uniform token length across the entire dataset
    all_texts = []
    for _cat, _txts in items_by_category.items():
        all_texts.extend(_txts)
    target_len_global = _compute_uniform_token_length(tokenizer, all_texts, fast_mode=fast_mode)
    print(f"  Using uniform token length (global across dataset): {target_len_global}")

    for category, texts in items_by_category.items():
        print(f"  Category: {category}, count={len(texts)}")
        target_len = target_len_global

        category_dir = os.path.join(base_output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for i, text in enumerate(tqdm(texts, desc=f"Category {category}")):
            all_layer_activations = get_all_activations(model, tokenizer, text, device, max_length=target_len)
            if all_layer_activations is None:
                print(f"Skipping item {i} in category {category} due to inference error.")
                continue
            if len(all_layer_activations) != num_layers:
                print(f"Warning: Expected {num_layers} layers, got {len(all_layer_activations)}. Skipping item.")
                continue

            # Save per-layer activations
            for layer_idx, layer_acts in enumerate(all_layer_activations):
                layer_dir = os.path.join(category_dir, f"layer_{layer_idx}")
                os.makedirs(layer_dir, exist_ok=True)
                out_path = os.path.join(layer_dir, f"sample_{i}.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'text_index': i,
                        'activations': layer_acts.tolist()
                    }, f, ensure_ascii=False)

    print("--- Activation extraction complete. ---")
