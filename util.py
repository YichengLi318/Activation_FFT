import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

def load_model(model_name='gpt2-large'):
    model_path = os.path.join(os.path.dirname(__file__), 'model', model_name)
    print(f"Loading model from '{model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Please ensure the model files are downloaded and placed in '{model_path}'.")
        return None, None

def get_model_attributes(model):
    num_layers = model.config.n_layer
    hidden_size = model.config.n_embd
    print(f"Model has {num_layers} hidden layers.")
    print(f"Activation vector length (hidden size): {hidden_size}")
    return num_layers, hidden_size

def get_activations(model, tokenizer, text, layer_idx):
    """Extract activations for a given text and layer index."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # The first element of the hidden_states tuple is the word embedding output, so the layer number needs to be +1
    activations = outputs.hidden_states[layer_idx + 1]
    return inputs, activations.squeeze(0)  # Remove batch dimension

def find_target_token_index(tokenizer, inputs, word):
    """Find the index of the target word in the token list."""
    token_ids = inputs['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # Handle potential tokenization differences (with/without leading space)
    possible_tokens = [tokenizer.tokenize(' ' + word)[0], tokenizer.tokenize(word)[0]]
    for i, token in enumerate(tokens):
        if token in possible_tokens:
            return i
            
    print(f"Warning: Could not find token for target word '{word}' in '{tokenizer.decode(token_ids)}'.")
    print(f"Tokenized sentence: {tokens}")
    return None

def perform_dft(activation_vector):
    """Perform DFT on an activation vector."""
    return np.abs(np.fft.fft(activation_vector))

def analyze_frequency_bands(freq, dft_data):
    """Calculate the average magnitude in predefined frequency bands."""
    bands = {
        "low": (0, 0.1),
        "mid-low": (0.1, 0.2),
        "mid": (0.2, 0.3),
        "mid-high": (0.3, 0.4),
        "high": (0.4, 0.5)
    }
    band_magnitudes = {}
    for name, (low_freq, high_freq) in bands.items():
        indices = np.where((freq >= low_freq) & (freq < high_freq))
        if len(indices[0]) > 0:
            band_magnitudes[name] = np.mean(dft_data[indices])
        else:
            band_magnitudes[name] = 0
    return band_magnitudes

def analyze_pair_activations(model, tokenizer, pair, layer_to_extract):
    pair_name = f"{pair['catalog']}_{pair['type']}_{pair['id']}"
    sentence_A, sentence_B = pair["A"], pair["B"]
    target_A, target_B = pair["target_word_A"], pair["target_word_B"]

    inputs_A, activations_A = get_activations(model, tokenizer, sentence_A, layer_to_extract)
    inputs_B, activations_B = get_activations(model, tokenizer, sentence_B, layer_to_extract)

    if activations_A.shape[0] != activations_B.shape[0]:
        print(f"Warning: Token count mismatch for '{pair_name}' at layer {layer_to_extract}. Skipping.")
        return None

    target_idx_A = find_target_token_index(tokenizer, inputs_A, target_A)
    target_idx_B = find_target_token_index(tokenizer, inputs_B, target_B)

    if target_idx_A is None or target_idx_B is None:
        print(f"Warning: Could not find target word for '{pair_name}' at layer {layer_to_extract}. Skipping.")
        return None

    tokens_A = tokenizer.convert_ids_to_tokens(inputs_A['input_ids'][0])
    tokens_B = tokenizer.convert_ids_to_tokens(inputs_B['input_ids'][0])

    dft_all_A = np.abs(np.fft.fft(activations_A.numpy(), axis=1))
    dft_all_B = np.abs(np.fft.fft(activations_B.numpy(), axis=1))
    
    N = dft_all_A.shape[-1]
    freq = np.fft.fftfreq(N, d=1.0)[:N // 2]
    
    # Keep only positive frequencies for all tokens
    dft_all_A_pos = dft_all_A[:, :N // 2]
    dft_all_B_pos = dft_all_B[:, :N // 2]

    pair_result = {
        "pair_name": pair_name,
        "sentence_A": sentence_A,
        "sentence_B": sentence_B,
        "target_word_A": target_A,
        "target_word_B": target_B,
        "target_idx_A": target_idx_A,
        "target_idx_B": target_idx_B,
        "tokens_A": tokens_A,
        "tokens_B": tokens_B,
        "frequencies": freq.tolist(),
        "dft_all_tokens_A": dft_all_A_pos.tolist(),  # All tokens spectrum for sentence A
        "dft_all_tokens_B": dft_all_B_pos.tolist(),  # All tokens spectrum for sentence B
    }
    return pair_result

def create_test_set(data_files, seed=42):
    """创建测试集，从每个类别和类型中随机选择一对"""
    import random
    import json
    
    random.seed(seed)
    all_pairs = []
    for file_path in data_files:
        with open(file_path, 'r') as f:
            all_pairs.extend(json.load(f))

    # Group pairs by catalog and type
    grouped_pairs = {}
    for pair in all_pairs:
        catalog = pair.get('catalog', 'unknown')
        type = pair.get('type', 'unknown')
        if catalog not in grouped_pairs:
            grouped_pairs[catalog] = {}
        if type not in grouped_pairs[catalog]:
            grouped_pairs[catalog][type] = []
        grouped_pairs[catalog][type].append(pair)

    # Sample one pair from each type
    test_set = []
    for catalog in grouped_pairs:
        for type in grouped_pairs[catalog]:
            if grouped_pairs[catalog][type]:
                test_set.append(random.choice(grouped_pairs[catalog][type]))

    # Save the test set
    test_set_path = os.path.join('dataset', 'test_set.json')
    with open(test_set_path, 'w') as f:
        json.dump(test_set, f, indent=4)
    
    return test_set