from transformers import AutoTokenizer
import os

# Load the tokenizer
model_path = os.path.join(os.path.dirname(__file__), 'model', 'gpt2-large')
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer for gpt2-large loaded from '{model_path}'.\n")

    multi_token_numbers = []
    for i in range(5000):
        number_str = str(i)
        tokens = tokenizer.tokenize(number_str)
        if len(tokens) > 1:
            multi_token_numbers.append((number_str, tokens))

    if multi_token_numbers:
        print("The following numbers up to 5000 are tokenized into multiple tokens:")
        for number_str, tokens in multi_token_numbers:
            print(f"- '{number_str}' is tokenized into: {tokens}")
    else:
        print("All numbers up to 5000 are tokenized as a single token.")

except Exception as e:
    print(f"Failed to load tokenizer: {e}")