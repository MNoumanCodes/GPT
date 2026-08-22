import torch

# 1. Read the raw text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Extract unique characters and create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Unique characters: {''.join(chars)}\n")

# 3. Create mapping dictionaries between characters and integers
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

# 4. Define encode and decode functions
encode = lambda s: [stoi[c] for c in s]          # string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # list of integers -> string

# Test tokenization with a sample string
sample_str = "Hello World!"
encoded = encode(sample_str)
decoded = decode(encoded)

print(f"Original : {sample_str}")
print(f"Encoded  : {encoded}")
print(f"Decoded  : {decoded}\n")

# 5. Convert the entire dataset into a 1D PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Full data tensor shape: {data.shape}, dtype: {data.dtype}")

# 6. Split into 90% train and 10% validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training data length   : {len(train_data)} tokens")
print(f"Validation data length : {len(val_data)} tokens")