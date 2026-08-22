import torch

# Set random seed for reproducibility
torch.manual_seed(1337)

# 1. Load data and encode
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 2. Define batching hyperparameters
batch_size = 4  # How many independent sequences to process in parallel
block_size = 8  # Maximum context length for predictions

# 3. Data loader function
def get_batch(split):
    # Select train or validation split
    data_split = train_data if split == 'train' else val_data
    # Generate random starting offsets in the text
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    # Stack batch_size chunks into a 2D tensor (B, T)
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

print(f"Inputs  shape (xb): {xb.shape}  -> (batch_size, block_size)")
print(f"Targets shape (yb): {yb.shape}  -> (batch_size, block_size)\n")

print("--- Sample Batch Content (First Batch Element) ---")
print(f"xb[0]: {xb[0].tolist()} -> Text: {decode(xb[0].tolist())!r}")
print(f"yb[0]: {yb[0].tolist()} -> Text: {decode(yb[0].tolist())!r}\n")

# 4. Illustrate how 1 chunk contains multiple prediction examples
print("--- Unpacking 8 individual training examples in sequence 0 ---")
for t in range(block_size):
    context = xb[0, :t+1]
    target = yb[0, t]
    print(f"When input is {context.tolist()} ({decode(context.tolist())!r}) -> Target is {target.item()} ({decode([target.item()])!r})")