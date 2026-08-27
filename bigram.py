import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# 1. Load data and setup vocab
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
train_data, val_data = data[:n], data[n:]

batch_size = 32
block_size = 8

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

# 2. Bigram Language Model Definition
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # A lookup table where each token directly maps to prediction scores (logits) for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # Shape: (B, T, C) where C = vocab_size

        if targets is None:
            loss = None
        else:
            # PyTorch F.cross_entropy expects (B*T, C) for logits and (B*T) for targets
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 1. Get predictions from current context
            logits, _ = self(idx)
            # 2. Focus only on the last time step: (B, T, C) -> (B, C)
            logits = logits[:, -1, :]
            # 3. Convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 4. Sample 1 token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 5. Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# 3. Instantiate model and evaluate initial loss
model = BigramLanguageModel(vocab_size)
xb, yb = get_batch('train')
logits, loss = model(xb, yb)

print(f"Untrained Model Loss: {loss.item():.4f}")
print(f"Expected Initial Loss: {-torch.log(torch.tensor(1.0 / vocab_size)).item():.4f}\n")

# 4. Generate text using the untrained model (Starts with a single newline character token 0)
context = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()

print("--- Generated Text (Before Training) ---")
print(decode(generated_tokens))