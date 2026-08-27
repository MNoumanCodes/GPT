import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Hyperparameters for the test run
B = 4          # Batch size (independent sequences)
T = 8          # Time steps (sequence / context length)
C = 32         # Embedding dimension (n_embd)
head_size = 16 # Dimensionality of Query, Key, and Value vectors

# 1. Simulate an input embedding tensor X of shape (B, T, C)
x = torch.randn(B, T, C)
print(f"Input tensor x shape: {x.shape} -> (Batch, Time, Channels)\n")

# 2. Define a Single Causal Self-Attention Head
class Head(nn.Module):
    """ One head of causal self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Lower triangular mask buffer (not a trainable parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 1. Compute attention affinities via dot product: (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # Scaled by 1 / sqrt(d_k) to prevent vanishing gradients in softmax
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        # 2. Apply causal mask (tokens can only attend to the past and current positions)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 3. Normalize affinities to probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # 4. Perform weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out, wei

# 3. Instantiate and run through the attention head
attention_head = Head(head_size=head_size, n_embd=C, block_size=T, dropout=0.0)
out, weights = attention_head(x)

print(f"Output tensor shape       : {out.shape} -> (B, T, head_size)")
print(f"Attention weights shape   : {weights.shape} -> (B, T, T)\n")

# 4. Inspect the lower triangular attention weights for the first batch element
print("--- Attention Matrix for Sequence 0 (Affinities sum to 1.0 per row) ---")
print(torch.round(weights[0], decimals=3))