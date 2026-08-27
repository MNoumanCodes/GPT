import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Hyperparameters
B = 4          # Batch size
T = 8          # Context length (block_size)
C = 32         # Embedding dimension (n_embd)
num_heads = 4  # Number of attention heads running in parallel
head_size = C // num_heads  # 32 // 4 = 8 dimensions per head
dropout = 0.2

# 1. Single Attention Head
class Head(nn.Module):
    """ One head of causal self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

# 2. Multi-Head Attention (Parallel Heads + Projection)
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)
        ])
        # Linear projection to mix channel information after concatenation
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads along the channel dimension: (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# 3. Position-Wise Feed-Forward Network
class FeedForward(nn.Module):
    """ Simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Standard 4x dimension expansion
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Project back to n_embd
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# 4. Complete Transformer Block
class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation """
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LayerNorm formulation with residual skip connections (x + sublayer(LN(x)))
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Test Block Forward Pass
x = torch.randn(B, T, C)
block = Block(n_embd=C, n_head=num_heads, block_size=T, dropout=dropout)
out = block(x)

print(f"Input shape  : {x.shape} -> (Batch={B}, Time={T}, Embedding_Dim={C})")
print(f"Output shape : {out.shape} -> (Batch={B}, Time={T}, Embedding_Dim={C})")
print("Transformer Block forward pass executed successfully.")