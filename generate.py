import torch
from model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load the checkpoint
checkpoint = torch.load('model.pt', map_location=device)
chars = checkpoint['chars']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

config = checkpoint['config']

# 2. Rebuild model with saved configuration
model = GPTLanguageModel(
    vocab_size=checkpoint['vocab_size'],
    n_embd=config['n_embd'],
    n_head=config['n_head'],
    n_layer=config['n_layer'],
    block_size=config['block_size'],
    dropout=config['dropout']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Choose a prompt or start with a newline
prompt = "ROMEO:\n"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

print(f"--- Generating Text (Starting with prompt: {prompt!r}) ---\n")
generated = model.generate(context, max_new_tokens=600)[0].tolist()
print(decode(generated))