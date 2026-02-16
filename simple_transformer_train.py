import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import sentencepiece as spm

# ========================
# Config
# ========================

BATCH_SIZE = 64
SEQ_LEN = 256
EPOCHS = 5000
LR = 3e-4

D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 8

GENERATE_LEN = 500

TOKENIZER_MODEL = "spm_large.model"
MODEL_FILE = "gpt_subword_large.pt"

TEMPERATURE = 0.8  # controls randomness
TOP_K = 50         # top-k sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# Prepare SentencePiece tokenizer
# ========================

if not os.path.exists(TOKENIZER_MODEL):
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.Train(
        input="dataset.txt",
        model_prefix="spm_large",
        vocab_size=10000,      # larger subword vocab for bigger dataset
        model_type="bpe",
        character_coverage=1.0
    )

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_MODEL)

vocab_size = sp.GetPieceSize()
print("Tokenizer vocab size:", vocab_size)

# Encode dataset
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(sp.EncodeAsIds(text), dtype=torch.long)
print("Dataset length (tokens):", len(data))

# ========================
# Batch generator
# ========================

def get_batch():
    x_batch, y_batch = [], []
    for _ in range(BATCH_SIZE):
        start = random.randint(0, len(data) - SEQ_LEN - 1)
        x = data[start:start + SEQ_LEN]
        y = data[start + 1:start + SEQ_LEN + 1]
        x_batch.append(x)
        y_batch.append(y)
    return torch.stack(x_batch).to(device), torch.stack(y_batch).to(device)

# ========================
# Positional Encoding
# ========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ========================
# Masked Multi-Head Attention
# ========================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.fc(out)

# ========================
# Feed Forward
# ========================

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x):
        return self.net(x)

# ========================
# Transformer Block
# ========================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# ========================
# GPT Model
# ========================

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL)
        self.blocks = nn.Sequential(*[TransformerBlock(D_MODEL, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.fc = nn.Linear(D_MODEL, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits

# ========================
# Initialize model
# ========================

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    model.load_state_dict(torch.load(MODEL_FILE))

# ========================
# Training loop
# ========================

print("Training...")

for step in range(EPOCHS):
    x, y = get_batch()
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), MODEL_FILE)

# ========================
# Text generation with temperature and top-k
# ========================

def generate(start_text, temperature=TEMPERATURE, top_k=TOP_K):
    model.eval()
    tokens = sp.EncodeAsIds(start_text)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(GENERATE_LEN):
        logits = model(x)
        last = logits[:, -1, :]
        last = last / temperature
        if top_k > 0:
            top_values, top_indices = torch.topk(last, top_k)
            probs = torch.zeros_like(last).scatter_(-1, top_indices, F.softmax(top_values, dim=-1))
        else:
            probs = F.softmax(last, dim=-1)
        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)
        if x.size(1) > SEQ_LEN:
            x = x[:, -SEQ_LEN:]
    return sp.DecodeIds(x.squeeze().tolist())

# ========================
# Generate sample
# ========================

print("\nGenerated text:\n")
print(generate("Once upon a time "))
