import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import sentencepiece as spm
from torch.cuda.amp import autocast, GradScaler

# ========================
# Config
# ========================
BATCH_SIZE = 32         # GPU memory safe batch
ACCUM_STEPS = 4         # Gradient accumulation
SEQ_LEN = 256
EPOCHS = 5000
LR = 3e-4

D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4

GENERATE_LEN = 300

TOKENIZER_MODEL = "spm_large.model"
MODEL_FILE = "gpt_subword_large_amp.pt"

TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# Tokenizer
# ========================
if not os.path.exists(TOKENIZER_MODEL):
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.Train(
        input="dataset.txt",
        model_prefix="spm_large",
        vocab_size=10000,
        model_type="bpe",
        character_coverage=1.0
    )

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_MODEL)
vocab_size = sp.GetPieceSize()
print("Tokenizer vocab size:", vocab_size)

# ========================
# Load dataset
# ========================
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
# Model Components
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
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.fc(out)

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
        return self.fc(x)

# ========================
# Initialize
# ========================
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    model.load_state_dict(torch.load(MODEL_FILE))

# ========================
# Training loop with mixed precision + gradient accumulation
# ========================
print("Training with mixed precision and gradient accumulation...")
for step in range(EPOCHS):
    optimizer.zero_grad()
    total_loss = 0
    for acc_step in range(ACCUM_STEPS):
        x, y = get_batch()
        with autocast():
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)) / ACCUM_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}")
        torch.save(model.state_dict(), MODEL_FILE)
