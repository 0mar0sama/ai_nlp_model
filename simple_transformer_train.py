import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# =====================
# CONFIG
# =====================

SEQ_LEN = 32
BATCH_SIZE = 192   # increased batch size thanks to FP16
EMBED_DIM = 128
HEADS = 4
LAYERS = 3
LR = 3e-4
EPOCHS = 50
CHECKPOINT_PATH = "checkpoint.pt"
DATASET_PATH = "dataset.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =====================
# LOAD DATASET
# =====================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# =====================
# DATASET
# =====================

class TextDataset(Dataset):

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):

        x = self.data[idx:idx+SEQ_LEN]
        y = self.data[idx+1:idx+SEQ_LEN+1]

        return x, y

dataset = TextDataset(data, SEQ_LEN)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)

print("Total sequences:", len(dataset))

# =====================
# MODEL
# =====================

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)

        self.pos_embedding = nn.Embedding(SEQ_LEN, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=HEADS,
            dim_feedforward=EMBED_DIM*4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=LAYERS
        )

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):

        positions = torch.arange(0, x.size(1), device=x.device)

        x = self.embedding(x) + self.pos_embedding(positions)

        x = self.transformer(x)

        x = self.norm(x)

        x = self.fc(x)

        return x


model = TransformerModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

loss_fn = nn.CrossEntropyLoss()

# AMP scaler
scaler = torch.cuda.amp.GradScaler()

start_epoch = 0

# =====================
# LOAD CHECKPOINT
# =====================

if os.path.exists(CHECKPOINT_PATH):

    print("Loading checkpoint...")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"]

    print("Resuming from epoch", start_epoch)

# =====================
# TRAINING LOOP
# =====================

print("Training started...")

for epoch in range(start_epoch, EPOCHS):

    total_loss = 0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, y in progress:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # FP16 forward
        with torch.cuda.amp.autocast():

            output = model(x)

            loss = loss_fn(
                output.view(-1, vocab_size),
                y.view(-1)
            )

        # FP16 backward
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch+1,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }, CHECKPOINT_PATH)

print("Training complete!")
