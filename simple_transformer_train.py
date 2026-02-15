# ===============================
# Transformer Training on TPU/GPU
# ===============================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------------
# Detect device: GPU or TPU
# -----------------------------
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    use_tpu = True
    print(f"Using TPU device: {device}")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tpu = False
    print(f"Using device: {device}")

# -----------------------------
# Load dataset
# -----------------------------
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# Vocabulary
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

# -----------------------------
# Encode dataset
# -----------------------------
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# -----------------------------
# Create sequences with stride
# -----------------------------
seq_len = 8
stride = 4  # skip every 4 characters to reduce number of sequences

X = []
Y = []

for i in range(0, len(data) - seq_len, stride):
    X.append(data[i:i+seq_len])
    Y.append(data[i+1:i+seq_len+1])

X = torch.stack(X)
Y = torch.stack(Y)

print(f"Number of sequences: {len(X)}")

# -----------------------------
# Dataset & DataLoader
# -----------------------------
batch_size = 128  # increase on TPU for speed
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Transformer Model
# -----------------------------
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.pos_embedding = nn.Embedding(100, 32)  # max seq_len = 100

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.fc = nn.Linear(32, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# -----------------------------
# Initialize model, optimizer, loss
# -----------------------------
model = SimpleTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Mixed precision for TPU
if use_tpu:
    model = model.to(dtype=torch.bfloat16)

# -----------------------------
# Training loop
# -----------------------------
epochs = 10  # adjust as needed
print("Training started...")

for epoch in range(epochs):
    epoch_loss = 0
    for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        if use_tpu:
            xb = xb.to(dtype=torch.bfloat16)

        optimizer.zero_grad()
        output = model(xb)
        loss = loss_fn(output.view(-1, vocab_size), yb.view(-1))
        loss.backward()

        if use_tpu:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }, f"checkpoint_epoch{epoch+1}.pt")

print("Training complete!")
