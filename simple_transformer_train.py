import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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

# Encode the text
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# -----------------------------
# Create sequences
# -----------------------------
seq_len = 8
X = []
Y = []

for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    Y.append(data[i+1:i+seq_len+1])

X = torch.stack(X)
Y = torch.stack(Y)

# -----------------------------
# Dataset and DataLoader
# -----------------------------
batch_size = 32  # reduce if GPU memory is low
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
# Initialize model, loss, optimizer
# -----------------------------
model = SimpleTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # smaller lr
loss_fn = nn.CrossEntropyLoss()

# -----------------------------
# Training loop
# -----------------------------
epochs = 10  # start small to test
print("Training started...")

for epoch in range(epochs):
    epoch_loss = 0
    for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        output = model(xb)
        loss = loss_fn(output.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    avg_loss = epoch_loss / len(dataset)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # Save checkpoint every epoch
    torch.save({
        "model_state_dict": model.state_dict(),
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }, f"transformer_checkpoint_epoch{epoch+1}.pt")

print("Training complete!")
