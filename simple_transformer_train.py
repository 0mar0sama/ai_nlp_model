import torch
import torch.nn as nn
from tqdm import tqdm

# Load dataset
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}

# Encode
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# Create sequences
seq_len = 8
X = []
Y = []

for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    Y.append(data[i+1:i+seq_len+1])

X = torch.stack(X)
Y = torch.stack(Y)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple Transformer Model
class SimpleTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 32)
        self.pos_embedding = nn.Embedding(100, 32)

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

        positions = torch.arange(x.size(1)).to(device)

        x = self.embedding(x) + self.pos_embedding(positions)

        x = self.transformer(x)

        x = self.fc(x)

        return x


model = SimpleTransformer().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

X = X.to(device)
Y = Y.to(device)

epochs = 200

print("Training started...")

for epoch in range(epochs):

    output = model(X)

    loss = loss_fn(
        output.view(-1, vocab_size),
        Y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save({
    "model": model.state_dict(),
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char
}, "transformer_model.pt")

print("Training complete!")
