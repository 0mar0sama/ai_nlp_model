import torch
import torch.nn as nn
from tqdm import tqdm

checkpoint = torch.load("transformer_model.pt")

char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]

vocab_size = len(char_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        return self.fc(x)

model = SimpleTransformer().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

prompt = "hello"

input = torch.tensor(
    [[char_to_idx[c] for c in prompt]],
    dtype=torch.long
).to(device)

print("Generating...")

for _ in tqdm(range(30)):

    output = model(input)

    next_char = torch.argmax(output[0, -1]).item()

    input = torch.cat(
        [input, torch.tensor([[next_char]]).to(device)],
        dim=1
    )

result = "".join(idx_to_char[i] for i in input[0].tolist())

print("\nResult:")
print(result)
