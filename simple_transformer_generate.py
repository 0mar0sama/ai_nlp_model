import torch
import torch.nn as nn

SEQ_LEN = 32
EMBED_DIM = 128
HEADS = 4
LAYERS = 3
CHECKPOINT_PATH = "checkpoint.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]

vocab_size = len(char_to_idx)

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
model.load_state_dict(checkpoint["model"])
model.eval()

# =====================
# GENERATE FUNCTION
# =====================

def generate(prompt, length=500, temperature=0.7):

    input_seq = torch.tensor(
        [char_to_idx[c] for c in prompt],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    output_text = prompt

    for _ in range(length):

        input_cut = input_seq[:, -SEQ_LEN:]

        with torch.no_grad():

            logits = model(input_cut)

            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)

            next_char = torch.multinomial(probs, 1)

        input_seq = torch.cat([input_seq, next_char], dim=1)

        output_text += idx_to_char[next_char.item()]

    return output_text


# =====================
# INTERACTIVE PROMPT
# =====================

prompt = input("Enter prompt: ")

generated = generate(prompt, length=500, temperature=0.7)

print("\nGenerated:\n")
print(generated)
