import torch
import torch.nn as nn

# -----------------------------
# Load trained model & mappings
# -----------------------------
checkpoint = torch.load("transformer_checkpoint_epoch10.pt", map_location="cpu")
char_to_idx = checkpoint["char_to_idx"]
idx_to_char = checkpoint["idx_to_char"]

vocab_size = len(char_to_idx)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model definition (same as training)
# -----------------------------
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, vocab_size)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# -----------------------------
# Load model state
# -----------------------------
model = SimpleTransformer().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # evaluation mode

# -----------------------------
# Text generation function
# -----------------------------
def generate_text(model, start_text, length=200, temperature=1.0):
    """
    model       : trained transformer model
    start_text  : initial prompt string
    length      : number of characters to generate
    temperature : randomness of predictions (higher = more random)
    """
    model.eval()
    generated = list(start_text)
    input_seq = torch.tensor([char_to_idx[c] for c in start_text], dtype=torch.long, device=device)
    input_seq = input_seq.unsqueeze(0)  # add batch dimension

    for _ in range(length):
        if input_seq.size(1) > 8:  # maintain seq_len=8
            input_seq = input_seq[:, -8:]

        with torch.no_grad():
            logits = model(input_seq)
            logits = logits[:, -1, :] / temperature  # focus on last character
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            next_char = idx_to_char[next_idx.item()]
            generated.append(next_char)
            input_seq = torch.cat([input_seq, next_idx.unsqueeze(0)], dim=1)

    return "".join(generated)

# -----------------------------
# Example usage
# -----------------------------
prompt = "Once upon a time"
generated_text = generate_text(model, prompt, length=300, temperature=0.8)
print("=== Generated Text ===")
print(generated_text)
