import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class Config:
    p: int = 97
    seed: int = 0

    train_frac: float = 0.3

    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 1
    dropout: float = 0.0

    lr: float = 1e-3
    weight_decay: float = 1e-1
    steps: int = 200_000
    eval_every: int = 1000


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path, model, cfg, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "history": history}, path)


def make_addition_splits(p: int, train_frac: float, seed: int):
    pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_train = int(len(pairs) * train_frac)
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    def to_tensors(pairs_list):
        x = torch.tensor(pairs_list, dtype=torch.long)
        y = (x[:, 0] + x[:, 1]) % p
        return x, y

    return to_tensors(train_pairs), to_tensors(test_pairs)


class TinyTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb
        h = self.encoder(h)
        h = self.ln(h[:, -1, :])
        return self.head(h)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    pred = model(x).argmax(dim=-1)
    return (pred == y).float().mean().item()


def main():
    cfg = Config()
    device = get_device()
    print(f"Using device: {device}")

    set_seed(cfg.seed)

    (x_train, y_train), (x_test, y_test) = make_addition_splits(cfg.p, cfg.train_frac, cfg.seed)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = TinyTransformer(cfg.p, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"step": [], "train_acc": [], "test_acc": [], "loss": []}
    ckpt_path = "runs/addition_p97/checkpoint.pt"

    pbar = tqdm(range(1, cfg.steps + 1))
    for step in pbar:
        model.train()
        logits = model(x_train)  # FULL-BATCH
        loss = F.cross_entropy(logits, y_train)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0:
            train_acc = accuracy(model, x_train, y_train)
            test_acc = accuracy(model, x_test, y_test)

            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["loss"].append(loss.item())

            save_checkpoint(ckpt_path, model, cfg, history)
            pbar.set_description(f"step={step} loss={loss.item():.4f} train={train_acc:.3f} test={test_acc:.3f}")
            print(f"[saved] {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
