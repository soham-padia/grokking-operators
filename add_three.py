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
    batch_size: int = 4096

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    lr: float = 3e-4
    weight_decay: float = 1e-1
    steps: int = 200_000
    eval_every: int = 2000

    # fast eval instead of full pass
    eval_batches: int = 10


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path, model, cfg, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "history": history}, path)


def in_train_split(a: int, b: int, c: int, p: int, train_frac: float, seed: int) -> bool:
    # deterministic hash split like before
    x = a * (p * p) + b * p + c
    x ^= (seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= (x >> 31)
    r = (x & ((1 << 53) - 1)) / float(1 << 53)
    return r < train_frac

def batch_add3(cfg, device, mode: str, c_k: int):
    """
    mode: "train" or "test"
    c_k controls curriculum:
      - c_k = 0  -> c is constant 0
      - c_k = 1  -> c uniform in [0..1]
      - ...
      - c_k = p-1 -> c uniform in [0..p-1]
    """
    p = cfg.p
    B = cfg.batch_size

    a = torch.randint(0, p, (B,), device=device)
    b = torch.randint(0, p, (B,), device=device)
    if c_k == 0:
        c = torch.zeros((B,), dtype=torch.long, device=device)
    else:
        c = torch.randint(0, min(c_k, p-1) + 1, (B,), device=device)

    # deterministic split mask (vectorized over batch via python loop is fine for B~4k)
    keep = torch.empty((B,), dtype=torch.bool, device=device)
    for i in range(B):
        ai, bi, ci = int(a[i]), int(b[i]), int(c[i])
        keep[i] = in_train_split(ai, bi, ci, p, cfg.train_frac, cfg.seed)

    if mode == "train":
        m = keep
    else:
        m = ~keep

    # if mask is too small, resample (rare)
    if m.sum().item() < B // 4:
        return batch_add3(cfg, device, mode, c_k)

    a, b, c = a[m], b[m], c[m]
    # pad back up to B by repeating (ok)
    if a.numel() < B:
        rep = B - a.numel()
        a = torch.cat([a, a[:rep]])
        b = torch.cat([b, b[:rep]])
        c = torch.cat([c, c[:rep]])

    x = torch.stack([a, b, c], dim=1)               # [B,3]
    y = (a + b + c) % p                             # [B]
    return x, y


def build_add3_dataset(cfg: Config, device: torch.device):
    p = cfg.p

    a = torch.arange(p, dtype=torch.long)
    b = torch.arange(p, dtype=torch.long)
    c = torch.arange(p, dtype=torch.long)
    A, B, C = torch.meshgrid(a, b, c, indexing="ij")  # [p,p,p]

    x_all = torch.stack([A, B, C], dim=-1).reshape(-1, 3)  # [N,3]
    y_all = (x_all[:, 0] + x_all[:, 1] + x_all[:, 2]) % p

    mask = torch.empty(x_all.size(0), dtype=torch.bool)
    for i in range(x_all.size(0)):
        ai = int(x_all[i, 0])
        bi = int(x_all[i, 1])
        ci = int(x_all[i, 2])
        mask[i] = in_train_split(ai, bi, ci, p, cfg.train_frac, cfg.seed)

    x_train, y_train = x_all[mask].to(device), y_all[mask].to(device)
    x_test, y_test = x_all[~mask].to(device), y_all[~mask].to(device)

    return (x_train, y_train), (x_test, y_test)


def batch_from_dataset(x, y, batch_size: int, device: torch.device):
    n = x.size(0)
    idx = torch.randint(0, n, (batch_size,), device=device)
    return x[idx], y[idx]


@torch.no_grad()
def sampled_accuracy(model, x, y, batch_size: int, n_batches: int, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    n = x.size(0)
    for _ in range(n_batches):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb, yb = x[idx], y[idx]
        pred = model(xb).argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


class TinyTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, n_layers: int, dropout: float, seq_len: int = 3):
        super().__init__()
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))

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

    def forward(self, x):  # x: [B,3]
        h = self.tok_emb(x) + self.pos_emb
        h = self.encoder(h)
        h = self.ln(h[:, -1, :])
        return self.head(h)


def main():
    cfg = Config()
    device = get_device()
    print(f"Using device: {device}")
    set_seed(cfg.seed)

    run_dir = f"runs/add3_p{cfg.p}_tf{cfg.train_frac}_wd{cfg.weight_decay}"
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    os.makedirs(run_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = build_add3_dataset(cfg, device)
    print(f"Dataset sizes: train={x_train.size(0)} test={x_test.size(0)}")

    model = TinyTransformer(cfg.p, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout, seq_len=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"step": [], "train_acc": [], "test_acc": [], "loss": []}

    pbar = tqdm(range(1, cfg.steps + 1))
    for step in pbar:
        model.train()
        xb, yb = batch_from_dataset(x_train, y_train, cfg.batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0:
            train_acc = sampled_accuracy(model, x_train, y_train, batch_size=4096, n_batches=cfg.eval_batches, device=device)
            test_acc = sampled_accuracy(model, x_test, y_test, batch_size=4096, n_batches=cfg.eval_batches, device=device)

            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["loss"].append(loss.item())

            save_checkpoint(ckpt_path, model, cfg, history)
            pbar.set_description(f"step={step} loss={loss.item():.4f} train={train_acc:.3f} test={test_acc:.3f}")
            print(f"[saved] {ckpt_path}")

            if test_acc > 0.999:
                print("Test ~1.0, stopping.")
                break

    print("Done.")


if __name__ == "__main__":
    main()