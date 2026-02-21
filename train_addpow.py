import os
import random
import sysconfig
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class Config:
    p: int = 97
    seed: int = 0

    # data
    train_frac: float = 0.2
    batch_size: int = 2048
    c_max: int = 32        # start with 32; once it works, switch to 96

    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.0

    # opt
    lr: float = 3e-4
    weight_decay: float = 1e-1
    steps: int = 200_000
    eval_every: int = 2000
    eval_batches: int = 50  # accuracy estimate batches

    # training speed
    use_amp: bool = True
    use_compile: bool = False

def save_checkpoint(path, model, opt, cfg, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "opt_state": opt.state_dict(),
            "cfg": cfg.__dict__,
            "history": history,
        },
        path,
    )

def load_checkpoint(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)

    ok = _try_load_state_dict(model, ckpt["state_dict"])
    if not ok:
        raise RuntimeError("Checkpoint state_dict could not be loaded.")

    if opt is not None and "opt_state" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt_state"])
        except Exception:
            pass

    history = ckpt.get("history", {"step": [], "train_acc": [], "test_acc": [], "loss": []})
    start_step = history["step"][-1] if history.get("step") else 0

    return start_step, history

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def can_use_torch_compile() -> bool:
    py_include = sysconfig.get_paths().get("include")
    if not py_include:
        return False
    return os.path.exists(os.path.join(py_include, "Python.h"))


def _try_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError:
        pass

    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        sd2 = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        try:
            model.load_state_dict(sd2)
            return True
        except RuntimeError:
            pass

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        sd3 = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(sd3)
            return True
        except RuntimeError:
            pass

    return False


# ----------------------------
# Deterministic split for triples (a,b,c)
# ----------------------------
def in_train_split(a: int, b: int, c: int, p: int, train_frac: float, seed: int) -> bool:
    x = a * (p * p) + b * p + c
    x ^= (seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= (x >> 31)
    r = (x & ((1 << 53) - 1)) / float(1 << 53)
    return r < train_frac

def build_addpow_dataset(cfg: Config, device: torch.device):
    """
    Build full dataset for (a+b)^c mod p for a,b in [0..p-1], c in [0..c_max].
    Then deterministically split into train/test using in_train_split.

    Returns:
      (x_train, y_train), (x_test, y_test) on `device`.
    """
    p = cfg.p
    c_max = cfg.c_max

    # All triples (a,b,c)
    a = torch.arange(p, dtype=torch.long)
    b = torch.arange(p, dtype=torch.long)
    c = torch.arange(c_max + 1, dtype=torch.long)
    A, B, C = torch.meshgrid(a, b, c, indexing="ij")  # [p,p,c_max+1]

    x_all = torch.stack([A, B, C], dim=-1).reshape(-1, 3)  # [N,3], N=p*p*(c_max+1)

    # Precompute pow_table[base, c] to avoid calling pow per example repeatedly
    pow_table = torch.empty((p, c_max + 1), dtype=torch.long)
    bases = list(range(p))
    for exp in range(c_max + 1):
        pow_table[:, exp] = torch.tensor([pow(base, exp, p) for base in bases], dtype=torch.long)

    base = (x_all[:, 0] + x_all[:, 1]) % p
    y_all = pow_table[base, x_all[:, 2]]  # [N]

    # Deterministic split mask (once-time Python loop; N<=~1e6 so OK)
    mask = torch.empty(x_all.size(0), dtype=torch.bool)
    for i in range(x_all.size(0)):
        ai = int(x_all[i, 0])
        bi = int(x_all[i, 1])
        ci = int(x_all[i, 2])
        mask[i] = in_train_split(ai, bi, ci, p, cfg.train_frac, cfg.seed)

    x_train = x_all[mask]
    y_train = y_all[mask]
    x_test = x_all[~mask]
    y_test = y_all[~mask]

    # Move once to device
    return (x_train.to(device), y_train.to(device)), (x_test.to(device), y_test.to(device))


def batch_from_tensor_dataset(x, y, batch_size: int, device: torch.device):
    n = x.size(0)
    idx = torch.randint(0, n, (batch_size,), device=device)
    return x[idx], y[idx]


@torch.no_grad()
def accuracy_on_dataset(model: nn.Module, x, y, batch_size: int):
    model.eval()
    correct = 0
    total = 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i+batch_size]
        yb = y[i:i+batch_size]
        pred = model(xb).argmax(dim=-1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total



def sample_batch(cfg: Config, split: str, device: torch.device):
    p = cfg.p
    xs, ys = [], []
    while len(xs) < cfg.batch_size:
        a = random.randrange(p)
        b = random.randrange(p)
        c = random.randrange(cfg.c_max + 1)

        is_train = in_train_split(a, b, c, p, cfg.train_frac, cfg.seed)
        if split == "train" and not is_train:
            continue
        if split == "test" and is_train:
            continue

        base = (a + b) % p
        y = pow(base, c, p)

        xs.append([a, b, c])
        ys.append(y)

    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def estimate_accuracy(cfg: Config, model: nn.Module, split: str, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    for _ in range(cfg.eval_batches):
        x, y = sample_batch(cfg, split, device)
        pred = model(x).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
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

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb
        h = self.encoder(h)
        h = self.ln(h[:, -1, :])
        return self.head(h)

def main():
    cfg = Config()
    device = get_device()
    print(f"Using device: {device}")
    set_seed(cfg.seed)

    run_dir = f"runs/addpow_p{cfg.p}_cmax{cfg.c_max}_tf{cfg.train_frac}_wd{cfg.weight_decay}"
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    os.makedirs(run_dir, exist_ok=True)

    # ---- Build dataset ONCE (fast training afterwards)
    (x_train, y_train), (x_test, y_test) = build_addpow_dataset(cfg, device)
    print(f"Dataset sizes: train={x_train.size(0)} test={x_test.size(0)}")

    model = TinyTransformer(cfg.p, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"step": [], "train_acc": [], "test_acc": [], "loss": []}
    start_step = 0

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    use_amp = cfg.use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if os.path.exists(ckpt_path):
        start_step, history = load_checkpoint(ckpt_path, model, opt, device)
        print(f"Resuming from step {start_step}")

    if cfg.use_compile and device.type == "cuda":
        if can_use_torch_compile():
            model = torch.compile(model)
        else:
            print("torch.compile disabled: Python.h not found in this environment.")

    pbar = tqdm(range(start_step + 1, cfg.steps + 1))
    for step in pbar:
        model.train()
        xb, yb = batch_from_tensor_dataset(x_train, y_train, cfg.batch_size, device)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

        if step % cfg.eval_every == 0:
            # Full-dataset accuracy (chunked); stable and paper-friendly
            train_acc = accuracy_on_dataset(model, x_train, y_train, batch_size=8192)
            test_acc = accuracy_on_dataset(model, x_test, y_test, batch_size=8192)

            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["loss"].append(loss.item())

            save_checkpoint(ckpt_path, model, opt, cfg, history)
            pbar.set_description(
                f"step={step} loss={loss.item():.4f} train={train_acc:.3f} test={test_acc:.3f}"
            )
            print(f"[saved] {ckpt_path}")

            # Optional early stop if fully solved
            if test_acc > 0.999:
                print("Test ~1.0, stopping.")
                break

    print("Done.")


if __name__ == "__main__":
    main()
