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

    train_frac: float = 0.3

    d_model: int = 1024
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0

    lr: float = 1e-3
    weight_decay: float = 1e-1
    steps: int = 5_000_000
    eval_every: int = 10000

    # training
    batch_size: int = 2048          # minibatch size (capped at n_train)
    use_amp: bool = True            # mixed precision on CUDA
    use_compile: bool = False       # disabled on HPC: avoids Triton/Inductor compile
    perfect_acc_target: float = 1.0
    perfect_patience_steps: int = 10_000


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


def _try_load_state_dict(model, state_dict):
    """
    Robustly load state_dict whether:
      - checkpoint was saved from compiled model (keys start with '_orig_mod.')
      - checkpoint was saved from normal model
      - current model is compiled or normal
    """
    # 1) direct attempt
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError:
        pass

    # 2) if ckpt keys are uncompiled but model is compiled, add _orig_mod.
    if not any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        sd2 = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
        try:
            model.load_state_dict(sd2)
            return True
        except RuntimeError:
            pass

    # 3) if ckpt keys are compiled but model is uncompiled, strip _orig_mod.
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        sd3 = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(sd3)
            return True
        except RuntimeError:
            pass

    return False


def load_checkpoint(path, model, opt, device):
    ckpt = torch.load(path, map_location=device)

    ok = _try_load_state_dict(model, ckpt["state_dict"])
    if not ok:
        raise RuntimeError(
            "Checkpoint state_dict could not be loaded. "
            "This usually means the model architecture changed "
            "(d_model/n_layers/n_heads/etc.) or keys are incompatible."
        )

    if opt is not None and "opt_state" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt_state"])
        except Exception:
            # Optimizer state can fail to load if you changed params; keep going.
            pass

    history = ckpt.get(
        "history", {"step": [], "train_acc": [], "test_acc": [], "loss": []}
    )
    start_step = history["step"][-1] if history["step"] else 0

    return start_step, history


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

    # AMP setup (new API)
    use_amp = cfg.use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = "runs/addition_p97"
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    done_path = os.path.join(run_dir, "SOLVED_DONE")
    max_steps_path = os.path.join(run_dir, "MAX_STEPS_REACHED")
    history = {"step": [], "train_acc": [], "test_acc": [], "loss": []}
    start_step = 0

    if os.path.exists(done_path):
        print(f"Already solved. Found marker: {done_path}")
        return
    if os.path.exists(max_steps_path):
        print(f"Hard max steps already reached. Found marker: {max_steps_path}")
        return

    # Load checkpoint BEFORE compile to avoid key mismatch
    if os.path.exists(ckpt_path):
        start_step, history = load_checkpoint(ckpt_path, model, opt, device)
        print(f"Resuming from step {start_step}")
    if start_step >= cfg.steps:
        with open(max_steps_path, "w") as f:
            f.write(f"max_steps={cfg.steps}\n")
            f.write(f"last_step={start_step}\n")
        print(f"Reached hard max steps ({cfg.steps}). Wrote {max_steps_path}")
        return
    history.setdefault("consec_perfect_steps", 0)

    # Optional compilation for speed (CUDA only) AFTER loading
    if cfg.use_compile and device.type == "cuda":
        if can_use_torch_compile():
            model = torch.compile(model)
        else:
            print("torch.compile disabled: Python.h not found in this environment.")

    n_train = x_train.size(0)
    batch_size = min(cfg.batch_size, n_train)

    pbar = tqdm(range(start_step + 1, cfg.steps + 1))
    for step in pbar:
        model.train()

        idx = torch.randint(0, n_train, (batch_size,), device=device)
        xb = x_train[idx]
        yb = y_train[idx]

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
            train_acc = accuracy(model, x_train, y_train)
            test_acc = accuracy(model, x_test, y_test)

            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["loss"].append(loss.item())

            if train_acc >= cfg.perfect_acc_target and test_acc >= cfg.perfect_acc_target:
                history["consec_perfect_steps"] += cfg.eval_every
            else:
                history["consec_perfect_steps"] = 0

            save_checkpoint(ckpt_path, model, opt, cfg, history)
            pbar.set_description(
                f"step={step} loss={loss.item():.4f} train={train_acc:.3f} test={test_acc:.3f}"
            )
            print(f"[saved] {ckpt_path}")

            if history["consec_perfect_steps"] >= cfg.perfect_patience_steps:
                with open(done_path, "w") as f:
                    f.write(f"solved_step={step}\n")
                    f.write(f"train_acc={train_acc:.6f}\n")
                    f.write(f"test_acc={test_acc:.6f}\n")
                    f.write(f"consec_perfect_steps={history['consec_perfect_steps']}\n")
                print(f"Solved for >= {cfg.perfect_patience_steps} continuous steps. Wrote {done_path}")
                break

    print("Done.")
    if not os.path.exists(done_path) and history.get("step") and history["step"][-1] >= cfg.steps:
        with open(max_steps_path, "w") as f:
            f.write(f"max_steps={cfg.steps}\n")
            f.write(f"last_step={history['step'][-1]}\n")
        print(f"Reached hard max steps ({cfg.steps}). Wrote {max_steps_path}")


if __name__ == "__main__":
    main()
