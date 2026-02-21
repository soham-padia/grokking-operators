import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.p = p
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, d_model))

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


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


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


def find_checkpoint() -> str:
    default = "runs/addpow_p97_cmax32_tf0.2_wd0.1/checkpoint.pt"
    if os.path.exists(default):
        return default

    candidates = glob.glob("runs/addpow_*/checkpoint.pt")
    if not candidates:
        raise FileNotFoundError("No addpow checkpoint found under runs/addpow_*/checkpoint.pt")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def pow_true_table(p: int, c: int, device: torch.device):
    a = torch.arange(p, device=device)
    b = torch.arange(p, device=device)
    A, B = torch.meshgrid(a, b, indexing="ij")
    base = (A + B) % p
    vals = torch.tensor([pow(i, c, p) for i in range(p)], device=device)
    true = vals[base]
    return A, B, true


@torch.no_grad()
def pred_true_err_for_c(model, p: int, c: int, device: torch.device):
    A, B, true = pow_true_table(p, c, device)
    C = torch.full_like(A, c)
    x = torch.stack([A.reshape(-1), B.reshape(-1), C.reshape(-1)], dim=1)
    pred = model(x).argmax(dim=-1).reshape(p, p)
    err = (pred != true).to(torch.int32)
    return pred.cpu(), true.cpu(), err.cpu()


@torch.no_grad()
def accuracy_by_c(model, p: int, c_max: int, device: torch.device):
    accs = []
    for c in range(c_max + 1):
        _, _, err = pred_true_err_for_c(model, p, c, device)
        acc = 1.0 - float(err.float().mean())
        accs.append(acc)
    return accs


def plot_learning_curves(history, outdir):
    steps = history.get("step", [])
    if not steps:
        return

    plt.figure()
    plt.plot(steps, history.get("loss", []))
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title("Training loss vs step")
    savefig(os.path.join(outdir, "01_loss_curve.png"))

    plt.figure()
    plt.plot(steps, history.get("train_acc", []), label="train")
    plt.plot(steps, history.get("test_acc", []), label="test")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs step")
    plt.legend()
    savefig(os.path.join(outdir, "02_accuracy_curve.png"))


def plot_c_slices(model, p, c_max, outdir, device):
    c_values = sorted(set([0, 1, 2, min(5, c_max), c_max]))
    for c in c_values:
        pred, true, err = pred_true_err_for_c(model, p, c, device)

        plt.figure()
        plt.imshow(pred, aspect="auto")
        plt.title(f"Pred table for c={c}")
        plt.xlabel("b")
        plt.ylabel("a")
        plt.colorbar()
        savefig(os.path.join(outdir, f"03_pred_table_c{c}.png"))

        plt.figure()
        plt.imshow(true, aspect="auto")
        plt.title(f"True table for c={c}")
        plt.xlabel("b")
        plt.ylabel("a")
        plt.colorbar()
        savefig(os.path.join(outdir, f"04_true_table_c{c}.png"))

        plt.figure()
        plt.imshow(err, aspect="auto")
        plt.title(f"Error map for c={c}")
        plt.xlabel("b")
        plt.ylabel("a")
        plt.colorbar()
        savefig(os.path.join(outdir, f"05_error_map_c{c}.png"))


def plot_embedding_geometry(model, outdir, p):
    E = model.tok_emb.weight.detach().to("cpu").float()
    E = E - E.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(E, full_matrices=False)
    z = E @ Vh[:2].T
    x1, x2 = z[:, 0], z[:, 1]

    plt.figure()
    plt.scatter(x1.numpy(), x2.numpy(), s=18)
    for i in range(p):
        if i % 8 == 0:
            plt.text(float(x1[i]), float(x2[i]), str(i), fontsize=7)
    plt.title("Token embeddings PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig(os.path.join(outdir, "06_embed_pca_scatter.png"))


def plot_accuracy_vs_c(model, p, c_max, outdir, device):
    accs = accuracy_by_c(model, p, c_max, device)
    plt.figure()
    plt.plot(range(c_max + 1), accs)
    plt.ylim(0.0, 1.01)
    plt.xlabel("c")
    plt.ylabel("accuracy over all (a,b)")
    plt.title("Accuracy vs exponent c")
    savefig(os.path.join(outdir, "07_accuracy_vs_c.png"))

    with open(os.path.join(outdir, "accuracy_by_c.txt"), "w") as f:
        for c, acc in enumerate(accs):
            f.write(f"c={c}\tacc={acc:.6f}\n")


def main():
    ckpt_path = find_checkpoint()
    outdir = os.path.join(os.path.dirname(ckpt_path), "viz")
    ensure_dir(outdir)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    p = cfg["p"]
    c_max = cfg["c_max"]

    model = TinyTransformer(
        p=p,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    ok = _try_load_state_dict(model, ckpt["state_dict"])
    if not ok:
        raise RuntimeError("Could not load state_dict (compiled/uncompiled mismatch or arch changed).")
    model.eval()

    history = ckpt.get("history", {})
    plot_learning_curves(history, outdir)
    plot_c_slices(model, p, c_max, outdir, device)
    plot_embedding_geometry(model, outdir, p)
    plot_accuracy_vs_c(model, p, c_max, outdir, device)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
