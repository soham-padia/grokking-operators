import glob
import math
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


def fft_energy_1d(signal):
    spec = torch.fft.rfft(signal)
    return torch.abs(spec)


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
    default = "runs/add3_p97_tf0.3_wd0.1/checkpoint.pt"
    if os.path.exists(default):
        return default

    candidates = glob.glob("runs/add3_*/checkpoint.pt")
    if not candidates:
        raise FileNotFoundError("No add_three checkpoint found under runs/add3_*/checkpoint.pt")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


@torch.no_grad()
def pred_true_err_for_c(model, p: int, c: int, device: torch.device):
    a = torch.arange(p, device=device)
    b = torch.arange(p, device=device)
    A, B = torch.meshgrid(a, b, indexing="ij")
    C = torch.full_like(A, c)
    x = torch.stack([A.reshape(-1), B.reshape(-1), C.reshape(-1)], dim=1)

    pred = model(x).argmax(dim=-1).reshape(p, p).cpu()
    true = ((A + B + C) % p).cpu()
    err = (pred != true).to(torch.int32)
    return pred, true, err


@torch.no_grad()
def accuracy_by_c(model, p: int, device: torch.device):
    accs = []
    for c in range(p):
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


def plot_c_slices(model, p, outdir, device):
    c_values = sorted(set([0, 1, 2, p // 2, p - 1]))
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


def plot_accuracy_vs_c(model, p, outdir, device):
    accs = accuracy_by_c(model, p, device)
    plt.figure()
    plt.plot(range(p), accs)
    plt.ylim(0.0, 1.01)
    plt.xlabel("c")
    plt.ylabel("accuracy over all (a,b)")
    plt.title("Accuracy vs fixed c")
    savefig(os.path.join(outdir, "07_accuracy_vs_c.png"))

    with open(os.path.join(outdir, "accuracy_by_c.txt"), "w") as f:
        for c, acc in enumerate(accs):
            f.write(f"c={c}\tacc={acc:.6f}\n")


def plot_embedding_fourier(model, outdir):
    E = model.tok_emb.weight.detach().to("cpu").float()
    var = E.var(dim=0)
    top = torch.topk(var, k=min(4, E.shape[1])).indices.tolist()

    for dim in top:
        mag = fft_energy_1d(E[:, dim])
        plt.figure()
        plt.plot(mag.numpy())
        plt.title(f"FFT magnitude of embedding dim {dim}")
        plt.xlabel("frequency (rfft bin)")
        plt.ylabel("magnitude")
        savefig(os.path.join(outdir, f"08_fft_dim_{dim}.png"))

def plot_fourier_circles(model, outdir):
    E = model.tok_emb.weight.detach().to("cpu").float()
    var = E.var(dim=0)
    top = torch.topk(var, k=min(4, E.shape[1])).indices.tolist()

    for dim in top:
        sig = E[:, dim] - E[:, dim].mean()
        coeff = torch.fft.rfft(sig)
        mags = torch.abs(coeff)
        phases = torch.angle(coeff)

        n_keep = min(8, max(0, coeff.shape[0] - 1))
        if n_keep == 0:
            continue
        non_dc = torch.arange(1, coeff.shape[0])
        top_h = non_dc[torch.topk(mags[1:], k=n_keep).indices]

        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.0, linewidth=0.8)

        max_r = float(mags[top_h].max().item()) if top_h.numel() > 0 else 1.0
        lim = max(1e-6, 1.2 * max_r)

        for k in top_h.tolist():
            r = float(mags[k].item())
            th = float(phases[k].item())
            x = r * math.cos(th)
            y = r * math.sin(th)
            ax.add_patch(plt.Circle((0.0, 0.0), r, fill=False, linewidth=0.8, alpha=0.18))
            ax.plot([0.0, x], [0.0, y], linewidth=1.4)
            ax.scatter([x], [y], s=18)
            ax.text(x, y, f" k={k}", fontsize=8)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"Fourier circles (embedding dim {dim})")
        ax.set_xlabel("real")
        ax.set_ylabel("imag")
        savefig(os.path.join(outdir, f"09_fourier_circles_dim_{dim}.png"))


def main():
    ckpt_path = find_checkpoint()
    outdir = os.path.join(os.path.dirname(ckpt_path), "viz")
    ensure_dir(outdir)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    p = cfg["p"]

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
    plot_c_slices(model, p, outdir, device)
    plot_embedding_geometry(model, outdir, p)
    plot_accuracy_vs_c(model, p, outdir, device)
    plot_embedding_fourier(model, outdir)
    plot_fourier_circles(model, outdir)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
