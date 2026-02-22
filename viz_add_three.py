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

def plot_embedding_circuit_gallery(model, outdir, p):
    E = model.tok_emb.weight.detach().to("cpu").float()
    var = E.var(dim=0)
    n_dims = min(6, E.shape[1])
    dims = torch.topk(var, k=n_dims).indices.tolist()
    X = E[:, dims]
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)

    colors = torch.linspace(0.0, 1.0, p).numpy()
    cmap = plt.get_cmap("viridis")
    k = X.shape[1]

    fig = plt.figure(figsize=(3.0 + 2.1 * k, 1.8 * k), facecolor="black")
    gs = fig.add_gridspec(k, k + 1, width_ratios=[1.4] + [1.0] * k, wspace=0.2, hspace=0.25)

    for i in range(k):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_facecolor("black")
        y = X[:, i].numpy()
        ax.scatter(range(p), y, c=colors, cmap=cmap, s=10, linewidths=0)
        ax.axhline(0.0, color="white", alpha=0.25, linewidth=0.8)
        ax.set_xlim(0, p - 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"dim {dims[i]}", color="white", fontsize=8, pad=2)

    for i in range(k):
        yi = X[:, i].numpy()
        for j in range(k):
            ax = fig.add_subplot(gs[i, j + 1])
            ax.set_facecolor("black")
            xj = X[:, j].numpy()
            ax.scatter(xj, yi, c=colors, cmap=cmap, s=10, linewidths=0)
            ax.axhline(0.0, color="white", alpha=0.25, linewidth=0.8)
            ax.axvline(0.0, color="white", alpha=0.25, linewidth=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Circuit-style embedding phase portraits", color="white", fontsize=12, y=0.995)
    fig.savefig(os.path.join(outdir, "10_embedding_circuit_gallery.png"), dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

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

@torch.no_grad()
def get_ffn_activations_for_sweep(model, p, device, layer_idx=0, fixed_b=0, fixed_c=0, sweep_over="a"):
    vals = torch.arange(p, device=device)
    if sweep_over == "a":
        a = vals
        b = torch.full_like(vals, fixed_b)
        c = torch.full_like(vals, fixed_c)
    elif sweep_over == "b":
        a = torch.full_like(vals, fixed_b)
        b = vals
        c = torch.full_like(vals, fixed_c)
    else:
        a = torch.full_like(vals, fixed_b)
        b = torch.full_like(vals, fixed_c)
        c = vals
    x = torch.stack([a, b, c], dim=1)

    h = model.tok_emb(x) + model.pos_emb
    layers = model.encoder.layers
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"layer_idx out of range: {layer_idx}")

    captured = None
    for i, layer in enumerate(layers):
        if layer.norm_first:
            src2 = layer.norm1(h)
            attn_out, _ = layer.self_attn(src2, src2, src2, need_weights=False)
            h = h + layer.dropout1(attn_out)
            src2 = layer.norm2(h)
            ff_hidden = layer.activation(layer.linear1(src2))
            ff = layer.linear2(layer.dropout(ff_hidden))
            h = h + layer.dropout2(ff)
        else:
            attn_out, _ = layer.self_attn(h, h, h, need_weights=False)
            h = layer.norm1(h + layer.dropout1(attn_out))
            ff_hidden = layer.activation(layer.linear1(h))
            ff = layer.linear2(layer.dropout(ff_hidden))
            h = layer.norm2(h + layer.dropout2(ff))

        if i == layer_idx:
            captured = ff_hidden[:, -1, :].detach().to("cpu").float()
            break

    if captured is None:
        raise RuntimeError("Failed to capture FFN activations.")
    return captured


def _style_dark_axis(ax):
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.25))
    ax.grid(False)


def plot_mlp_traces_and_pairgrid(model, outdir, p, device, layer_idx=0, fixed_b=0, fixed_c=0, n_neurons=7):
    acts = get_ffn_activations_for_sweep(
        model=model,
        p=p,
        device=device,
        layer_idx=layer_idx,
        fixed_b=fixed_b,
        fixed_c=fixed_c,
        sweep_over="a",
    )

    var = acts.var(dim=0)
    neuron_ids = torch.topk(var, k=min(n_neurons, acts.shape[1])).indices.tolist()
    t = torch.arange(p).numpy()
    colors = torch.linspace(0.0, 1.0, p).numpy()
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(7.2, 1.8 * len(neuron_ids)), facecolor="black")
    if len(neuron_ids) == 1:
        axes = [axes]
    for ax, nid in zip(axes, neuron_ids):
        y = acts[:, nid].numpy()
        _style_dark_axis(ax)
        ax.scatter(t, y, c=colors, cmap=cmap, s=16, linewidths=0)
        ax.axhline(0.0, color=(1.0, 1.0, 1.0, 0.4), linewidth=0.9)
        ax.set_xlim(0, p - 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Layer {layer_idx} FFN neuron {nid}", color="white", fontsize=8, pad=2)
    fig.suptitle(f"MLP traces: sweep a=0..{p-1}, b={fixed_b}, c={fixed_c}", color="white", fontsize=12, y=0.995)
    fig.savefig(os.path.join(outdir, "11_mlp_traces_layer0_b0_c0.png"), dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    n = len(neuron_ids)
    fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n), facecolor="black")
    for i in range(n):
        yi = acts[:, neuron_ids[i]].numpy()
        for j in range(n):
            xj = acts[:, neuron_ids[j]].numpy()
            ax = axes[i, j]
            _style_dark_axis(ax)
            ax.scatter(xj, yi, c=colors, cmap=cmap, s=12, linewidths=0)
            ax.axhline(0.0, color=(1.0, 1.0, 1.0, 0.3), linewidth=0.8)
            ax.axvline(0.0, color=(1.0, 1.0, 1.0, 0.3), linewidth=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(str(neuron_ids[j]), color="white", fontsize=7)
            if j == 0:
                ax.set_ylabel(str(neuron_ids[i]), color="white", fontsize=7)
    fig.suptitle("7x7 neuron pair phase portraits (same sweep coloring)", color="white", fontsize=12, y=0.995)
    fig.savefig(os.path.join(outdir, "12_mlp_pairgrid_layer0_b0_c0.png"), dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


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
    plot_embedding_circuit_gallery(model, outdir, p)
    plot_mlp_traces_and_pairgrid(model, outdir, p, device, layer_idx=0, fixed_b=0, fixed_c=0, n_neurons=7)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
