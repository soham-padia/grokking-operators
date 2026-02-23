import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Model definition (must match training)
# ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.p = p
        self.tok_emb = nn.Embedding(p, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 2, d_model))  # seq_len=2

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

    def forward(self, x):  # x: [B,2]
        h = self.tok_emb(x) + self.pos_emb
        h = self.encoder(h)
        h = self.ln(h[:, -1, :])
        return self.head(h)


# ----------------------------
# Helpers
# ----------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@torch.no_grad()
def full_table_predictions(model, p, device):
    # Build all (a,b) pairs
    a = torch.arange(p, device=device)
    b = torch.arange(p, device=device)
    A, B = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)  # [p^2,2]

    logits = model(x)
    pred = logits.argmax(dim=-1).reshape(p, p).cpu()
    true = ((A + B) % p).cpu()
    err = (pred != true).to(torch.int32)
    return pred, true, err


def pca_2d(X):
    # X: [p, d] on CPU float
    X = X - X.mean(dim=0, keepdim=True)
    # SVD for PCA
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Z = X @ Vh[:2].T
    return Z[:, 0], Z[:, 1]


def cosine_sim_matrix(E):
    # E: [p,d] CPU float
    E = E / (E.norm(dim=1, keepdim=True) + 1e-12)
    return (E @ E.T)


def fft_energy_1d(signal):
    # signal: [p] float CPU
    # Return magnitude spectrum (real FFT)
    spec = torch.fft.rfft(signal)
    mag = torch.abs(spec)
    return mag

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
    
@torch.no_grad()
def forward_with_attention(model, x, layer_idx=0):
    """
    Run TinyTransformer forward but capture per-head attention weights
    from encoder layer `layer_idx`.

    Returns:
      logits: [B, p]
      attn_w: [B, n_heads, T, T]  (T=2 here)
    """
    model.eval()

    # Embedding
    h = model.tok_emb(x) + model.pos_emb  # [B,T,D]

    attn_w_captured = None

    # Manually run TransformerEncoder layers
    layers = model.encoder.layers
    for i, layer in enumerate(layers):
        # This matches your training config: norm_first=True, batch_first=True
        if layer.norm_first:
            # Self-attention block
            src2 = layer.norm1(h)
            attn_out, attn_w = layer.self_attn(
                src2, src2, src2,
                need_weights=True,
                average_attn_weights=False,  # keep per-head
            )
            h = h + layer.dropout1(attn_out)

            if i == layer_idx:
                attn_w_captured = attn_w  # [B, heads, T, T]

            # Feedforward block
            src2 = layer.norm2(h)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(src2))))
            h = h + layer.dropout2(ff)

        else:
            # Included for completeness (you aren't using norm_first=False)
            attn_out, attn_w = layer.self_attn(
                h, h, h,
                need_weights=True,
                average_attn_weights=False,
            )
            h = layer.norm1(h + layer.dropout1(attn_out))

            if i == layer_idx:
                attn_w_captured = attn_w

            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(h + layer.dropout2(ff))

    # Your model head
    h = model.ln(h[:, -1, :])      # [B,D] take last token
    logits = model.head(h)         # [B,p]

    if attn_w_captured is None:
        raise ValueError(f"layer_idx={layer_idx} out of range for n_layers={len(layers)}")

    return logits, attn_w_captured

# ----------------------------
# Plots
# ----------------------------
def plot_learning_curves(history, outdir):
    steps = history["step"]
    loss = history["loss"]
    tr = history["train_acc"]
    te = history["test_acc"]

    plt.figure()
    plt.plot(steps, loss)
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title("Training loss vs step")
    savefig(os.path.join(outdir, "01_loss_curve.png"))

    plt.figure()
    plt.plot(steps, tr, label="train")
    plt.plot(steps, te, label="test")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs step (watch for grokking jump)")
    plt.legend()
    savefig(os.path.join(outdir, "02_accuracy_curve.png"))


def plot_function_tables(pred, true, err, outdir, p):
    plt.figure()
    plt.imshow(true, aspect="auto")
    plt.title("Ground truth: (a+b) mod p")
    plt.xlabel("b")
    plt.ylabel("a")
    plt.colorbar()
    savefig(os.path.join(outdir, "03_true_table.png"))

    plt.figure()
    plt.imshow(pred, aspect="auto")
    plt.title("Model prediction table")
    plt.xlabel("b")
    plt.ylabel("a")
    plt.colorbar()
    savefig(os.path.join(outdir, "04_pred_table.png"))

    plt.figure()
    plt.imshow(err, aspect="auto")
    plt.title("Error map (1 = wrong)")
    plt.xlabel("b")
    plt.ylabel("a")
    plt.colorbar()
    savefig(os.path.join(outdir, "05_error_map.png"))

    acc = 1.0 - err.float().mean().item()
    with open(os.path.join(outdir, "table_accuracy.txt"), "w") as f:
        f.write(f"Full-table accuracy: {acc:.6f}\n")
        f.write(f"Chance: {1/p:.6f}\n")


def plot_embedding_geometry(model, outdir, p, device):
    E = model.tok_emb.weight.detach().to("cpu").float()  # [p,d]

    x1, x2 = pca_2d(E)
    plt.figure()
    plt.scatter(x1.numpy(), x2.numpy(), s=18)
    for i in range(p):
        if i % 8 == 0:
            plt.text(float(x1[i]), float(x2[i]), str(i), fontsize=7)
    plt.title("Token embeddings PCA (numbers 0..p-1)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig(os.path.join(outdir, "06_embed_pca_scatter.png"))

    # Plot PC coordinates vs token index (often sinusoidal if DFT-like)
    plt.figure()
    plt.plot(range(p), x1.numpy(), label="PC1")
    plt.plot(range(p), x2.numpy(), label="PC2")
    plt.title("PCA coordinates vs token value")
    plt.xlabel("token value")
    plt.ylabel("coordinate")
    plt.legend()
    savefig(os.path.join(outdir, "07_embed_pca_vs_value.png"))


def plot_embedding_similarity(model, outdir, p):
    E = model.tok_emb.weight.detach().to("cpu").float()
    C = cosine_sim_matrix(E).numpy()

    plt.figure()
    plt.imshow(C, aspect="auto", vmin=-1, vmax=1)
    plt.title("Cosine similarity of token embeddings")
    plt.xlabel("token")
    plt.ylabel("token")
    plt.colorbar()
    savefig(os.path.join(outdir, "08_embed_cosine_similarity.png"))


def plot_embedding_fourier(model, outdir, p):
    E = model.tok_emb.weight.detach().to("cpu").float()  # [p,d]
    # Pick a couple dimensions with highest variance (more signal)
    var = E.var(dim=0)
    top = torch.topk(var, k=min(4, E.shape[1])).indices.tolist()

    for j, dim in enumerate(top):
        sig = E[:, dim]
        mag = fft_energy_1d(sig)
        plt.figure()
        plt.plot(mag.numpy())
        plt.title(f"FFT magnitude of embedding dim {dim} (top-variance)")
        plt.xlabel("frequency (rfft bin)")
        plt.ylabel("magnitude")
        savefig(os.path.join(outdir, f"09_fft_dim_{dim}.png"))

def plot_embedding_circuit_gallery(model, outdir, p):
    E = model.tok_emb.weight.detach().to("cpu").float()  # [p,d]
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
    fig.savefig(os.path.join(outdir, "12_embedding_circuit_gallery.png"), dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def plot_fourier_circles(model, outdir):
    E = model.tok_emb.weight.detach().to("cpu").float()  # [p,d]
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
        savefig(os.path.join(outdir, f"11_fourier_circles_dim_{dim}.png"))

@torch.no_grad()
def get_ffn_activations_for_sweep(model, p, device, layer_idx=0, fixed_b=0, sweep_over="a"):
    """
    Capture FFN hidden activations (after GELU(linear1)) for one encoder layer.
    Sweep one input while fixing the other:
      - sweep_over='a': (a=0..p-1, b=fixed_b)
      - sweep_over='b': (a=fixed_b, b=0..p-1)
    Returns tensor of shape [p, d_ff] from the last token position.
    """
    vals = torch.arange(p, device=device)
    if sweep_over == "a":
        a = vals
        b = torch.full_like(vals, fixed_b)
    else:
        a = torch.full_like(vals, fixed_b)
        b = vals
    x = torch.stack([a, b], dim=1)  # [p,2]

    h = model.tok_emb(x) + model.pos_emb  # [p,2,d_model]
    layers = model.encoder.layers
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"layer_idx out of range: {layer_idx}")

    captured = None
    for i, layer in enumerate(layers):
        if layer.norm_first:
            src2 = layer.norm1(h)
            attn_out, _ = layer.self_attn(
                src2, src2, src2, need_weights=False
            )
            h = h + layer.dropout1(attn_out)

            src2 = layer.norm2(h)
            ff_hidden = layer.activation(layer.linear1(src2))  # [B,T,d_ff]
            ff = layer.linear2(layer.dropout(ff_hidden))
            h = h + layer.dropout2(ff)
        else:
            attn_out, _ = layer.self_attn(
                h, h, h, need_weights=False
            )
            h = layer.norm1(h + layer.dropout1(attn_out))
            ff_hidden = layer.activation(layer.linear1(h))
            ff = layer.linear2(layer.dropout(ff_hidden))
            h = layer.norm2(h + layer.dropout2(ff))

        if i == layer_idx:
            captured = ff_hidden[:, -1, :].detach().to("cpu").float()  # [p,d_ff]
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


def _select_neurons_by_fourier_peak(acts, n_neurons):
    # acts: [p, d_ff]
    sig = acts - acts.mean(dim=0, keepdim=True)
    mags = torch.abs(torch.fft.rfft(sig, dim=0))  # [freq, d_ff]
    if mags.shape[0] > 1:
        mags = mags[1:]  # drop DC
    scores = mags.max(dim=0).values
    return torch.topk(scores, k=min(n_neurons, acts.shape[1])).indices.tolist()


def plot_mlp_traces_and_pairgrid(model, outdir, p, device, layer_idx=0, fixed_b=0, n_neurons=7):
    acts = get_ffn_activations_for_sweep(
        model=model,
        p=p,
        device=device,
        layer_idx=layer_idx,
        fixed_b=fixed_b,
        sweep_over="a",
    )  # [p,d_ff]

    neuron_ids = _select_neurons_by_fourier_peak(acts, n_neurons=n_neurons)
    t = torch.arange(p).numpy()
    colors = torch.linspace(0.0, 1.0, p).numpy()
    cmap = plt.get_cmap("viridis")

    # 1) Vertical neuron traces like the reference style
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
    fig.suptitle(
        f"MLP traces (Fourier-selected): sweep a=0..{p-1}, b={fixed_b}, layer={layer_idx}",
        color="white",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(os.path.join(outdir, "13_mlp_traces_layer0_b0.png"), dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2) 7x7 pairwise scatter grid
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
    fig.suptitle(
        "7x7 neuron pair phase portraits (Fourier-selected, same sweep coloring)",
        color="white",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(os.path.join(outdir, "14_mlp_pairgrid_layer0_b0.png"), dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---- Attention visualization (one layer only; good enough for p=97 demo)
def plot_attention_examples(model, outdir, device, examples, layer_idx=0):
    """
    Plots per-head attention weights from encoder layer `layer_idx`
    for a few (a,b) examples.
    """
    n_layers = len(model.encoder.layers)
    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"layer_idx must be in [0, {n_layers-1}], got {layer_idx}")

    for k, (a, b) in enumerate(examples):
        x = torch.tensor([[a, b]], dtype=torch.long, device=device)  # [1,2]

        _, attn_w = forward_with_attention(model, x, layer_idx=layer_idx)
        # attn_w: [B, heads, T, T]
        aw = attn_w[0].detach().to("cpu").float()  # [heads,2,2]

        n_heads = aw.shape[0]
        plt.figure(figsize=(2.5 * n_heads, 2.5))
        for hi in range(n_heads):
            plt.subplot(1, n_heads, hi + 1)
            plt.imshow(aw[hi].numpy(), vmin=0, vmax=1)
            plt.title(f"layer {layer_idx} head {hi}")
            plt.xticks([0, 1], ["a", "b"])
            plt.yticks([0, 1], ["a", "b"])
        plt.suptitle(f"Attention weights for input (a={a}, b={b})")
        savefig(os.path.join(outdir, f"10_attn_L{layer_idx}_ex{k}_a{a}_b{b}.png"))


def main():
    ckpt_path = "runs/addition_p97/checkpoint.pt"
    outdir = "runs/addition_p97/viz"
    ensure_dir(outdir)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    history = ckpt.get("history", None)

    print("Loaded checkpoint config:", cfg)
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

    # 1) Learning curves
    if history is not None and len(history.get("step", [])) > 0:
        plot_learning_curves(history, outdir)

    # 2) Full table predictions + error map
    pred, true, err = full_table_predictions(model, p, device)
    plot_function_tables(pred, true, err, outdir, p)

    # 3) Embedding PCA geometry
    plot_embedding_geometry(model, outdir, p, device)

    # 4) Embedding cosine similarity matrix
    plot_embedding_similarity(model, outdir, p)

    # 5) FFT / frequency analysis on embedding dims
    plot_embedding_fourier(model, outdir, p)
    plot_fourier_circles(model, outdir)
    plot_embedding_circuit_gallery(model, outdir, p)
    plot_mlp_traces_and_pairgrid(
        model,
        outdir,
        p,
        device,
        layer_idx=len(model.encoder.layers) - 1,
        fixed_b=0,
        n_neurons=7,
    )

    # 6) Attention examples (works best if n_layers=1)
    examples = [(0, 0), (1, 2), (10, 20), (40, 70), (96, 96)]
    plot_attention_examples(model, outdir, device, examples, layer_idx=0)
    plot_attention_examples(model, outdir, device, examples, layer_idx=len(model.encoder.layers)-1)

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
