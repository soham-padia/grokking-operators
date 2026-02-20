import os
import math
import torch
import torch.nn as nn
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


# ---- Attention visualization (one layer only; good enough for p=97 demo)
def plot_attention_examples(model, outdir, device, examples):
    """
    We’ll hook the self-attn weights from the TransformerEncoderLayer.
    PyTorch's TransformerEncoderLayer does not expose attn weights by default,
    so we re-run a manual MultiheadAttention pass using the layer's modules.
    This works reliably for 1-layer models (your setup).
    """
    if len(model.encoder.layers) != 1:
        # Keep it simple; for multi-layer we can extend later.
        with open(os.path.join(outdir, "attention_note.txt"), "w") as f:
            f.write("Attention plots implemented for n_layers=1. Reduce layers or ask me to extend.\n")
        return

    layer = model.encoder.layers[0]
    mha = layer.self_attn

    # Token+pos embeddings
    tok_emb = model.tok_emb
    pos_emb = model.pos_emb

    for k, (a, b) in enumerate(examples):
        x = torch.tensor([[a, b]], dtype=torch.long, device=device)  # [1,2]
        h = tok_emb(x) + pos_emb  # [1,2,d]

        # MultiheadAttention expects [B, T, D] if batch_first=True
        attn_out, attn_w = mha(h, h, h, need_weights=True, average_attn_weights=False)
        # attn_w: [B, n_heads, T, T]
        aw = attn_w[0].detach().to("cpu").float()  # [heads,2,2]

        # Plot per-head attention
        n_heads = aw.shape[0]
        plt.figure(figsize=(2.5 * n_heads, 2.5))
        for hi in range(n_heads):
            plt.subplot(1, n_heads, hi + 1)
            plt.imshow(aw[hi].numpy(), vmin=0, vmax=1)
            plt.title(f"head {hi}")
            plt.xticks([0, 1], ["a", "b"])
            plt.yticks([0, 1], ["a", "b"])
        plt.suptitle(f"Attention weights for input (a={a}, b={b})")
        savefig(os.path.join(outdir, f"10_attention_example_{k}_a{a}_b{b}.png"))


def main():
    ckpt_path = "runs/addition_p97/checkpoint.pt"
    outdir = "runs/addition_p97/viz"
    ensure_dir(outdir)

    device = get_device()
    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    model.load_state_dict(ckpt["state_dict"])
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

    # 6) Attention examples (works best if n_layers=1)
    examples = [(0, 0), (1, 2), (10, 20), (40, 70), (96, 96)]
    plot_attention_examples(model, outdir, device, examples)

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
