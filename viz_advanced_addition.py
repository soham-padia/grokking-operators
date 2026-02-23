import glob
import json
import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VizCfg:
    checkpoint_path: str = "runs/addition_p97/checkpoint.pt"
    outdir_suffix: str = "viz_advanced"
    outdir_base: str = "adv_viz"
    probe_layer: int = -1
    top_neurons: int = 7
    device: str = "cpu"
    max_ckpts_for_time_series: int = 40
    hessian_power_iters: int = 8
    hessian_batch_size: int = 512


class TinyTransformer(nn.Module):
    def __init__(self, p: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.p = p
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def all_addition_inputs(p: int, device: torch.device):
    a = torch.arange(p, device=device)
    b = torch.arange(p, device=device)
    A, B = torch.meshgrid(a, b, indexing="ij")
    x = torch.stack([A.reshape(-1), B.reshape(-1)], dim=1)
    y = (x[:, 0] + x[:, 1]) % p
    return x, y, A, B


def model_from_ckpt(ckpt, device: torch.device):
    cfg = ckpt["cfg"]
    model = TinyTransformer(
        p=cfg["p"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    ok = _try_load_state_dict(model, ckpt["state_dict"])
    if not ok:
        raise RuntimeError("Could not load checkpoint state_dict.")
    model.eval()
    return model


def discover_checkpoints(base_ckpt_path: str):
    run_dir = os.path.dirname(base_ckpt_path)
    candidates = []
    for pat in [
        os.path.join(run_dir, "checkpoint*.pt"),
        os.path.join(run_dir, "checkpoints", "*.pt"),
        os.path.join(run_dir, "snapshots", "*.pt"),
    ]:
        candidates.extend(glob.glob(pat))
    seen = set()
    ordered = []
    for c in sorted(candidates, key=os.path.getmtime):
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    if base_ckpt_path not in seen and os.path.exists(base_ckpt_path):
        ordered.append(base_ckpt_path)
    return ordered


def ckpt_step(ckpt):
    h = ckpt.get("history", {})
    steps = h.get("step", [])
    return int(steps[-1]) if steps else 0


def pca_2d(X: np.ndarray):
    Xc = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    z = Xc @ vt[:2].T
    return z[:, 0], z[:, 1]


def try_umap_2d(X: np.ndarray):
    try:
        import umap  # type: ignore
    except Exception:
        return None
    reducer = umap.UMAP(n_components=2, random_state=0)
    z = reducer.fit_transform(X)
    return z[:, 0], z[:, 1]


@torch.no_grad()
def full_table_preds(model, p: int, device: torch.device):
    x, y, _, _ = all_addition_inputs(p, device)
    logits = model(x)
    pred = logits.argmax(dim=-1)
    err = (pred != y).float()
    return logits.detach().cpu(), pred.detach().cpu(), err.detach().cpu()


def plot_checkpoint_trajectory(ckpt_paths, device, outdir):
    if len(ckpt_paths) < 2:
        with open(os.path.join(outdir, "checkpoint_trajectory_note.txt"), "w") as f:
            f.write("Need >=2 checkpoints for trajectory PCA/UMAP.\n")
        return
    feats = []
    steps = []
    for pth in ckpt_paths:
        ckpt = torch.load(pth, map_location=device)
        model = model_from_ckpt(ckpt, device)
        p = ckpt["cfg"]["p"]
        _, pred, _ = full_table_preds(model, p, device)
        feats.append(pred.numpy().astype(np.float32))
        steps.append(ckpt_step(ckpt))
    X = np.stack(feats, axis=0)
    x1, x2 = pca_2d(X)
    plt.figure()
    plt.scatter(x1, x2, c=np.arange(len(steps)), cmap="viridis", s=36)
    for i, s in enumerate(steps):
        plt.text(x1[i], x2[i], str(s), fontsize=7)
    plt.title("Checkpoint Trajectory PCA (prediction-table features)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig(os.path.join(outdir, "20_ckpt_traj_pca.png"))

    um = try_umap_2d(X)
    if um is not None:
        u1, u2 = um
        plt.figure()
        plt.scatter(u1, u2, c=np.arange(len(steps)), cmap="viridis", s=36)
        for i, s in enumerate(steps):
            plt.text(u1[i], u2[i], str(s), fontsize=7)
        plt.title("Checkpoint Trajectory UMAP")
        plt.xlabel("U1")
        plt.ylabel("U2")
        savefig(os.path.join(outdir, "21_ckpt_traj_umap.png"))

    with open(os.path.join(outdir, "20_ckpt_traj_points.json"), "w") as f:
        json.dump([{"path": p, "step": int(s), "pca_x": float(px), "pca_y": float(py)}
                   for p, s, px, py in zip(ckpt_paths, steps, x1, x2)], f, indent=2)


def forward_capture(model, x, layer_idx):
    h = model.tok_emb(x) + model.pos_emb
    layer_outs = []
    ffn_hiddens = []
    for layer in model.encoder.layers:
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
        layer_outs.append(h[:, -1, :].detach())
        ffn_hiddens.append(ff_hidden[:, -1, :].detach())
    h_last = model.ln(h[:, -1, :])
    logits = model.head(h_last)
    idx = layer_idx if layer_idx >= 0 else len(layer_outs) + layer_idx
    return logits, layer_outs, ffn_hiddens, idx


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    nx = np.linalg.norm(X.T @ X, ord="fro")
    ny = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (nx * ny + 1e-12))


def linear_cca_score(X, Y, n_components=20):
    """
    Return mean canonical correlation between two representations.
    X: [n, dx], Y: [n, dy]
    """
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    n = Xc.shape[0]
    if n < 2:
        return 0.0
    reg = 1e-4
    Cxx = (Xc.T @ Xc) / (n - 1) + reg * np.eye(Xc.shape[1], dtype=Xc.dtype)
    Cyy = (Yc.T @ Yc) / (n - 1) + reg * np.eye(Yc.shape[1], dtype=Yc.dtype)
    Cxy = (Xc.T @ Yc) / (n - 1)

    ex, Ux = np.linalg.eigh(Cxx)
    ey, Uy = np.linalg.eigh(Cyy)
    ex = np.clip(ex, 1e-12, None)
    ey = np.clip(ey, 1e-12, None)
    inv_sqrt_x = Ux @ np.diag(1.0 / np.sqrt(ex)) @ Ux.T
    inv_sqrt_y = Uy @ np.diag(1.0 / np.sqrt(ey)) @ Uy.T
    T = inv_sqrt_x @ Cxy @ inv_sqrt_y
    s = np.linalg.svd(T, compute_uv=False)
    k = int(min(n_components, s.shape[0]))
    if k <= 0:
        return 0.0
    return float(np.mean(s[:k]))


def plot_rep_similarity(ckpt_paths, device, outdir):
    ckpt0 = torch.load(ckpt_paths[-1], map_location=device)
    model0 = model_from_ckpt(ckpt0, device)
    p = ckpt0["cfg"]["p"]
    x, _, _, _ = all_addition_inputs(p, device)
    _, layer_outs, _, _ = forward_capture(model0, x, layer_idx=-1)
    L = len(layer_outs)
    C = np.zeros((L, L), dtype=np.float32)
    reps = [r.cpu().numpy() for r in layer_outs]
    for i in range(L):
        for j in range(L):
            C[i, j] = linear_cka(reps[i], reps[j])
    plt.figure()
    plt.imshow(C, vmin=0, vmax=1)
    plt.title("Layer-to-layer CKA (final checkpoint)")
    plt.xlabel("layer")
    plt.ylabel("layer")
    plt.colorbar()
    savefig(os.path.join(outdir, "30_layer_cka_final.png"))
    Ccca = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            Ccca[i, j] = linear_cca_score(reps[i], reps[j])
    plt.figure()
    plt.imshow(Ccca, vmin=0, vmax=1)
    plt.title("Layer-to-layer CCA (final checkpoint)")
    plt.xlabel("layer")
    plt.ylabel("layer")
    plt.colorbar()
    savefig(os.path.join(outdir, "30b_layer_cca_final.png"))

    if len(ckpt_paths) < 2:
        return
    reps_t = []
    steps = []
    for pth in ckpt_paths:
        ck = torch.load(pth, map_location=device)
        model = model_from_ckpt(ck, device)
        _, louts, _, _ = forward_capture(model, x, layer_idx=-1)
        reps_t.append(louts[-1].cpu().numpy())
        steps.append(ckpt_step(ck))
    n = len(reps_t)
    Ct = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            Ct[i, j] = linear_cka(reps_t[i], reps_t[j])
    plt.figure()
    plt.imshow(Ct, vmin=0, vmax=1)
    plt.title("CKA over checkpoints (last layer)")
    plt.xlabel("checkpoint idx")
    plt.ylabel("checkpoint idx")
    plt.colorbar()
    savefig(os.path.join(outdir, "31_ckpt_cka_last_layer.png"))
    Ct_cca = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            Ct_cca[i, j] = linear_cca_score(reps_t[i], reps_t[j])
    plt.figure()
    plt.imshow(Ct_cca, vmin=0, vmax=1)
    plt.title("CCA over checkpoints (last layer)")
    plt.xlabel("checkpoint idx")
    plt.ylabel("checkpoint idx")
    plt.colorbar()
    savefig(os.path.join(outdir, "31b_ckpt_cca_last_layer.png"))
    with open(os.path.join(outdir, "31_ckpt_cka_meta.json"), "w") as f:
        json.dump({"steps": [int(s) for s in steps]}, f, indent=2)


def make_error_movie(ckpt_paths, device, outdir):
    if len(ckpt_paths) < 2:
        with open(os.path.join(outdir, "40_error_movie_note.txt"), "w") as f:
            f.write("Need >=2 checkpoints for error manifold movie.\n")
        return
    frames = []
    steps = []
    p = None
    for pth in ckpt_paths:
        ck = torch.load(pth, map_location=device)
        model = model_from_ckpt(ck, device)
        p = ck["cfg"]["p"]
        _, _, err = full_table_preds(model, p, device)
        frames.append(err.reshape(p, p).numpy())
        steps.append(ckpt_step(ck))

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], vmin=0, vmax=1, cmap="magma")
    ttl = ax.set_title(f"step={steps[0]}")
    plt.colorbar(im, ax=ax)

    def _upd(i):
        im.set_data(frames[i])
        ttl.set_text(f"step={steps[i]}")
        return [im, ttl]

    ani = animation.FuncAnimation(fig, _upd, frames=len(frames), interval=300, blit=False)
    gif_path = os.path.join(outdir, "40_error_manifold_movie.gif")
    try:
        ani.save(gif_path, writer=animation.PillowWriter(fps=3))
    except Exception as e:
        with open(os.path.join(outdir, "40_error_movie_note.txt"), "w") as f:
            f.write(f"Could not save GIF: {e}\n")
    plt.close(fig)


def plot_logit_fourier(model, p, device, outdir):
    a = torch.arange(p, device=device)
    b = torch.zeros_like(a)
    x = torch.stack([a, b], dim=1)
    logits = model(x).detach().cpu()  # [p,p]
    spec = torch.fft.rfft(logits - logits.mean(dim=0, keepdim=True), dim=0)  # [f,p]
    mag = torch.abs(spec).T.numpy()  # [class,f]
    plt.figure(figsize=(8, 5))
    plt.imshow(mag, aspect="auto")
    plt.title("Logit Fourier magnitude (path: a=0..p-1, b=0)")
    plt.xlabel("frequency bin")
    plt.ylabel("class")
    plt.colorbar()
    savefig(os.path.join(outdir, "50_logit_fourier_heatmap.png"))


def select_neurons_by_fourier(ffn_acts, k):
    sig = ffn_acts - ffn_acts.mean(dim=0, keepdim=True)
    spec = torch.fft.rfft(sig, dim=0)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    if mag.shape[0] > 1:
        ndc = mag[1:]
        ndc_phase = phase[1:]
        peak_vals, peak_idx = ndc.max(dim=0)
        peak_freq = peak_idx + 1
        purity = peak_vals / (ndc.sum(dim=0) + 1e-12)
        sel = torch.topk(peak_vals, k=min(k, ffn_acts.shape[1])).indices
    else:
        peak_freq = torch.zeros(ffn_acts.shape[1], dtype=torch.long)
        purity = torch.zeros(ffn_acts.shape[1])
        ndc_phase = torch.zeros((1, ffn_acts.shape[1]))
        sel = torch.arange(min(k, ffn_acts.shape[1]))
    peak_phase = ndc_phase[(peak_freq - 1).clamp(min=0), torch.arange(ffn_acts.shape[1])]
    return sel.tolist(), peak_freq.cpu(), peak_phase.cpu(), purity.cpu()


def plot_freq_phase_atlas(model, p, device, outdir, layer_idx):
    a = torch.arange(p, device=device)
    b = torch.zeros_like(a)
    x = torch.stack([a, b], dim=1)
    _, _, ffn_hiddens, li = forward_capture(model, x, layer_idx=layer_idx)
    acts = ffn_hiddens[li].cpu()
    _, freq, phase, purity = select_neurons_by_fourier(acts, k=acts.shape[1])
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(freq.numpy(), phase.numpy(), c=purity.numpy(), s=8, cmap="viridis")
    plt.title(f"Frequency-Phase Neuron Atlas (layer {li})")
    plt.xlabel("dominant frequency bin")
    plt.ylabel("phase (rad)")
    plt.colorbar(sc, label="spectral purity")
    savefig(os.path.join(outdir, "60_freq_phase_atlas.png"))
    np.savetxt(
        os.path.join(outdir, "60_freq_phase_atlas.csv"),
        np.stack([freq.numpy(), phase.numpy(), purity.numpy()], axis=1),
        delimiter=",",
        header="dominant_freq,phase,purity",
        comments="",
    )
    return acts


def segment_intersections(points):
    # points: [N,2] ordered polyline; count crossings for non-adjacent segments
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    n = len(points)
    cnt = 0
    for i in range(n - 1):
        a1, a2 = points[i], points[i + 1]
        for j in range(i + 2, n - 1):
            if j == i + 1:
                continue
            b1, b2 = points[j], points[j + 1]
            inter = ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
            cnt += int(inter)
    return cnt


def plot_pair_topology_scores(acts, outdir):
    sel, _, _, _ = select_neurons_by_fourier(acts, k=7)
    A = acts.numpy()
    rows = []
    n = len(sel)
    circ = np.zeros((n, n))
    lin = np.zeros((n, n))
    inter = np.zeros((n, n))
    for i, ni in enumerate(sel):
        yi = A[:, ni]
        for j, nj in enumerate(sel):
            xj = A[:, nj]
            pts = np.stack([xj, yi], axis=1)
            c = np.cov(pts.T)
            ev = np.linalg.eigvalsh(c)
            ev = np.sort(ev)
            linr = ev[1] / (ev[0] + ev[1] + 1e-12)
            cir = 1.0 - abs(ev[1] - ev[0]) / (ev[1] + ev[0] + 1e-12)
            si = segment_intersections(pts)
            circ[i, j] = cir
            lin[i, j] = linr
            inter[i, j] = si
            rows.append([ni, nj, cir, linr, si])
    plt.figure(figsize=(5, 4))
    plt.imshow(circ, vmin=0, vmax=1)
    plt.title("Pair Circularity Score")
    plt.colorbar()
    savefig(os.path.join(outdir, "70_pair_circularity.png"))
    plt.figure(figsize=(5, 4))
    plt.imshow(lin, vmin=0, vmax=1)
    plt.title("Pair Linearity Score")
    plt.colorbar()
    savefig(os.path.join(outdir, "71_pair_linearity.png"))
    plt.figure(figsize=(5, 4))
    plt.imshow(inter)
    plt.title("Pair Self-intersection Count")
    plt.colorbar()
    savefig(os.path.join(outdir, "72_pair_self_intersections.png"))
    np.savetxt(
        os.path.join(outdir, "70_pair_topology_scores.csv"),
        np.array(rows, dtype=np.float64),
        delimiter=",",
        header="neuron_i,neuron_j,circularity,linearity,self_intersections",
        comments="",
    )


def top_hessian_eig(model, x, y, iters=8):
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gflat = torch.cat([g.reshape(-1) for g in grads])
    v = torch.randn_like(gflat)
    v = v / (v.norm() + 1e-12)
    lam = 0.0
    for _ in range(iters):
        gv = torch.dot(gflat, v)
        hv = torch.autograd.grad(gv, params, retain_graph=True)
        hvflat = torch.cat([h.reshape(-1) for h in hv]).detach()
        lam = float(torch.dot(v, hvflat).item())
        v = hvflat / (hvflat.norm() + 1e-12)
    return lam


def plot_hessian_sharpness(ckpt_paths, device, outdir, batch_size=512, iters=8):
    vals = []
    steps = []
    for pth in ckpt_paths:
        ckpt = torch.load(pth, map_location=device)
        model = model_from_ckpt(ckpt, device)
        p = ckpt["cfg"]["p"]
        x_all, y_all, _, _ = all_addition_inputs(p, device)
        idx = torch.randperm(x_all.size(0), device=device)[: min(batch_size, x_all.size(0))]
        lam = top_hessian_eig(model, x_all[idx], y_all[idx], iters=iters)
        vals.append(lam)
        steps.append(ckpt_step(ckpt))
    plt.figure()
    plt.plot(steps, vals, marker="o")
    plt.xlabel("step")
    plt.ylabel("top Hessian eigenvalue (approx)")
    plt.title("Hessian / Sharpness Trend")
    savefig(os.path.join(outdir, "80_hessian_sharpness_trend.png"))


@torch.no_grad()
def plot_input_path_probing(model, p, device, outdir):
    a = torch.arange(p, device=device)
    paths = {
        "a_0": torch.stack([a, torch.zeros_like(a)], dim=1),
        "a_a": torch.stack([a, a], dim=1),
        "a_pm1_minus_a": torch.stack([a, (p - 1 - a) % p], dim=1),
    }
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    rand = torch.randint(0, p, (p, 2), generator=rng, device=device)
    paths["random_pairs"] = rand

    summary = {}
    plt.figure(figsize=(8, 4))
    for name, x in paths.items():
        logits = model(x)
        pred = logits.argmax(dim=-1)
        y = (x[:, 0] + x[:, 1]) % p
        acc = (pred == y).float().mean().item()
        conf = logits.softmax(dim=-1).max(dim=-1).values.detach().cpu().numpy()
        summary[name] = {"accuracy": float(acc), "mean_conf": float(conf.mean())}
        plt.plot(conf, label=f"{name} (acc={acc:.3f})")
    plt.title("Input Path Probing: confidence along path")
    plt.xlabel("path index")
    plt.ylabel("max softmax confidence")
    plt.legend()
    savefig(os.path.join(outdir, "90_input_path_confidence.png"))
    with open(os.path.join(outdir, "90_input_path_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def persistent_homology_optional(acts, outdir):
    try:
        from ripser import ripser  # type: ignore
    except Exception:
        with open(os.path.join(outdir, "95_persistent_homology_note.txt"), "w") as f:
            f.write("ripser not installed; skipped persistent homology.\n")
        return
    X = acts.numpy()
    if X.shape[0] > 256:
        X = X[:256]
    dgms = ripser(X, maxdim=1)["dgms"]
    plt.figure(figsize=(6, 5))
    for i, dgm in enumerate(dgms):
        if dgm.size == 0:
            continue
        plt.scatter(dgm[:, 0], dgm[:, 1], s=10, label=f"H{i}")
    lim = max(1.0, plt.xlim()[1], plt.ylim()[1])
    plt.plot([0, lim], [0, lim], "k--", linewidth=0.8)
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.title("Persistent Homology Diagram (activation cloud)")
    plt.xlabel("birth")
    plt.ylabel("death")
    plt.legend()
    savefig(os.path.join(outdir, "95_persistent_homology.png"))


@torch.no_grad()
def causal_patch_neurons(model, p, device, outdir, layer_idx):
    src = torch.tensor([[1, 2]], dtype=torch.long, device=device)
    tgt = torch.tensor([[40, 70]], dtype=torch.long, device=device)
    y_tgt = int((40 + 70) % p)

    _, _, ffn_src, li = forward_capture(model, src, layer_idx)
    src_vals = ffn_src[li][0]

    x_sweep = torch.arange(p, device=device)
    x = torch.stack([x_sweep, torch.zeros_like(x_sweep)], dim=1)
    _, _, ffn_path, _ = forward_capture(model, x, layer_idx=li)
    sel, _, _, _ = select_neurons_by_fourier(ffn_path[li].cpu(), k=12)

    def forward_with_patch(inp, neuron_idx, patch_val):
        h = model.tok_emb(inp) + model.pos_emb
        for i, layer in enumerate(model.encoder.layers):
            if layer.norm_first:
                src2 = layer.norm1(h)
                attn_out, _ = layer.self_attn(src2, src2, src2, need_weights=False)
                h = h + layer.dropout1(attn_out)
                src2 = layer.norm2(h)
                ff_hidden = layer.activation(layer.linear1(src2))
                if i == li:
                    ff_hidden[:, -1, neuron_idx] = patch_val
                ff = layer.linear2(layer.dropout(ff_hidden))
                h = h + layer.dropout2(ff)
            else:
                attn_out, _ = layer.self_attn(h, h, h, need_weights=False)
                h = layer.norm1(h + layer.dropout1(attn_out))
                ff_hidden = layer.activation(layer.linear1(h))
                if i == li:
                    ff_hidden[:, -1, neuron_idx] = patch_val
                ff = layer.linear2(layer.dropout(ff_hidden))
                h = layer.norm2(h + layer.dropout2(ff))
        h = model.ln(h[:, -1, :])
        return model.head(h)

    base = model(tgt).softmax(dim=-1)[0, y_tgt].item()
    deltas = []
    for n in sel:
        patched = forward_with_patch(tgt, n, src_vals[n]).softmax(dim=-1)[0, y_tgt].item()
        deltas.append((n, patched - base))
    deltas.sort(key=lambda t: abs(t[1]), reverse=True)
    ns = [d[0] for d in deltas]
    dv = [d[1] for d in deltas]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(ns)), dv)
    plt.xticks(range(len(ns)), [str(n) for n in ns], rotation=45)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Mechanistic Causal Patching (delta target-class prob)")
    plt.xlabel("neuron index")
    plt.ylabel("patched - base probability")
    savefig(os.path.join(outdir, "96_causal_patching_neurons.png"))


def feature_movies_from_one_ckpt(acts, outdir):
    ensure_dir(outdir)
    sig = acts - acts.mean(dim=0, keepdim=True)
    spec = torch.abs(torch.fft.rfft(sig, dim=0))
    peak = spec[1:].max(dim=0).values if spec.shape[0] > 1 else spec.max(dim=0).values
    sel = torch.topk(peak, k=min(20, acts.shape[1])).indices.tolist()
    colors = np.linspace(0.0, 1.0, acts.shape[0])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="black")
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(colors="white")
    scat = ax.scatter([], [], s=18)
    title = ax.set_title("", color="white")
    ax.set_xlim(0, acts.shape[0] - 1)
    y_min = float(acts[:, sel].min().item())
    y_max = float(acts[:, sel].max().item())
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

    t = np.arange(acts.shape[0])
    def _upd(i):
        y = acts[:, sel[i]].numpy()
        offs = np.stack([t, y], axis=1)
        scat.set_offsets(offs)
        scat.set_color(cmap(colors))
        title.set_text(f"Neuron trace movie: neuron {sel[i]}")
        return [scat, title]

    ani = animation.FuncAnimation(fig, _upd, frames=len(sel), interval=300, blit=False)
    try:
        ani.save(os.path.join(outdir, "97_feature_movie_neuron_traces.gif"), writer=animation.PillowWriter(fps=4))
    except Exception as e:
        with open(os.path.join(outdir, "97_feature_movie_note.txt"), "w") as f:
            f.write(f"Could not save feature movie GIF: {e}\n")
    plt.close(fig)


@torch.no_grad()
def spatial_movies_from_one_ckpt(model, p, device, outdir, layer_idx=-1):
    # Full table inputs
    x, _, A, B = all_addition_inputs(p, device)
    logits, _, ffn_hiddens, li = forward_capture(model, x, layer_idx=layer_idx)
    logits = logits.detach().cpu()  # [p^2, p]
    acts = ffn_hiddens[li].detach().cpu()  # [p^2, d_ff]

    # Movie 1: logit spatial maps per class
    max_classes = min(64, p)
    class_ids = np.linspace(0, p - 1, num=max_classes, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="black")
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(colors="white")
    m0 = logits[:, int(class_ids[0])].reshape(p, p).numpy()
    im = ax.imshow(m0, cmap="viridis")
    ttl = ax.set_title(f"logit map class={int(class_ids[0])}", color="white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    ax.set_xlabel("b", color="white")
    ax.set_ylabel("a", color="white")

    def _upd_cls(i):
        cid = int(class_ids[i])
        z = logits[:, cid].reshape(p, p).numpy()
        im.set_data(z)
        im.set_clim(vmin=float(z.min()), vmax=float(z.max()) + 1e-12)
        ttl.set_text(f"logit map class={cid}")
        return [im, ttl]

    ani = animation.FuncAnimation(fig, _upd_cls, frames=len(class_ids), interval=180, blit=False)
    try:
        ani.save(os.path.join(outdir, "98_spatial_movie_logit_classes.gif"), writer=animation.PillowWriter(fps=6))
    except Exception as e:
        with open(os.path.join(outdir, "98_spatial_movie_note.txt"), "w") as f:
            f.write(f"Could not save logit class movie: {e}\n")
    plt.close(fig)

    # Movie 2: FFN spatial activation maps for strongest Fourier neurons on path (a,0)
    a = torch.arange(p, device=device)
    path = torch.stack([a, torch.zeros_like(a)], dim=1)
    _, _, path_ffn, _ = forward_capture(model, path, layer_idx=li)
    path_acts = path_ffn[li].detach().cpu()
    sel, _, _, _ = select_neurons_by_fourier(path_acts, k=min(20, path_acts.shape[1]))

    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor="black")
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(colors="white")
    z0 = acts[:, sel[0]].reshape(p, p).numpy()
    im = ax.imshow(z0, cmap="magma")
    ttl = ax.set_title(f"FFN map neuron={sel[0]} layer={li}", color="white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    ax.set_xlabel("b", color="white")
    ax.set_ylabel("a", color="white")

    def _upd_neu(i):
        nid = int(sel[i])
        z = acts[:, nid].reshape(p, p).numpy()
        im.set_data(z)
        im.set_clim(vmin=float(z.min()), vmax=float(z.max()) + 1e-12)
        ttl.set_text(f"FFN map neuron={nid} layer={li}")
        return [im, ttl]

    ani = animation.FuncAnimation(fig, _upd_neu, frames=len(sel), interval=240, blit=False)
    try:
        ani.save(os.path.join(outdir, "99_spatial_movie_ffn_neurons.gif"), writer=animation.PillowWriter(fps=4))
    except Exception as e:
        with open(os.path.join(outdir, "99_spatial_movie_note.txt"), "w") as f:
            f.write(f"Could not save FFN spatial movie: {e}\n")
    plt.close(fig)


def main():
    cfg = VizCfg()
    device = torch.device(cfg.device)
    ckpt_paths_all = discover_checkpoints(cfg.checkpoint_path)
    if not ckpt_paths_all:
        raise FileNotFoundError(f"No checkpoints found from base path: {cfg.checkpoint_path}")
    ckpt_paths = ckpt_paths_all[-cfg.max_ckpts_for_time_series:]

    base_ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model = model_from_ckpt(base_ckpt, device)
    p = base_ckpt["cfg"]["p"]
    run_dir = os.path.dirname(cfg.checkpoint_path)
    run_name = os.path.basename(run_dir.rstrip(os.sep)) or "run"
    outdir = os.path.join(cfg.outdir_base, run_name, cfg.outdir_suffix)
    ensure_dir(outdir)

    plot_checkpoint_trajectory(ckpt_paths, device, outdir)
    plot_rep_similarity(ckpt_paths, device, outdir)
    make_error_movie(ckpt_paths, device, outdir)
    plot_logit_fourier(model, p, device, outdir)
    acts = plot_freq_phase_atlas(model, p, device, outdir, layer_idx=cfg.probe_layer)
    plot_pair_topology_scores(acts, outdir)
    plot_hessian_sharpness(
        ckpt_paths,
        device,
        outdir,
        batch_size=cfg.hessian_batch_size,
        iters=cfg.hessian_power_iters,
    )
    plot_input_path_probing(model, p, device, outdir)
    persistent_homology_optional(acts, outdir)
    causal_patch_neurons(model, p, device, outdir, layer_idx=cfg.probe_layer)
    feature_movies_from_one_ckpt(acts, outdir)
    spatial_movies_from_one_ckpt(model, p, device, outdir, layer_idx=cfg.probe_layer)

    with open(os.path.join(outdir, "00_manifest.json"), "w") as f:
        json.dump(
            {
                "checkpoint_base": cfg.checkpoint_path,
                "n_checkpoints_used": len(ckpt_paths),
                "checkpoints_used": ckpt_paths,
                "output_dir": outdir,
            },
            f,
            indent=2,
        )
    print(f"Saved advanced visualizations to: {outdir}")


if __name__ == "__main__":
    main()
