#!/usr/bin/env python3
"""
Adaptive memory law experiment.

Hypothesis:
    The model's effective memory length is state-dependent and tracks local
    dynamical instability.

This script is intentionally minimal and self-contained. It only uses assets
already in the repo (data/*.pkl and data/tiny_lm.pt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

# Allow running this script directly from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icicl.markov import (
    teacher_projected_markov_probs,
    kl_sweep_and_marginal_improvement,
)
from icicl.models import ChronosTokenizer, load_model


SPLITS = {
    "train_id": "data/traj_train.pkl",
    "test_id": "data/traj_test.pkl",
    "test_ood": "data/traj_test_out.pkl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive memory law experiment")
    p.add_argument("--model-path", type=str, default="data/tiny_lm.pt")
    p.add_argument("--outdir", type=str, default="results/adaptive_memory_law")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1024)

    p.add_argument("--vocab-size", type=int, default=100)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--k-max", type=int, default=48)
    p.add_argument("--n-samples", type=int, default=8000)

    p.add_argument("--div-horizon", type=int, default=5)
    p.add_argument("--div-theiler", type=int, default=20)
    p.add_argument("--div-neighbors", type=int, default=30)

    p.add_argument("--perm-trials", type=int, default=200)
    p.add_argument("--trunc-ks", type=str, default="1,2,4,8,16,32,64,128")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def batched_next_logits(
    model: torch.nn.Module,
    contexts_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    out = []
    for start in range(0, contexts_cpu.shape[0], batch_size):
        stop = min(start + batch_size, contexts_cpu.shape[0])
        x = contexts_cpu[start:stop].to(device, non_blocking=True)
        logits = model(x)[:, -1, :].float().cpu()
        out.append(logits)
    return torch.cat(out, dim=0)


def local_divergence(
    traj: np.ndarray,
    horizon: int,
    theiler: int,
    neighbors: int,
    eps: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(traj, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]

    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= horizon + 2:
        return out

    idx = np.arange(0, n - horizon, dtype=np.int64)
    base = x[idx]

    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, len(base)))
    nn.fit(base)
    _, nbrs = nn.kneighbors(base, return_distance=True)

    partner = np.full(len(idx), -1, dtype=np.int64)
    for i, row in enumerate(nbrs[:, 1:]):
        valid = row[np.abs(row - i) > theiler]
        if valid.size:
            partner[i] = int(valid[0])

    good = partner >= 0
    if not np.any(good):
        return out

    d0 = np.linalg.norm(base[good] - base[partner[good]], axis=1)
    dh = np.linalg.norm(
        x[idx[good] + horizon] - x[idx[partner[good]] + horizon], axis=1
    )
    lam = np.log((dh + eps) / (d0 + eps)) / float(horizon)
    out[idx[good]] = lam
    return out


def build_contexts(
    traj: np.ndarray,
    tokenizer: ChronosTokenizer,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = traj[:, 0]
    tok, _ = tokenizer.encode_series(x, 100, 10)

    # Exclude the final context whose next token would be EOS.
    contexts = np.lib.stride_tricks.sliding_window_view(tok[:-1], context_length)[:-1]
    targets = tok[context_length:-1]

    starts = np.arange(contexts.shape[0], dtype=np.int64)
    end_times = starts + context_length - 1
    return contexts.astype(np.int64), targets.astype(np.int64), end_times


def choose_rows(
    valid_mask: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    idx = np.flatnonzero(valid_mask)
    if idx.size == 0:
        raise ValueError("No valid rows available after masking.")
    if idx.size <= n_samples:
        return idx
    return np.sort(rng.choice(idx, size=n_samples, replace=False))


def summarize_array(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
    }


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)) + 1e-12
    return float(np.sum(rx * ry) / denom)


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    return float((np.sum(np.abs(null) >= abs(observed)) + 1.0) / (len(null) + 1.0))


def evaluate_split(
    split_name: str,
    traj_path: str,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    rng: np.random.Generator,
    tokenizer: ChronosTokenizer,
    trunc_ks: list[int],
) -> tuple[dict, dict]:
    print(f"\n[{split_name}] loading {traj_path}", flush=True)
    traj = np.load(traj_path, allow_pickle=True)

    contexts_all, targets_all, end_times_all = build_contexts(
        traj=traj,
        tokenizer=tokenizer,
        context_length=args.context_length,
    )
    div_all = local_divergence(
        traj=traj,
        horizon=args.div_horizon,
        theiler=args.div_theiler,
        neighbors=args.div_neighbors,
    )

    valid_rows = np.isfinite(div_all[end_times_all])
    row_idx = choose_rows(valid_rows, args.n_samples, rng)

    x_np = contexts_all[row_idx]
    y_np = targets_all[row_idx]
    div_np = div_all[end_times_all[row_idx]]

    x_cpu = torch.from_numpy(x_np).long()
    y_cpu = torch.from_numpy(y_np).long()

    print(f"[{split_name}] contexts={len(x_np)}", flush=True)
    logits_full = batched_next_logits(model, x_cpu, device, args.batch_size)
    logp_full = torch.log_softmax(logits_full, dim=-1)
    p_full = torch.exp(logp_full).numpy()
    nll_full = (-logp_full[torch.arange(len(y_cpu)), y_cpu]).numpy()

    print(f"[{split_name}] computing KL-optimal projected orders", flush=True)
    combs = [
        np.arange(args.context_length - 1, args.context_length - 1 - k, -1)
        for k in range(1, args.k_max + 1)
    ]
    probs_proj = teacher_projected_markov_probs(x_np, p_full, combs)
    all_kl, lk, delta = kl_sweep_and_marginal_improvement(probs_proj, p_full)

    best_idx = np.argmin(all_kl, axis=0)
    best_k = best_idx + 1
    best_kl = all_kl[best_idx, np.arange(len(best_idx))]
    gain = all_kl[0] - best_kl

    rho_best = spearman_rho(best_k, div_np)
    rho_gain = spearman_rho(gain, div_np)

    null_corr_best = np.empty(args.perm_trials, dtype=np.float64)
    null_corr_gain = np.empty(args.perm_trials, dtype=np.float64)
    for i in range(args.perm_trials):
        div_perm = rng.permutation(div_np)
        null_corr_best[i] = spearman_rho(best_k, div_perm)
        null_corr_gain[i] = spearman_rho(gain, div_perm)

    p_best = permutation_pvalue(rho_best, null_corr_best)
    p_gain = permutation_pvalue(rho_gain, null_corr_gain)

    print(f"[{split_name}] computing truncation curves", flush=True)
    trunc_kl = []
    trunc_nll = []
    for k in trunc_ks:
        logits_k = batched_next_logits(model, x_cpu[:, -k:], device, args.batch_size)
        logp_k = torch.log_softmax(logits_k, dim=-1)
        p_k = torch.exp(logp_k).numpy()

        kl = np.sum(p_full * (np.log(p_full + 1e-12) - np.log(p_k + 1e-12)), axis=1)
        nll_k = (-logp_k[torch.arange(len(y_cpu)), y_cpu]).numpy()
        trunc_kl.append(kl)
        trunc_nll.append(nll_k)

    trunc_kl = np.asarray(trunc_kl)
    trunc_nll = np.asarray(trunc_nll)

    metrics = {
        "n_samples": int(len(x_np)),
        "nll_full": summarize_array(nll_full),
        "best_k": summarize_array(best_k.astype(np.float64)),
        "gain": summarize_array(gain),
        "corr_best_k_vs_local_div": {
            "rho": float(rho_best),
            "pvalue": float(p_best),
        },
        "corr_gain_vs_local_div": {
            "rho": float(rho_gain),
            "pvalue": float(p_gain),
        },
        "lk_curve": lk.tolist(),
        "delta_curve": delta.tolist(),
        "truncation_ks": trunc_ks,
        "truncation_mean_kl_to_full": trunc_kl.mean(axis=1).tolist(),
        "truncation_mean_nll_delta": (trunc_nll - nll_full[None, :])
        .mean(axis=1)
        .tolist(),
    }

    arrays = {
        "row_idx": row_idx,
        "local_div": div_np,
        "best_k": best_k,
        "gain": gain,
        "nll_full": nll_full,
        "all_kl": all_kl,
        "lk": lk,
        "delta": delta,
        "null_corr_best": null_corr_best,
        "null_corr_gain": null_corr_gain,
        "trunc_kl": trunc_kl,
        "trunc_nll": trunc_nll,
    }
    return metrics, arrays


def plot_best_k_hist(results: dict, k_max: int, outpath: Path) -> None:
    plt.figure(figsize=(8, 4))
    bins = np.arange(0.5, k_max + 1.5, 1.0).tolist()
    for split_name, split in results.items():
        plt.hist(
            split["best_k"],
            bins=bins,
            alpha=0.4,
            density=True,
            label=split_name,
            edgecolor="none",
        )
    plt.xlim(0.5, k_max + 0.5)
    plt.xlabel("effective order k_eff")
    plt.ylabel("density")
    plt.title("Effective memory order distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_scatter(results: dict, metrics: dict, outpath: Path) -> None:
    keys = list(results.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.8), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, split_name in zip(axes, keys):
        best_k = results[split_name]["best_k"]
        local_div = results[split_name]["local_div"]

        if len(best_k) > 2000:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(best_k), size=2000, replace=False)
            x = best_k[sel]
            y = local_div[sel]
        else:
            x = best_k
            y = local_div

        ax.scatter(x, y, s=8, alpha=0.25)
        rho = metrics[split_name]["corr_best_k_vs_local_div"]["rho"]
        pval = metrics[split_name]["corr_best_k_vs_local_div"]["pvalue"]
        ax.set_title(f"{split_name}\nrho={rho:.3f}, p={pval:.2e}")
        ax.set_xlabel("k_eff")
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("local divergence proxy")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_truncation(metrics: dict, outpath: Path) -> None:
    plt.figure(figsize=(7.2, 4.0))
    for split_name, item in metrics.items():
        ks = np.array(item["truncation_ks"], dtype=np.float64)
        y = np.array(item["truncation_mean_kl_to_full"], dtype=np.float64)
        plt.plot(ks, y, marker="o", linewidth=1.8, label=split_name)
    plt.xscale("log", base=2)
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 128])
    plt.xlabel("retained context length k")
    plt.ylabel("mean KL(p_full || p_k)")
    plt.title("Context truncation curve")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_permutation(results: dict, metrics: dict, outpath: Path) -> None:
    keys = list(results.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, split_name in zip(axes, keys):
        null_corr = results[split_name]["null_corr_best"]
        rho_obs = metrics[split_name]["corr_best_k_vs_local_div"]["rho"]
        pval = metrics[split_name]["corr_best_k_vs_local_div"]["pvalue"]

        ax.hist(null_corr, bins=30, alpha=0.75)
        ax.axvline(rho_obs, color="tab:red", linewidth=2)
        ax.set_title(f"{split_name}\nperm p={pval:.3g}")
        ax.set_xlabel("Spearman rho")
        ax.grid(alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trunc_ks = sorted({int(x) for x in args.trunc_ks.split(",") if x.strip()})
    trunc_ks = [k for k in trunc_ks if 1 <= k <= args.context_length]
    if args.context_length not in trunc_ks:
        trunc_ks.append(args.context_length)
        trunc_ks = sorted(trunc_ks)

    device = torch.device(args.device)
    print(f"loading model from {args.model_path} on {device}", flush=True)
    model = load_model(args.model_path, device=device)

    tokenizer = ChronosTokenizer(args.vocab_size, -3, 3)

    metrics = {}
    arrays = {}

    for i, (split_name, split_path) in enumerate(SPLITS.items()):
        split_rng = np.random.default_rng(args.seed + 1000 * (i + 1))
        split_metrics, split_arrays = evaluate_split(
            split_name=split_name,
            traj_path=split_path,
            model=model,
            device=device,
            args=args,
            rng=split_rng,
            tokenizer=tokenizer,
            trunc_ks=trunc_ks,
        )
        metrics[split_name] = split_metrics
        arrays[split_name] = split_arrays

    comparisons = {}
    if "test_id" in arrays and "test_ood" in arrays:
        best_id = arrays["test_id"]["best_k"].astype(np.float64)
        best_ood = arrays["test_ood"]["best_k"].astype(np.float64)
        comparisons["test_ood_minus_test_id_best_k_mean"] = float(
            best_ood.mean() - best_id.mean()
        )
        comparisons["test_ood_over_test_id_best_k_ratio"] = float(
            best_ood.mean() / (best_id.mean() + 1e-12)
        )

    payload = {
        "config": {
            "model_path": args.model_path,
            "device": str(device),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "k_max": args.k_max,
            "n_samples": args.n_samples,
            "div_horizon": args.div_horizon,
            "div_theiler": args.div_theiler,
            "div_neighbors": args.div_neighbors,
            "perm_trials": args.perm_trials,
            "trunc_ks": trunc_ks,
        },
        "splits": metrics,
        "comparisons": comparisons,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, allow_nan=False)

    np_payload = {}
    for split_name, split in arrays.items():
        for k, v in split.items():
            np_payload[f"{split_name}__{k}"] = v
    np.savez_compressed(outdir / "arrays.npz", **np_payload)

    plot_best_k_hist(arrays, args.k_max, outdir / "fig_keff_hist.png")
    plot_scatter(arrays, metrics, outdir / "fig_keff_vs_local_div.png")
    plot_truncation(metrics, outdir / "fig_truncation_kl.png")
    plot_permutation(arrays, metrics, outdir / "fig_permutation_control.png")

    summary_lines = [
        "Adaptive memory law run complete.",
        f"output_dir={outdir}",
        "",
    ]
    for split_name in SPLITS:
        m = metrics[split_name]
        summary_lines.append(
            (
                f"{split_name}: N={m['n_samples']} | "
                f"mean(k_eff)={m['best_k']['mean']:.3f} | "
                f"rho(k_eff, local_div)={m['corr_best_k_vs_local_div']['rho']:.3f} "
                f"(perm_p={m['corr_best_k_vs_local_div']['pvalue']:.3g})"
            )
        )
    if comparisons:
        summary_lines.append("")
        summary_lines.append(
            "test_ood - test_id mean(k_eff): "
            f"{comparisons['test_ood_minus_test_id_best_k_mean']:.3f}"
        )

    summary_txt = "\n".join(summary_lines)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt + "\n")

    print("\n" + summary_txt, flush=True)


if __name__ == "__main__":
    main()
