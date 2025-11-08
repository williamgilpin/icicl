import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import math, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAD_ID = 0  # reserved
# time-series bins: [1..B]
# EOS token appended at end: B+1

@dataclass
class ChronosTokenizer:
    B: int                        # number of quantization bins
    c_min: float                  # smallest bin center
    c_max: float                  # largest bin center
    eps: float = 1e-8             # for safe division

    def _bin_centers_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform centers c_j in [c_min, c_max]; edges midpoints; outer edges at +/-inf."""
        centers = np.linspace(self.c_min, self.c_max, self.B, dtype=np.float64)
        # inner edges at midpoints; add -inf, +inf
        inner = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate(([-np.inf], inner, [np.inf]))
        return centers, edges

    def mean_scale(self, x: np.ndarray, C: int) -> Tuple[np.ndarray, float]:
        """
        Mean scaling: \tilde{x}_i = x_i / s with s = (1/C) * sum_{i=1}^C |x_i|.
        Preserves zeros. Uses eps to avoid division by 0.
        """
        assert C >= 1 and C <= len(x)
        s = np.mean(np.abs(x[:C])).astype(np.float64)
        s = max(s, self.eps)
        return (x / s).astype(np.float64), s

    def quantize(self, x_scaled: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Map real values to discrete ids in [1..B] using edges.
        Returns (ids, aux) where aux contains centers and edges for dequantization.
        """
        centers, edges = self._bin_centers_edges()
        # np.digitize returns indices in 1..len(edges)-1 for left-closed, right-open bins when right=False
        ids = np.digitize(x_scaled, edges[1:-1], right=False).astype(np.int64) + 1  # -> [1..B]
        return ids, {"centers": centers, "edges": edges}

    def dequantize(self, ids: np.ndarray, centers: Optional[np.ndarray] = None) -> np.ndarray:
        """Map ids in [1..B] back to centers c_j; PAD->np.nan; EOS ignored."""
        if centers is None:
            centers, _ = self._bin_centers_edges()
        y = np.empty_like(ids, dtype=np.float64)
        mask_pad = (ids == PAD_ID)
        mask_eos = (ids == (self.B + 1))
        mask_bins = ~(mask_pad | mask_eos)
        y[mask_pad] = np.nan
        y[mask_eos] = np.nan
        y[mask_bins] = centers[ids[mask_bins] - 1]
        return y

    def encode_series(
        self,
        x: np.ndarray,
        C: int,
        H: int,
        pad_to: Optional[int] = None,
        replace_nans_with_pad: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        x: shape (C+H,) real-valued series.
        C: number of context points
        H: number of future points
        Returns token ids including EOS; PAD used for NaNs and optional right-padding.
        """
        assert len(x) >= C + H
        x = x.astype(np.float64)
        if replace_nans_with_pad and np.isnan(x).any():
            # keep NaNs for scale computation by treating them as 0 contribution (Chronos pads missing with PAD)
            nan_mask = np.isnan(x)
            x_safe = x.copy()
            x_safe[nan_mask] = 0.0
        else:
            x_safe = x

        x_scaled, s = self.mean_scale(x_safe, C=C)
        token_ids, aux = self.quantize(x_scaled)
        eos_id = self.B + 1
        out = np.concatenate([token_ids, np.array([eos_id], dtype=np.int64)], axis=0)

        # replace places where original x was NaN with PAD
        if replace_nans_with_pad and np.isnan(x).any():
            pad_mask = np.isnan(x)
            pad_ids = np.where(pad_mask, PAD_ID, out[:-1])  # exclude EOS during masking
            out = np.concatenate([pad_ids, np.array([eos_id], dtype=np.int64)], axis=0)

        if pad_to is not None and pad_to > len(out):
            pad_len = pad_to - len(out)
            out = np.concatenate([out, np.full(pad_len, PAD_ID, dtype=np.int64)], axis=0)

        meta = {
            "scale_s": float(s),
            "B": int(self.B),
            "c_min": float(self.c_min),
            "c_max": float(self.c_max),
        }
        # store centers for exact inverse later if needed
        self._cached_centers = aux["centers"]
        return out, meta

    def decode_series(self, ids: np.ndarray, scale_s: float) -> np.ndarray:
        """
        Dequantize to centers, then invert scaling: x_hat = \tilde{x} * s.
        PAD/EOS -> NaN.
        """
        tilde = self.dequantize(ids, getattr(self, "_cached_centers", None))
        with np.errstate(invalid="ignore"):
            return tilde * scale_s


import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_autoregressive(
    model,
    context_ids: torch.LongTensor,      # shape (T,) or (B,T)
    max_new_tokens: int,               # H
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
):
    """
    Generate a sequence of tokens from a model autoregressively.

    Args:
        model: A model that takes a tensor of shape (B, T) and returns a tensor of logits shape (B, T, V).
        context_ids: A tensor of shape (T,) or (B, T) containing the initial context.
        max_new_tokens: The number of new tokens to generate.
        temperature: The temperature to use for sampling.
        top_k: The number of top-k tokens to sample from.
        top_p: The top-p value to use for sampling.
        greedy: Whether to sample greedily.
    """
    device = next(model.parameters()).device
    x = context_ids.to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1,T)

    model.eval()
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.block_size:]                 # crop to context window
        logits = model(x_cond)                            # (B,t,V)
        logits = logits[:, -1, :]                         # (B,V)

        if greedy:
            next_id = torch.argmax(logits, dim=-1)        # (B,)
        else:
            # temperature + (optional) top-k/top-p sampling
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

            if top_p is not None:
                # nucleus sampling
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf > top_p
                # keep at least one token
                mask[..., 0] = False
                # set probs outside nucleus to zero, then renormalize
                sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                # sample in sorted space then map back
                next_sorted = torch.multinomial(sorted_probs, num_samples=1)  # (B,1)
                next_id = torch.gather(sorted_idx, -1, next_sorted).squeeze(-1)
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

        x = torch.cat([x, next_id.unsqueeze(-1)], dim=-1)  # append

    return x  # shape: (B, T + H)

from torch import nn
import math

class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias added to attention logits (not to token embeddings).

    Args:
        num_buckets (int): Number of relative-position buckets.
        max_distance (int): Distances >= max_distance are bucketed together.
        n_heads (int): Number of attention heads (for multi-head; for single-head use 1).

    Returns:
        (T, T) bias for single-head or (n_heads, T, T) for multi-head (broadcast over batch).
    """
    def __init__(self, num_buckets: int = 32, max_distance: int = 128, n_heads: int = 1):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.embedding = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """
        Implements T5 bucketing: small distances get their own buckets, larger share log-scaled buckets.
        relative_positions: (T, T) with values j - i (col - row).
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # We use only causal (i attends to <= i), so we only need negative or zero offsets.
        rp = -relative_positions
        rp = torch.clamp(rp, min=0)  # future positions -> 0 distance (won't be attended anyway due to mask)

        # Half the buckets for exact increments, half for log scale
        n_exact = num_buckets // 2
        is_small = rp < n_exact
        small_bucket = rp

        # log-scale for larger distances
        # avoid log(0) by clamping min to 1
        large_val = n_exact + (
            (torch.log(rp.float().clamp(min=1) / n_exact) / math.log(max_distance / n_exact))
            * (num_buckets - n_exact)
        ).long()
        large_val = torch.clamp(large_val, max=num_buckets - 1)

        return torch.where(is_small, small_bucket, large_val)

    def forward(self, T: int, device: torch.device):
        # relative positions j - i for i (rows), j (cols)
        ctx = torch.arange(T, device=device)
        rel = ctx[None, :] - ctx[:, None]  # (T, T)
        buckets = self._relative_position_bucket(rel)  # (T, T)
        # (T, T, n_heads) -> (n_heads, T, T)
        values = self.embedding(buckets).permute(2, 0, 1)
        if self.n_heads == 1:
            return values[0]  # (T, T)
        return values  # (n_heads, T, T)

def pick_device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")      # Apple Silicon
    elif torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

class StreamWindows(Dataset):
    """
    A dataset that streams windows of tokens from a tokenized sequence.

    Args:
        tokens: A torch.LongTensor of shape [N] containing the tokenized sequence.
        block_size: The size of the window to sample from the sequence.

    Returns:
        A tuple of two torch.LongTensor objects, x and y, of shape [T] and [T] respectively, where T is the block_size.
        x contains the current window of tokens, and y contains the next window of tokens shifted by one position.
    """
    def __init__(self, tokens: torch.LongTensor, block_size: int):
        assert tokens.ndim == 1
        self.tok = tokens
        self.block = block_size
        if len(tokens) <= block_size:
            raise ValueError("tokens must be longer than block_size")
        
    def __len__(self):
        # each index samples a random window; choose a large virtual length
        return 10_000
    
    def __getitem__(self, _):
        # sample start so that x has length block and y is shifted by +1
        i = torch.randint(0, len(self.tok) - self.block - 1, (1,)).item()
        x = self.tok[i:i+self.block]            # (T,)
        y = self.tok[i+1:i+self.block+1]        # (T,)
        return x, y

def make_block_size(step, warm=2_000, max_bs=1024, min_bs=256):
    if step < warm:  # fast early training
        return min_bs
    # linear ramp to max_bs
    frac = min(1.0, (step - warm) / (warm))
    return int(min_bs + frac * (max_bs - min_bs))

# 1) RoPE with context extension (positional interpolation / scaled time index)
def apply_rope(q, k, base_theta=10000.0, scale=10.0):
    """
    Rotary position embedding (RoPE) with optional time scaling.
    Setting scale>1.0 stretches positions so effective context grows ~linearly with scale.
    """
    B, T, D = q.shape
    assert D % 2 == 0
    d_half = D // 2
    device = q.device

    i = torch.arange(d_half, device=device).float()
    inv_freq = 1.0 / (base_theta ** (2 * i / D))
    # Divide positions by 'scale' (positional interpolation)
    t = torch.arange(T, device=device).float() / scale
    ang = torch.einsum('t,d->td', t, inv_freq)
    sin, cos = ang.sin()[None, ...], ang.cos()[None, ...]

    def rot(x):
        x1, x2 = x[..., :d_half], x[..., d_half:]
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        return torch.cat([xr1, xr2], dim=-1)

    return rot(q), rot(k)


def alibi_bias(T, device, slope=0.0):
    """
    ALiBi: additive linear bias proportional to distance (causal).
    Returns (T,T) tensor to add to attention logits.
    """
    pos = torch.arange(T, device=device)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()      # (T,T), non-negative lags only
    return -slope * dist



class SingleHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int = 128, ffn_mult: int = 2,
                 pos_mode: str = "alibi",      # {"rpb","rope","alibi","nope"}
                 rpb_num_buckets: int = 32, 
                 rpb_max_distance: int = 128,
                 alibi_slope: float = 1.0): # change this
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # value projection
        self.scale = d_k ** -0.5
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.latest_attn = None

        self.pos_mode = pos_mode
        if pos_mode == "rpb":
            self.rel_pos_bias = RelativePositionBias(
                num_buckets=rpb_num_buckets, max_distance=rpb_max_distance, n_heads=1
            )
        elif pos_mode == "alibi":
            self.alibi_slope = alibi_slope

    def forward(self, x, collect_attn: bool = False):  # x: (B,T,D)
        B, T, _ = x.size()
        h = self.ln1(x)
        q = self.q(h); k = self.k(h)
        v = self.v_proj(h)  # <-- FIX 1: compute V from h

        if self.pos_mode == "rope":
            q, k = apply_rope(q, k)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B,T,T)
        if self.pos_mode == "rpb":
            att = att + self.rel_pos_bias(T, device=x.device)
        elif self.pos_mode == "alibi":
            att = att + alibi_bias(T, x.device, slope=self.alibi_slope)

        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)

        if collect_attn:
            self.latest_attn = att.detach()

        ctx = att @ v
        x = x + ctx
        x = x + self.ffn(self.ln2(x))
        return x

class TinyCausalLM(nn.Module):
    """
    Two-layer, single-head causal LM.
    pos_mode:
        "rpb"   -> T5-style relative position BIAS inside attention (Chronos-like)
        "rope"  -> RoPE (rotary) applied to q,k; no abs pe added
        "alibi" -> ALiBi additive bias; no abs pe added. Best currently
        "nope"  -> no positions at all (not recommended unless testing)
        "abs"   -> optional learned GPT-2-style absolute embeddings added to x
        "onehot" -> one-hot absolute positions (concat then project)
    """
    def __init__(self, vocab_size: int, d_model: int = 256, d_k: int = 128, block_size: int = 512,
                 pos_mode: str = "alibi"):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.use_abs = (pos_mode == "abs")
        if self.use_abs:
            self.pos_emb = nn.Embedding(block_size, d_model)

        # pass pos_mode through to both blocks
        self.attn1 = SingleHeadCausalAttention(d_model, d_k, pos_mode=pos_mode)
        self.attn2 = SingleHeadCausalAttention(d_model, d_k, pos_mode=pos_mode)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

        # self.pos_P = 65
        # self.in_proj = nn.Linear(d_model + self.pos_P, d_model)

    def forward(self, idx, collect_attn: bool = False):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)  # (B,T,D)

        # absolute positions only if pos_mode=="abs"
        if self.use_abs:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]

        ## One hot embedding
        # pos_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T) % self.pos_P
        # pos_oh = F.one_hot(pos_ids, num_classes=self.pos_P).float()
        # x = torch.cat([x, pos_oh], dim=-1)
        # x = self.in_proj(x)

        x = self.attn1(x, collect_attn=collect_attn)
        x = self.attn2(x, collect_attn=collect_attn)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if collect_attn:
            return logits, [self.attn1.latest_attn, self.attn2.latest_attn]
        return logits

    # def forward_float(self, x_float: torch.Tensor, collect_attn: bool = False):
    #     """
    #     Args:
    #         x_float (torch.Tensor): Shape (B, T) float-valued inputs where integer
    #             values correspond to token positions along the real line (e.g., 1 -> 1.0).
    #         collect_attn (bool, optional): If True, also returns per-layer attention
    #             matrices (detached). Defaults to False.

    #     Returns:
    #         torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]:
    #             If collect_attn=False: logits of shape (B, T, vocab_size).
    #             If collect_attn=True: (logits, [attn1, attn2]) where each attn is (T, T).
    #     """
    #     assert x_float.dim() == 2, "x_float must be (B, T)"
    #     assert x_float.dtype.is_floating_point, "forward_float expects float inputs"
    #     B, T = x_float.shape
    #     assert T <= self.block_size

    #     # --- Differentiable interpolation between adjacent token embeddings ---
    #     # clamp so that i1 stays in-range
    #     V = self.tok_emb.num_embeddings
    #     x_clamped = x_float.clamp(min=0.0, max=float(V - 1))
    #     i0 = torch.floor(x_clamped).long()                         # (B, T)
    #     i1 = torch.clamp(i0 + 1, max=V - 1)                        # (B, T)
    #     w = (x_clamped - i0.to(x_clamped.dtype)).unsqueeze(-1)     # (B, T, 1)

    #     E0 = self.tok_emb(i0)                                      # (B, T, D)
    #     E1 = self.tok_emb(i1)                                      # (B, T, D)
    #     x = E0 + w * (E1 - E0)                                     # (B, T, D)
    #     # --------------------------------------------------------------------

    #     # absolute positions only if pos_mode=="abs" (kept identical to .forward)
    #     if self.use_abs:
    #         pos = torch.arange(T, device=x.device)
    #         x = x + self.pos_emb(pos)[None, :, :]

    #     x = self.attn1(x, collect_attn=collect_attn)
    #     x = self.attn2(x, collect_attn=collect_attn)
    #     x = self.ln_f(x)
    #     logits = self.lm_head(x)

    #     if collect_attn:
    #         return logits, [self.attn1.latest_attn, self.attn2.latest_attn]
    #     return logits



def train_next_token(
    tokens_train: torch.LongTensor,
    tokens_val: torch.LongTensor,
    tokens_test_out: torch.LongTensor,
    vocab_size: int,
    block_size: int = 512,
    batch_size: int = 32,
    d_model: int = 256,
    d_k: int = 128,
    steps: int = 5_000,
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    grad_accum: int = 1,
    device: str | torch.device | None = None,
):
    """
    Train a tiny causal language model to predict the next token in a sequence.

    Args:
        tokens: A torch.LongTensor of shape [N] containing the tokenized sequence.
        vocab_size: The size of the vocabulary.
        block_size: The size of the window to sample from the sequence.
        batch_size: The batch size for training.
        d_model: The dimension of the model.
        d_k: The dimension of the key.
        steps: The number of steps to train the model.
        lr: The learning rate for the optimizer.
        weight_decay: The weight decay for the optimizer.
        grad_accum: The number of micro-batches to accumulate gradients over when 
            training on a single GPU.
        device: The device to train the model on.

    Returns:
        A TinyCausalLM model.
    """
    # device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = pick_device()
    
    ## Create a dataloader that streams contiguous windows of tokens from the tokenized sequence.
    dl_train = DataLoader(StreamWindows(tokens_train, block_size), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val = DataLoader(StreamWindows(tokens_val, block_size), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val_ood = DataLoader(StreamWindows(tokens_test_out, block_size), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    it = iter(dl_train)

    model = TinyCausalLM(vocab_size, d_model=d_model, d_k=d_k, block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # schedule = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=steps)
    # schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    train_losses = []
    val_losses = []
    val_losses_ood = []
    model.train()
    for step in range(1, steps + 1):
        # gradient accumulation over grad_accum micro-batches
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl_train)
                x, y = next(it)
            x = x.to(device)                     # (B,T)
            y = y.to(device)                     # (B,T)
            logits = model(x)                    # (B,T,V)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)) # flatten
            loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # schedule.step()
        if step % 100 == 0:
            train_losses.append(loss_accum)
            # evaluate on validation set
            model.eval()
            with torch.no_grad():
                x_val, y_val = next(iter(dl_val))
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits = model(x_val)
                loss = loss_fn(logits.view(-1, vocab_size), y_val.view(-1))
                val_losses.append(loss.item())

                x_val_ood, y_val_ood = next(iter(dl_val_ood))
                x_val_ood = x_val_ood.to(device)
                y_val_ood = y_val_ood.to(device)
                logits = model(x_val_ood)
                loss = loss_fn(logits.view(-1, vocab_size), y_val_ood.view(-1))
                val_losses_ood.append(loss.item())

            print(f"step {step} | train loss {train_losses[-1]:.4f} | val loss {val_losses[-1]:.4f} | val ood loss {val_losses_ood[-1]:.4f}")
        

    return model, train_losses, val_losses, val_losses_ood

def _pack_config_from(model: TinyCausalLM) -> dict:
    return {
        "vocab_size": model.tok_emb.num_embeddings,
        "d_model": model.tok_emb.embedding_dim,
        "d_k": model.attn1.q.out_features,
        "block_size": model.block_size,
        "pos_mode": model.attn1.pos_mode,  # propagated to both blocks
    }

def batched_forward(model, X, batch_size=1000):
    """Given a model and a tensor X, return the logits for X in batches to avoid OOM"""
    all_logits = list()
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i:i+batch_size])
        all_logits.append(logits)
    return torch.cat(all_logits, dim=0)


def save_checkpoint(path: str, model: TinyCausalLM, optimizer: torch.optim.Optimizer | None = None, step: int | None = None, **extra):
    ckpt = {
        "config": _pack_config_from(model),
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optim"] = optimizer.state_dict()
    if step is not None:
        ckpt["step"] = step
    ckpt.update(extra)
    torch.save(ckpt, path)

def load_model(path: str, device: str | torch.device = "cpu") -> TinyCausalLM:
    ckpt = torch.load(path, map_location=device)
    m = TinyCausalLM(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m

def load_for_training(path: str, device: str | torch.device, lr: float, weight_decay: float = 0.0):
    ckpt = torch.load(path, map_location=device)
    m = TinyCausalLM(**ckpt["config"]).to(device)
    m.load_state_dict(ckpt["state_dict"])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    if "optim" in ckpt:
        opt.load_state_dict(ckpt["optim"])
    start_step = ckpt.get("step", 0)
    m.train()
    return m, opt, start_step