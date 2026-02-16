# Minimal autoregressive (next-token) training on a long token stream.
# - Tokens: 1D LongTensor of shape [N] with values in [0, V-1]
# - Trains a 2-layer, single-head, attention-only causal LM
# - Uses sliding windows of length block_size sampled from the 1D stream

import math, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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



# def apply_rope(q, k):
#     """
#     Rotary position embedding (RoPE) applied to q and k in-place style.

#     Args:
#         q, k: (B, T, D) float tensors with the same shape; D must be even.
#     Returns:
#         (q_rot, k_rot): RoPE-transformed q, k with same shapes as inputs.
#     """
#     B, T, D = q.shape
#     assert D % 2 == 0, "RoPE requires even d_k"
#     d_half = D // 2

#     # frequencies: θ^(2i/D) with θ=10_000 (RoFormer paper)
#     device = q.device
#     i = torch.arange(d_half, device=device).float()                # (D/2,)
#     inv_freq = 1.0 / (10000 ** (2 * i / D))                       # (D/2,)

#     t = torch.arange(T, device=device).float()                     # (T,)
#     ang = torch.einsum('t,d->td', t, inv_freq)                     # (T, D/2)
#     sin, cos = ang.sin()[None, ...], ang.cos()[None, ...]          # (1, T, D/2)

#     def rot(x):
#         x1, x2 = x[..., :d_half], x[..., d_half:]
#         # (B,T,D/2) each; broadcast sin/cos over batch
#         xr1 = x1 * cos - x2 * sin
#         xr2 = x1 * sin + x2 * cos
#         return torch.cat([xr1, xr2], dim=-1)

#     return rot(q), rot(k)


# 1) RoPE with context extension (positional interpolation / scaled time index)
def apply_rope(q, k, base_theta=10000.0, scale=5.0):
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


class SingleHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int = 128, ffn_mult: int = 2,
                 pos_mode: str = "alibi",      # {"rpb","rope","alibi","nope"}
                 rpb_num_buckets: int = 32, 
                 rpb_max_distance: int = 128,
                 alibi_slope: float = 1.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_k ** -0.5
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.latest_attn = None

        # new: choose positional scheme per layer
        self.pos_mode = pos_mode
        if pos_mode == "rpb":
            self.rel_pos_bias = RelativePositionBias(
                num_buckets=rpb_num_buckets, max_distance=rpb_max_distance, n_heads=1
            )
        elif pos_mode == "alibi":
            self.alibi_slope = alibi_slope
        # "rope" and "nope" require no parameters here

    def forward(self, x, collect_attn: bool = False):  # x: (B,T,D)
        B, T, _ = x.size()
        h = self.ln1(x)
        q = self.q(h); k = self.k(h)

        # RoPE rotates q, k BEFORE dot product
        if self.pos_mode == "rope":
            q, k = apply_rope(q, k)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B,T,T)

        # Additive biases (AFTER dot product, BEFORE mask/softmax)
        if self.pos_mode == "rpb":
            att = att + self.rel_pos_bias(T, device=x.device)      # (T,T) -> broadcast over batch
        elif self.pos_mode == "alibi":
            att = att + alibi_bias(T, x.device, slope=self.alibi_slope)

        # causal mask + softmax
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)

        if collect_attn:
            self.latest_attn = att.detach()

        ctx = att @ h
        x = x + self.v_proj(ctx)
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
    """
    def __init__(self, vocab_size: int, d_model: int = 256, d_k: int = 128, block_size: int = 512,
                 pos_mode: str = "alibi"):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.use_abs = (pos_mode == "alibi")
        if self.use_abs:
            self.pos_emb = nn.Embedding(block_size, d_model)

        # pass pos_mode through to both blocks
        self.attn1 = SingleHeadCausalAttention(d_model, d_k, pos_mode=pos_mode)
        self.attn2 = SingleHeadCausalAttention(d_model, d_k, pos_mode=pos_mode)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

    def forward(self, idx, collect_attn: bool = False):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)  # (B,T,D)

        # absolute positions only if pos_mode=="abs"
        if self.use_abs:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]

        x = self.attn1(x, collect_attn=collect_attn)
        x = self.attn2(x, collect_attn=collect_attn)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if collect_attn:
            return logits, [self.attn1.latest_attn, self.attn2.latest_attn]
        return logits




def make_param_groups(model: nn.Module, weight_decay: float = 0.1):
    """
    Args:
        model: PyTorch module.
        weight_decay: WD for decayed group.

    Returns:
        list[dict]: Two AdamW param groups (decay / no_decay).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if (p is None) or (not p.requires_grad):
            continue
        lname = name.lower()
        # Exclude: biases, all norm params (LayerNorm etc.), all embeddings,
        # and tied LM head (common in LMs) from weight decay.
        if (
            p.ndim == 1
            or lname.endswith(".bias")
            or "layernorm" in lname or ".ln" in lname or "norm" in lname
            or "emb" in lname
            or "lm_head" in lname
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# ---------- training ----------
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
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    ## Create a dataloader that streams contiguous windows of tokens from the tokenized sequence.
    dl_train = DataLoader(StreamWindows(tokens_train, block_size), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val = DataLoader(StreamWindows(tokens_val, block_size), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val_ood = DataLoader(StreamWindows(tokens_test_out, block_size), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    it = iter(dl_train)

    model = TinyCausalLM(vocab_size, d_model=d_model, d_k=d_k, block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # opt = torch.optim.AdamW(make_param_groups(model, weight_decay=weight_decay), lr=lr, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) ## does label smoothing help?

    ## Learning rate scheduler
    # total_updates = steps if grad_accum == 1 else math.ceil(steps / grad_accum)
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     opt, start_factor=1.0, end_factor=0.0, total_iters=total_updates
    # )


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
        # scheduler.step()
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


## Sample next tokens
@torch.no_grad()
def generate_next(model: TinyCausalLM, prefix: torch.LongTensor, max_new_tokens: int = 1):
    model.eval()
    device = next(model.parameters()).device
    seq = prefix.clone().to(device)             # (T,)
    for _ in range(max_new_tokens):
        x = seq[-model.block_size:].unsqueeze(0)  # crop to context
        logits = model(x)                        # (1,t,V)
        next_id = logits[:, -1, :].argmax(-1)    # greedy; swap for sampling as needed
        next_id = next_id.squeeze(0)  # Remove batch dimension
        if next_id.dim() == 0:  # If it's a scalar
            next_id = next_id.unsqueeze(0)  # Make it 1D
        seq = torch.cat([seq, next_id], dim=0)
    return seq

CONTEXT_LENGTH = 32*4
# CONTEXT_LENGTH = 32


# ---- example usage ----
if __name__ == "__main__":
    torch.manual_seed(0)
    # N = 10_000
    # toy periodic sequence
    tokens = torch.tensor(tok_train, dtype=torch.long)
    tokens_val = torch.tensor(tok_test, dtype=torch.long)
    tokens_test_out = torch.tensor(tok_test_out, dtype=torch.long)
    # model = train_next_token(tokens, vocab_size=(1 + VOCAB_SIZE), block_size=32, batch_size=64, steps=1000, d_model=128, d_k=64)
    # model, losses, val_losses, val_losses_ood = train_next_token(tokens, tokens_val, tokens_test_out, vocab_size=(1 + VOCAB_SIZE), block_size=CONTEXT_LENGTH, batch_size=64, steps=20000, d_model=128, d_k=64)
    # model, losses, val_losses, val_losses_ood = train_next_token(tokens, tokens_val, tokens_test_out, vocab_size=(1 + VOCAB_SIZE), block_size=CONTEXT_LENGTH, batch_size=128, steps=20000, d_model=128, d_k=64)
    # model, losses, val_losses, val_losses_ood = train_next_token(tokens, tokens_val[:N_TEST], tokens_test_out[:N_TEST], vocab_size=(1 + VOCAB_SIZE), 
    #                                                             block_size=CONTEXT_LENGTH, lr=1e-4, batch_size=64, steps=60000, d_model=128 // 2, d_k=64, weight_decay=1e0)
    model, losses, val_losses, val_losses_ood = train_next_token(tokens, tokens_val[:N_TEST], tokens_test_out[:N_TEST], vocab_size=(1 + VOCAB_SIZE), 
                                                                 block_size=CONTEXT_LENGTH, lr=1e-4, batch_size=64*2, steps=60000, d_model=128 * 2, d_k=64, weight_decay=1e0)
    # predict next token given last 50
    ctx = tokens[:50]
    out = generate_next(model, ctx, max_new_tokens=5)
    print("context:", ctx.tolist())
    print("preds:  ", out[-5:].tolist())



plt.figure(figsize=(10, 5))
plt.semilogy(losses, label="Train data in-domain")
plt.semilogy(val_losses, label="Test data in-domain")
plt.semilogy(val_losses_ood, label="Test data out-of-domain")
plt.ylim(1, None)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend(frameon=False)


