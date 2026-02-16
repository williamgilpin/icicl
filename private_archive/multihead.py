# --- put this near the top of your script ---

def pick_device():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")      # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

SMALL_MAC_PRESET = dict(
    # Model size (keeps capacity modest)
    # d_model=192,            # small but trainable on M1/M2/M3
    # n_layers=6,
    # n_heads=6,
    # d_head=32,              # 6 * 32 = 192
    d_model=192,            # small but trainable on M1/M2/M3
    n_layers=6,
    n_heads=6,
    d_head=32,              # 6 * 32 = 192

    # Context & regularization
    block_size=512,         # start here; go to 768/1024 if you can
    pos_mode="alibi",       # good long-range bias; try "rope" later
    rope_scale=6.0,         # used only if pos_mode="rope"
    attn_dropout=0.10,
    resid_dropout=0.10,

    # Training
    batch_size=16,          # small micro-batch (fits on M1/M2)
    grad_accum=4,           # effective batch ~ 64 sequences/step
    lr=3e-4,
    warmup_steps=1000,
    weight_decay=0.10,
    steps=1_000,           # adjust to your budget

    # Robustness
    token_noise_prob=0.01,
    rdrop_alpha=0.25,

    # Misc
    compile_model=False,    # torch.compile is not mature on MPS
    device=pick_device()
)


# Minimal autoregressive (next-token) training on a long token stream.
# Upgrades:
# - Multi-head attention with head-specific ALiBi or scaled RoPE
# - RMSNorm + SwiGLU MLP, dropout
# - R-Drop regularization + input token noise
# - EMA weights, cosine LR with warmup, proper AdamW param groups
# - Context-length curriculum, AMP, gradient clipping
#
# NOTE: Keep your existing imports for tok_* variables and plotting.

import math, torch, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------- Data -----------------------

class StreamWindows(Dataset):
    """
    Streams random windows of length `block_size` from a 1D token tensor.
    You can change `self.block` on-the-fly to ramp context length.
    """
    def __init__(self, tokens: torch.LongTensor, block_size: int):
        assert tokens.ndim == 1
        if len(tokens) <= block_size:
            raise ValueError("tokens must be longer than block_size")
        self.tok = tokens
        self.block = block_size

    def __len__(self):
        return 10_000

    def __getitem__(self, _):
        i = torch.randint(0, len(self.tok) - self.block - 1, (1,)).item()
        x = self.tok[i:i+self.block]          # (T,)
        y = self.tok[i+1:i+self.block+1]      # (T,)
        return x, y

# Context-length curriculum (optional)
def make_block_size(step, warm=2_000, max_bs=1024, min_bs=256):
    if step < warm:
        return min_bs
    frac = min(1.0, (step - warm) / warm)
    return int(min_bs + frac * (max_bs - min_bs))

# ----------------------- Positional helpers -----------------------

def apply_rope(q, k, base_theta=10_000.0, scale=5.0):
    """
    RoPE with time scaling (positional interpolation).
    q,k: (B, H, T, Dh). Returns rotated q,k with same shape.
    """
    B, H, T, Dh = q.shape
    assert Dh % 2 == 0
    d_half = Dh // 2
    device = q.device
    i = torch.arange(d_half, device=device).float()
    inv_freq = 1.0 / (base_theta ** (2 * i / Dh))
    t = (torch.arange(T, device=device).float() / scale)  # scaled time
    ang = torch.einsum('t,d->td', t, inv_freq)            # (T, d_half)
    sin, cos = ang.sin()[None, None, ...], ang.cos()[None, None, ...]  # (1,1,T,d_half)

    def rot(x):
        x1, x2 = x[..., :d_half], x[..., d_half:]
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        return torch.cat([xr1, xr2], dim=-1)

    return rot(q), rot(k)

def _alibi_slopes(n_heads: int, device):
    """
    Head-specific ALiBi slopes (as in the paper / common implementations).
    Produces some long-range and some short-range heads.
    """
    import math
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest)
        slopes += get_slopes_power_of_2(2 * closest)[0::2][:n_heads - closest]
    return torch.tensor(slopes, device=device).view(n_heads)  # (H,)

def alibi_bias_heads(T: int, slopes: torch.Tensor):
    """
    Returns (H, T, T) additive bias where bias[h] = -slopes[h] * distance.
    """
    device = slopes.device
    pos = torch.arange(T, device=device)
    dist = (pos[None, :] - pos[:, None]).clamp(min=0).float()  # (T,T) non-negative lags
    return -(slopes[:, None, None] * dist[None, :, :])         # (H,T,T)

class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias (per-head).
    """
    def __init__(self, num_buckets=32, max_distance=128, n_heads=8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.embedding = nn.Embedding(num_buckets, n_heads)

    def _bucket(self, rel):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        rp = (-rel).clamp(min=0)  # causal
        n_exact = num_buckets // 2
        is_small = rp < n_exact
        small = rp
        large = n_exact + (
            (torch.log(rp.float().clamp(min=1) / n_exact) / math.log(max_distance / n_exact))
            * (num_buckets - n_exact)
        ).long()
        large = torch.clamp(large, max=num_buckets - 1)
        return torch.where(is_small, small, large)

    def forward(self, T: int, device: torch.device):
        ctx = torch.arange(T, device=device)
        rel = ctx[None, :] - ctx[:, None]       # (T,T)
        buckets = self._bucket(rel)             # (T,T)
        vals = self.embedding(buckets)          # (T,T,H)
        return vals.permute(2, 0, 1)            # (H,T,T)

# ----------------------- Blocks -----------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        a = torch.nn.functional.silu(self.w1(x))
        b = self.w2(x)
        x = self.w3(a * b)
        return self.dropout(x)

class MultiHeadCausalAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int = 8, d_head: int = 64,
        pos_mode: str = "alibi",             # {"rpb","rope","alibi","nope","abs"}
        rpb_num_buckets: int = 32,
        rpb_max_distance: int = 128,
        rope_scale: float = 5.0,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_model // n_heads
        self.d_model = d_model
        self.pos_mode = pos_mode
        self.rope_scale = rope_scale

        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.scale = self.d_head ** -0.5

        if pos_mode == "rpb":
            self.rel_pos_bias = RelativePositionBias(
                num_buckets=rpb_num_buckets, max_distance=rpb_max_distance, n_heads=n_heads
            )
        elif pos_mode == "alibi":
            self.register_buffer("alibi_slopes", None, persistent=False)
        # "rope"/"nope"/"abs" need no params here

        self.ln2 = RMSNorm(d_model)
        # SwiGLU with 4x expansion is standard; with SwiGLU you can use ~(2/3)*4.
        d_ff = int(4 * d_model * 2 / 3)
        self.ffn = SwiGLU(d_model, d_ff, dropout=resid_dropout)

        self.latest_attn = None  # optional inspection

    def _maybe_init_alibi(self, T, device):
        if self.pos_mode == "alibi" and (self.alibi_slopes is None or self.alibi_slopes.numel() != self.n_heads):
            self.alibi_slopes = _alibi_slopes(self.n_heads, device)

    def forward(self, x, collect_attn: bool = False):
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        k = self.k_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        v = self.v_proj(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)

        if self.pos_mode == "rope":
            q, k = apply_rope(q, k, scale=self.rope_scale)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        if self.pos_mode == "rpb":
            att = att + self.rel_pos_bias(T, device=x.device)[None, :, :, :]
        elif self.pos_mode == "alibi":
            self._maybe_init_alibi(T, x.device)
            att = att + alibi_bias_heads(T, self.alibi_slopes)[None, :, :, :]

        # causal mask
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(dim=-1)
        if collect_attn:
            self.latest_attn = att.detach()
        att = self.attn_dropout(att)

        ctx = torch.matmul(att, v)                    # (B,H,T,Dh)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        x = x + self.resid_dropout(self.o_proj(ctx))

        x = x + self.ffn(self.ln2(x))
        return x

# ----------------------- Model -----------------------

class TinyCausalLM(nn.Module):
    """
    Small causal LM with n_layers blocks and multi-head attention.
    pos_mode:
        "alibi" (default), "rope", "rpb", "nope", "abs"
    """
    def __init__(
        self, vocab_size: int, d_model: int = 256, n_layers: int = 6,
        n_heads: int = 8, d_head: int = 64, block_size: int = 512,
        pos_mode: str = "alibi", rope_scale: float = 5.0,
        attn_dropout: float = 0.1, resid_dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Absolute positions ONLY if pos_mode == "abs"
        self.use_abs = (pos_mode == "abs")
        if self.use_abs:
            self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            MultiHeadCausalAttention(
                d_model=d_model, n_heads=n_heads, d_head=d_head,
                pos_mode=pos_mode, rope_scale=rope_scale,
                attn_dropout=attn_dropout, resid_dropout=resid_dropout
            )
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.tok_emb.weight

        # Init (lightweight, works well with RMSNorm)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, idx, collect_attn: bool = False):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        x = self.tok_emb(idx)  # (B,T,D)

        if self.use_abs:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]

        for blk in self.blocks:
            x = blk(x, collect_attn=collect_attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ----------------------- Optim utils -----------------------

def make_param_groups(model: nn.Module, weight_decay: float = 0.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if (p is None) or (not p.requires_grad):
            continue
        lname = name.lower()
        if (
            p.ndim == 1
            or lname.endswith(".bias")
            or "norm" in lname
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

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        self.backup = None
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def store(self, model: nn.Module):
        self.backup = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.shadow[k].data)
    @torch.no_grad()
    def restore(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.backup[k].data)

# ----------------------- Training -----------------------

def _eval_loss(model, dl, vocab_size, iters=50, amp_dtype=None):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    device = next(model.parameters()).device
    it = iter(dl)
    with torch.no_grad():
        for _ in range(iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl); x, y = next(it)
            x, y = x.to(device), y.to(device)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            else:
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            tot += loss.item(); n += 1
    return tot / max(1, n)

def train_next_token(
    tokens_train: torch.LongTensor,
    tokens_val: torch.LongTensor,
    tokens_val_ood: torch.LongTensor,
    vocab_size: int,
    block_size: int = 512,           # maximum context length used late in training
    batch_size: int = 64,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    d_head: int = 64,
    steps: int = 5_000,
    lr: float = 6e-4,
    warmup_steps: int = 400,
    weight_decay: float = 0.1,
    grad_accum: int = 1,
    token_noise_prob: float = 0.01,  # random token replacement on inputs
    rdrop_alpha: float = 0.5,        # set to 0.0 to disable R-Drop
    pos_mode: str = "alibi",
    rope_scale: float = 5.0,
    attn_dropout: float = 0.1,
    resid_dropout: float = 0.1,
    compile_model: bool = False,
    device: str | torch.device | None = None,
):
    """
    Trains a compact causal LM.
    Returns: model, train_losses, val_losses, val_losses_ood
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dl_train = DataLoader(StreamWindows(tokens_train, block_size=min(256, block_size)),
                          batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val = DataLoader(StreamWindows(tokens_val, block_size=block_size),
                        batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val_ood = DataLoader(StreamWindows(tokens_val_ood, block_size=block_size),
                            batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = TinyCausalLM(
        vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, block_size=block_size,
        pos_mode=pos_mode, rope_scale=rope_scale,
        attn_dropout=attn_dropout, resid_dropout=resid_dropout
    ).to(device)

    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    opt = torch.optim.AdamW(
        make_param_groups(model, weight_decay=weight_decay),
        lr=lr, betas=(0.9, 0.95), eps=1e-8
    )

    # Cosine LR with warmup
    def lr_lambda(step_idx):
        if step_idx < warmup_steps:
            return max(1e-4, step_idx + 1) / max(1, warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    ema = EMA(model, decay=0.999)

    loss_xent = nn.CrossEntropyLoss(label_smoothing=0.1)  # slight label smoothing

    # AMP dtype
    amp_dtype = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_losses, val_losses, val_losses_ood = [], [], []
    model.train()
    it = iter(dl_train)

    for step in range(1, steps + 1):
        # ramp up context length used in training
        new_block = make_block_size(step, warm=min(2_000, steps//3), max_bs=block_size, min_bs=min(256, block_size))
        dl_train.dataset.block = new_block

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl_train)
                x, y = next(it)
            x, y = x.to(device), y.to(device)

            # input token noise (improves OOD)
            if token_noise_prob > 0.0 and model.training:
                noise_mask = torch.rand_like(x.float()) < token_noise_prob
                rand_tokens = torch.randint(0, vocab_size, x.shape, device=x.device)
                x_noisy = torch.where(noise_mask, rand_tokens, x)
            else:
                x_noisy = x

            # R-Drop: two forward passes with dropout
            if rdrop_alpha > 0.0:
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        logits1 = model(x_noisy)
                        logits2 = model(x_noisy)
                        ce1 = loss_xent(logits1.view(-1, vocab_size), y.view(-1))
                        ce2 = loss_xent(logits2.view(-1, vocab_size), y.view(-1))
                        p1 = torch.log_softmax(logits1, dim=-1)
                        p2 = torch.log_softmax(logits2, dim=-1)
                        # symmetric KL on distributions over vocab for each position
                        kl = torch.mean(torch.sum(torch.exp(p1) * (p1 - p2), dim=-1) +
                                        torch.sum(torch.exp(p2) * (p2 - p1), dim=-1)) * 0.5
                        loss = (ce1 + ce2) * 0.5 + rdrop_alpha * kl
                else:
                    logits1 = model(x_noisy)
                    logits2 = model(x_noisy)
                    ce1 = loss_xent(logits1.view(-1, vocab_size), y.view(-1))
                    ce2 = loss_xent(logits2.view(-1, vocab_size), y.view(-1))
                    p1 = torch.log_softmax(logits1, dim=-1)
                    p2 = torch.log_softmax(logits2, dim=-1)
                    kl = torch.mean(torch.sum(torch.exp(p1) * (p1 - p2), dim=-1) +
                                    torch.sum(torch.exp(p2) * (p2 - p1), dim=-1)) * 0.5
                    loss = (ce1 + ce2) * 0.5 + rdrop_alpha * kl
            else:
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        logits = model(x_noisy)
                        loss = loss_xent(logits.view(-1, vocab_size), y.view(-1))
                else:
                    logits = model(x_noisy)
                    loss = loss_xent(logits.view(-1, vocab_size), y.view(-1))

            loss = loss / grad_accum

            scaler.scale(loss).backward()
            loss_accum += loss.item() * grad_accum

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        ema.update(model)

        if step % 100 == 0:
            train_losses.append(loss_accum)
            # evaluate with EMA weights for stability
            ema.store(model); ema.copy_to(model)
            v = _eval_loss(model, dl_val, vocab_size, iters=50, amp_dtype=amp_dtype)
            vo = _eval_loss(model, dl_val_ood, vocab_size, iters=50, amp_dtype=amp_dtype)
            ema.restore(model)

            val_losses.append(v)
            val_losses_ood.append(vo)
            print(f"step {step} | train {train_losses[-1]:.4f} | val {v:.4f} | val OOD {vo:.4f} | bs {dl_train.dataset.block}")

    return model, train_losses, val_losses, val_losses_ood

# ----------------------- Sampling -----------------------

@torch.no_grad()
def generate_next(model: TinyCausalLM, prefix: torch.LongTensor, max_new_tokens: int = 1):
    model.eval()
    device = next(model.parameters()).device
    seq = prefix.clone().to(device)
    for _ in range(max_new_tokens):
        x = seq[-model.block_size:].unsqueeze(0)
        logits = model(x)
        next_id = logits[:, -1, :].argmax(-1)
        next_id = next_id.squeeze(0)
        if next_id.dim() == 0:
            next_id = next_id.unsqueeze(0)
        seq = torch.cat([seq, next_id], dim=0)
    return seq

# ----------------------- Example usage -----------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Expect tok_train, tok_test, tok_test_out, VOCAB_SIZE, N_TEST defined upstream.
    tokens = torch.tensor(tok_train, dtype=torch.long)
    tokens_val = torch.tensor(tok_test, dtype=torch.long)
    tokens_test_out = torch.tensor(tok_test_out, dtype=torch.long)

    CONTEXT_LENGTH = 32 * 4

    # model, losses, val_losses, val_losses_ood = train_next_token(
    #     tokens, tokens_val[:N_TEST], tokens_test_out[:N_TEST],
    #     vocab_size=(1 + VOCAB_SIZE),
    #     block_size=CONTEXT_LENGTH,
    #     batch_size=128,
    #     d_model=256,
    #     n_layers=8,
    #     n_heads=8,
    #     d_head=64,
    #     steps=60_000,
    #     lr=6e-4,
    #     warmup_steps=2_000,
    #     weight_decay=0.1,
    #     grad_accum=1,
    #     token_noise_prob=0.01,
    #     rdrop_alpha=0.5,
    #     pos_mode="alibi",      # try "rope" with rope_scale=6.0 for extrapolation
    #     rope_scale=6.0,
    #     attn_dropout=0.1,
    #     resid_dropout=0.1,
    #     compile_model=False
    # )

    # Example call (unchanged data plumbing):
    model, losses, val_losses, val_losses_ood = train_next_token(
        torch.tensor(tok_train, dtype=torch.long),
        torch.tensor(tok_test, dtype=torch.long)[:N_TEST],
        torch.tensor(tok_test_out, dtype=torch.long)[:N_TEST],
        vocab_size=(1 + VOCAB_SIZE),
        **SMALL_MAC_PRESET
    )

    # predict next token given last 50
    ctx = tokens[:50]
    out = generate_next(model, ctx, max_new_tokens=5)
    print("context:", ctx.tolist())
    print("preds:  ", out[-5:].tolist())

    # Plot (requires matplotlib imported elsewhere)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.semilogy(losses, label="Train data in-domain")
    plt.semilogy(val_losses, label="Test data in-domain (EMA eval)")
    plt.semilogy(val_losses_ood, label="Test data out-of-domain (EMA eval)")
    plt.ylim(1, None)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.show()



