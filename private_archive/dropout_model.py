# === Practical anti-overfitting upgrades for your exact loop ===
# Drop-in edits: dropout, label smoothing, AdamW wd=0.1, cosine+warmup, early stop, better val averaging.

import math, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- dataset: optional token corruption (input-only) ----------
class StreamWindows(Dataset):
    def __init__(self, tokens: torch.LongTensor, block_size: int, corrupt_p: float = 0.0, vocab_size: int | None = None):
        assert tokens.ndim == 1
        self.tok = tokens; self.block = block_size
        self.corrupt_p = corrupt_p
        self.vocab_size = vocab_size
        if len(tokens) <= block_size:
            raise ValueError("tokens must be longer than block_size")
        if corrupt_p > 0 and vocab_size is None:
            raise ValueError("Set vocab_size when using corrupt_p > 0")

    def __len__(self): return 10_000

    def __getitem__(self, _):
        i = torch.randint(0, len(self.tok) - self.block - 1, (1,)).item()
        x = self.tok[i:i+self.block].clone()        # (T,)
        y = self.tok[i+1:i+self.block+1].clone()    # (T,)
        if self.corrupt_p > 0:
            # random token substitution on inputs only (does not touch labels)
            mask = torch.rand_like(x.float()) < self.corrupt_p
            rand_ids = torch.randint(0, self.vocab_size, x.shape, dtype=torch.long)
            x[mask] = rand_ids[mask]
        return x, y

# ---------- model: add dropout in embeddings, attention, and MLP ----------
def sinusoidal_positions(T, D, device):
    pos = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(D, device=device).float()[None, :]
    denom = torch.pow(10000.0, (2*(i//2))/D)
    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(pos / denom[:, 0::2]); pe[:, 1::2] = torch.cos(pos / denom[:, 1::2])
    return pe

class SingleHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int = 128, ffn_mult: int = 2, p_drop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_k ** -0.5
        self.attn_drop = nn.Dropout(p_drop)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(p_drop),
        )
        self.latest_attn = None

    def forward(self, x, collect_attn: bool = False):
        B, T, _ = x.size()
        h = self.ln1(x)
        q = self.q(h); k = self.k(h)
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        att = self.attn_drop(att)

        if collect_attn: self.latest_attn = att.detach()

        ctx = att @ h
        x = x + self.v_proj(ctx)
        x = x + self.ffn(self.ln2(x))
        return x

class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, d_k: int = 128, block_size: int = 512, p_drop: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(p_drop)
        self.attn1 = SingleHeadCausalAttention(d_model, d_k, p_drop=p_drop)
        self.attn2 = SingleHeadCausalAttention(d_model, d_k, p_drop=p_drop)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, collect_attn: bool = False):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)
        pe = sinusoidal_positions(T, x.size(-1), x.device)[None, :, :]
        x = self.emb_drop(x + pe)

        x = self.attn1(x, collect_attn=collect_attn)
        A1 = self.attn1.latest_attn if collect_attn else None
        x = self.attn2(x, collect_attn=collect_attn)
        A2 = self.attn2.latest_attn if collect_attn else None

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if collect_attn:
            return logits, [A1, A2]
        return logits


# ---------- training with label smoothing, cosine+warmup, early stopping ----------
def _warmup_cosine_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(step / max(1, warmup_steps), 1e-4)
        # cosine to 10% of base LR
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

@torch.no_grad()
def _eval_mean_loss(model, loss_fn, dl, device, n_batches: int = 8):
    model.eval()
    tot = 0.0
    it = iter(dl)
    for _ in range(n_batches):
        try: x, y = next(it)
        except StopIteration: it = iter(dl); x, y = next(it)
        x = x.to(device); y = y.to(device)
        logits = model(x)
        tot += loss_fn(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    return tot / n_batches

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
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_accum: int = 1,
    p_drop: float = 0.1,
    corrupt_p: float = 0.02,
    warmup_steps: int = 200,
    early_stop_patience: int = 10,
    device: str | torch.device | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dl_train = DataLoader(StreamWindows(tokens_train, block_size, corrupt_p=corrupt_p, vocab_size=vocab_size),
                          batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val = DataLoader(StreamWindows(tokens_val, block_size),
                        batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    dl_val_ood = DataLoader(StreamWindows(tokens_test_out, block_size),
                            batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    it = iter(dl_train)

    model = TinyCausalLM(vocab_size, d_model=d_model, d_k=d_k, block_size=block_size, p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_warmup_cosine_lambda(warmup_steps, steps))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_losses, val_losses, val_losses_ood = [], [], []
    best_val = float("inf"); best_state = None; bad_epochs = 0

    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(grad_accum):
            try: x, y = next(it)
            except StopIteration: it = iter(dl_train); x, y = next(it)
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1)) / grad_accum
            loss.backward()
            loss_accum += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            train_losses.append(loss_accum)
            v = _eval_mean_loss(model, loss_fn, dl_val, device, n_batches=8)
            vood = _eval_mean_loss(model, loss_fn, dl_val_ood, device, n_batches=8)
            val_losses.append(v); val_losses_ood.append(vood)
            print(f"step {step} | train {train_losses[-1]:.4f} | val {v:.4f} | val_ood {vood:.4f}")

            # early stopping on in-domain val
            if v + 1e-6 < best_val:
                best_val = v; best_state = {k: v_.cpu() for k, v_ in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    print(f"Early stop at step {step} (best val {best_val:.4f}).")
                    break

    if best_state is not None: model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_losses_ood


CONTEXT_LENGTH = 32
# ---- example usage ----
if __name__ == "__main__":
    torch.manual_seed(0)
    # N = 10_000
    # toy periodic sequence
    tokens = torch.tensor(tok_train, dtype=torch.long)
    tokens_val = torch.tensor(tok_test, dtype=torch.long)
    tokens_test_out = torch.tensor(tok_test_out, dtype=torch.long)

    # Better (regularized) call for the upgraded train_next_token:
    # model, losses, val_losses, val_losses_ood = train_next_token(
    #     tokens_train=tokens,
    #     tokens_val=tokens_val[:N_TEST],
    #     tokens_test_out=tokens_test_out[:N_TEST],
    #     vocab_size=1 + VOCAB_SIZE,
    #     block_size=CONTEXT_LENGTH,
    #     batch_size=64,
    #     d_model=128,
    #     d_k=64,
    #     steps=20000,
    #     lr=0.5e-4,                # with warmup+cosine
    #     weight_decay=0.1,        # AdamW decoupled wd
    #     grad_accum=1,
    #     p_drop=0.1,              # dropout in emb/attn/FFN
    #     corrupt_p=0.02,          # mild input token corruption
    #     warmup_steps=200,
    #     early_stop_patience=10,
    # )
    model, losses, val_losses, val_losses_ood = train_next_token(
        tokens_train=tokens,
        tokens_val=tokens_val[:N_TEST],
        tokens_test_out=tokens_test_out[:N_TEST],
        vocab_size=1 + VOCAB_SIZE,
        block_size=CONTEXT_LENGTH,
        batch_size=64,
        d_model=128,           # keep capacity fixed for now
        d_k=64*2,
        steps=60000,
        lr=2e-4,               # ↑ from 5e-5; do a short LR sweep 5e-5→3e-4
        weight_decay=0.01,     # ↓ from 0.1 (common for AdamW on transformers)
        grad_accum=1,
        p_drop=0.05,           # ↓ from 0.1 to reduce regularization
        corrupt_p=0.2,         # disable to rule out extra reg
        warmup_steps=1000,     # ~5% of 20k; helps avoid slow starts
        early_stop_patience=10,
    )
    # # predict next token given last 50
    # ctx = tokens[:50]
    # out = generate_next(model, ctx, max_new_tokens=5)
    # print("context:", ctx.tolist())
    # print("preds:  ", out[-5:].tolist())

plt.figure(figsize=(10, 5))
plt.semilogy(losses, label="Train data in-domain")
plt.semilogy(val_losses, label="Test data in-domain")
plt.semilogy(val_losses_ood, label="Test data out-of-domain")
plt.ylim(1, None)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend(frameon=False)
