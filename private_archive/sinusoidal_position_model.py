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


def sinusoidal_positions(T, D, device):
    pos = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(D, device=device).float()[None, :]
    denom = torch.pow(10000.0, (2*(i//2))/D)
    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(pos / denom[:, 0::2])
    pe[:, 1::2] = torch.cos(pos / denom[:, 1::2])
    return pe


# ---- model with attention tracking + rollout ----
class SingleHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int = 128, ffn_mult: int = 2):
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
        # storage for the most recent attention matrix (for rollout later)
        self.latest_attn = None

    def forward(self, x, collect_attn: bool = False):   # x: (B,T,D)
        B, T, _ = x.size()
        h = self.ln1(x)
        q = self.q(h); k = self.k(h)
        att = (q @ k.transpose(-2, -1)) * self.scale            # (B,T,T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf")).softmax(-1) # row-stochastic

        if collect_attn:
            # detach to avoid keeping autograd graphs; keep full-precision on device
            self.latest_attn = att.detach()

        ctx = att @ h                                           # (B,T,D)
        x = x + self.v_proj(ctx)                                # residual 1
        x = x + self.ffn(self.ln2(x))                           # residual 2 (tiny MLP)
        return x


class TinyCausalLM(nn.Module):
    """A two-layer, single-head, attention-only causal language model with attention rollout support."""
    def __init__(self, vocab_size: int, d_model: int = 256, d_k: int = 128, block_size: int = 512):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.attn1 = SingleHeadCausalAttention(d_model, d_k)
        self.attn2 = SingleHeadCausalAttention(d_model, d_k)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie

    def forward(self, idx, collect_attn: bool = False):  # idx: (B,T)
        B, T = idx.shape
        assert T <= self.block_size
        x = self.tok_emb(idx)                             # (B,T,D)

        # Sinusoidal positional embedding
        pe = sinusoidal_positions(T, x.size(-1), x.device)[None, :, :]
        x = x + pe

        # Transformer blocks
        x = self.attn1(x, collect_attn=collect_attn)
        x = self.attn2(x, collect_attn=collect_attn)
        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B,T,V)

        if collect_attn:
            # return logits and a list of per-layer attention tensors (B,T,T)
            return logits, [self.attn1.latest_attn, self.attn2.latest_attn]
        return logits


    @torch.no_grad()
    def attention_rollout(self, attn_list=None, add_residual: bool = True):
        """
        Compute attention rollout (Chefer et al. / Abnar & Zuidema) for a *causal* stack.

        Args:
            attn_list : list of (B,T,T) tensors, optional
                If None, uses the last collected attn matrices from the layers in order.
            add_residual : bool
                If True, augment each layer's attention with the identity and renormalize.

        Returns:
            rollout (B,T,T) tensor: Row-stochastic matrix mapping input tokens
                 (cols) -> output tokens (rows).
        """
        # Gather attentions from the most recent forward pass if not supplied
        if attn_list is None:
            attn_list = [self.attn1.latest_attn, self.attn2.latest_attn]
        assert all(a is not None for a in attn_list), "No stored attentions. Run a forward pass with collect_attn=True or pass attn_list."

        A_list = []
        for A in attn_list:
            # Optionally incorporate residual connection: A' = normalize(A + I)
            if add_residual:
                B, T, _ = A.shape
                I = torch.eye(T, device=A.device).expand(B, T, T)
                A = A + I
            # Row-normalize so rows sum to 1
            A = A / (A.sum(-1, keepdim=True) + 1e-8)
            A_list.append(A)

        # Joint attention: A_L @ A_{L-1} @ ... @ A_1
        joint = A_list[0]
        for A in A_list[1:]:
            joint = A @ joint
        return joint  # (B,T,T)



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
    loss_fn = nn.CrossEntropyLoss()

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




# ---------- generation (next-token inference) ----------
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

CONTEXT_LENGTH = 32
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
    model, losses, val_losses, val_losses_ood = train_next_token(tokens, tokens_val[:N_TEST], tokens_test_out[:N_TEST], vocab_size=(1 + VOCAB_SIZE), 
                                                                 block_size=CONTEXT_LENGTH, lr=5e-5, batch_size=64, steps=16000, d_model=128, d_k=64)
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
