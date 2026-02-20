import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from itertools import combinations
from typing import List, Tuple

# from scipy.stats import entropy
from .analysis import js_divergence as entropy


def next_token_empirical_probs(corpus: np.ndarray,
                               queries: np.ndarray,
                               L: int,
                               k: int) -> np.ndarray:
    """
    Compute empirical next-token probabilities for each query based on last k tokens.

    Args:
        corpus (np.ndarray): 1D int array of tokens (vocab size L).
        queries (np.ndarray): 2D int array of shape (B, C) with query prefixes.
        L (int): Vocabulary size (tokens in [0, L-1]).
        k (int): Context length (1 <= k <= C).

    Returns:
        np.ndarray: Float array of shape (B, L). Row b is the empirical distribution
            of the next token given the last k tokens of queries[b]. If the k-gram
            never appears in the corpus, the entire row is np.nan.
    """
    if k < 1 or k > queries.shape[1]:
        raise ValueError("k must be in [1, C].")
    if corpus.ndim != 1:
        raise ValueError("corpus must be a 1D array.")
    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array.")
    if len(corpus) < k + 1:
        # No (k+1)-grams exist at all
        out = np.full((queries.shape[0], L), np.nan, dtype=float)
        return out

    # Extract all (k+1)-grams from the corpus
    w = sliding_window_view(corpus, k + 1)  # shape: (N - k, k+1)
    ctx = w[:, :k]                           # shape: (N - k, k)
    nxt = w[:, -1]                           # shape: (N - k,)

    # Unique k-grams and inverse index for occurrences
    keys, inv = np.unique(ctx, axis=0, return_inverse=True)  # keys: (M, k)

    # Count next-token frequencies per k-gram (M, L)
    counts = np.zeros((keys.shape[0], L), dtype=np.int64)
    np.add.at(counts, (inv, nxt), 1)

    # Normalize to probabilities (avoid division by zero; rows with zero stay zero)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs_by_key = counts / totals

    # Map query last-k contexts to key indices
    qctx = queries[:, -k:]                   # (B, k)
    # Build dictionary for fast lookup
    key_map = {tuple(row): i for i, row in enumerate(keys)}

    out = np.full((queries.shape[0], L), np.nan, dtype=float)
    # Fill rows where the context was seen
    for b, row in enumerate(qctx):
        idx = key_map.get(tuple(row))
        if idx is not None and totals[idx, 0] > 0:
            out[b] = probs_by_key[idx]

    return out

def next_token_empirical_probs_all_positions(
    corpus: np.ndarray,
    queries: np.ndarray,
    L: int,
    k: int,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Empirical next-token probabilities for *all* k-gram position subsets
    (non-consecutive allowed) chosen from the C-length prefix.

    For each subset of positions S ⊂ {0,…,C-1} with |S|=k (0-based indices,
    where C-1 is the most recent token before the next-token), this computes
    P(x_{t} | x_{t-C:t-1}[S]) from the corpus and evaluates it for each query.

    Args:
        corpus (np.ndarray): 1D int array of tokens (vocab size L).
        queries (np.ndarray): 2D int array of shape (B, C) with query prefixes.
        L (int): Vocabulary size (tokens in [0, L-1]).
        k (int): Context size (1 <= k <= C).

    Returns:
        (np.ndarray, List[Tuple[int,...]]):
            probs: float array of shape (M, B, L), where M = comb(C, k).
                   probs[m, b] is the empirical distribution over next token
                   for query b using the position subset combs[m].
                   If a k-gram was unseen in the corpus, that row is np.nan.
            combs: list of the position tuples (in lexicographic order) used
                   to index the first dimension of probs.

    Raises:
        ValueError: On invalid inputs.
    """
    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array (B, C).")
    if corpus.ndim != 1:
        raise ValueError("corpus must be a 1D array.")
    B, C = queries.shape
    if k < 1 or k > C:
        raise ValueError("k must be in [1, C].")
    if L <= 0:
        raise ValueError("L must be positive.")

    # Need windows that expose the full C-token prefix preceding each next token.
    if corpus.size < C + 1:
        M = int(np.math.comb(C, k))
        return np.full((M, B, L), np.nan, dtype=float), list(combinations(range(C), k))

    w = sliding_window_view(corpus, C + 1)        # shape: (N - C, C+1)
    past = w[:, :C]                                # (N - C, C)
    nxt  = w[:, -1]                                # (N - C,)

    combs = list(combinations(range(C), k))
    probs = np.full((len(combs), B, L), np.nan, dtype=float)

    # For each position subset, build an empirical model from the corpus and evaluate on queries
    for m, comb in enumerate(combs):
        ctx = past[:, comb]                        # (N - C, k)

        # Deduplicate the observed k-grams and count next tokens
        keys, inv = np.unique(ctx, axis=0, return_inverse=True)  # keys: (M_k, k)
        counts = np.zeros((keys.shape[0], L), dtype=np.int64)
        np.add.at(counts, (inv, nxt), 1)

        totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs_by_key = counts / totals

        # Fast lookup dict for query contexts
        key_map = {tuple(row): i for i, row in enumerate(keys)}

        # Evaluate each query prefix under this subset
        qctx = queries[:, comb]                    # (B, k)
        for b in range(B):
            idx = key_map.get(tuple(qctx[b]))
            if idx is not None and totals[idx, 0] > 0:
                probs[m, b] = probs_by_key[idx]

    return np.array(probs), np.array(combs)
    

def next_token_empirical_probs_last_and_each_position(
    corpus: np.ndarray,
    queries: np.ndarray,
    L: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Empirical next-token probabilities for all 2-gram models that use the
    last context token and exactly one other context position.

    For each i in {0,…,C-2}, this computes P(x_t | x_{t-C+i}, x_{t-1}).

    Args:
        corpus (np.ndarray): 1D int array of tokens (vocab size L).
        queries (np.ndarray): 2D int array of shape (B, C) with query prefixes.
        L (int): Vocabulary size (tokens in [0, L-1]).

    Returns:
        (np.ndarray, List[Tuple[int,int]]):
            probs: float array of shape (C-1, B, L). probs[m, b] is the
                   empirical distribution over next token for query b using
                   positions combs[m] = (i, C-1). If a 2-gram was unseen,
                   that row is np.nan.
            combs: list of the (i, C-1) position pairs used.
    """
    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array (B, C).")
    if corpus.ndim != 1:
        raise ValueError("corpus must be a 1D array.")
    B, C = queries.shape
    if C < 2:
        raise ValueError("C must be at least 2 for 2-gram models.")
    if L <= 0:
        raise ValueError("L must be positive.")

    # Not enough data to expose any full C-prefix + next token windows
    if corpus.size < C + 1:
        return np.full((C - 1, B, L), np.nan, dtype=float), [(i, C - 1) for i in range(C - 1)]

    w = sliding_window_view(corpus, C + 1)  # (N - C, C+1)
    past = w[:, :C]                         # (N - C, C)
    nxt  = w[:, -1]                         # (N - C,)

    combs = [(i, C - 1) for i in range(C - 1)]
    probs = np.full((C - 1, B, L), np.nan, dtype=float)

    # For each (i, C-1), map the 2-gram to a single integer key = a*L + b
    last_col = past[:, C - 1]
    q_last   = queries[:, C - 1]

    for m, i in enumerate(range(C - 1)):
        key = past[:, i] * L + last_col                  # (N - C,)
        uniq, inv = np.unique(key, return_inverse=True)  # uniq: (M_i,)
        counts = np.zeros((uniq.size, L), dtype=np.int64)
        np.add.at(counts, (inv, nxt), 1)                 # accumulate next-token counts

        totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs_by_key = counts / totals               # rows with zero totals -> NaN

        # Fast lookup: integer key -> row index
        row_of = {int(k): j for j, k in enumerate(uniq)}

        # Evaluate queries for this pair (i, C-1)
        qkey = queries[:, i] * L + q_last                # (B,)
        for b in range(B):
            j = row_of.get(int(qkey[b]))
            if j is not None and totals[j, 0] > 0:
                probs[m, b] = probs_by_key[j]

    return probs, combs


def next_token_empirical_probs_custom_comb(
    corpus: np.ndarray,
    queries: np.ndarray,
    combs: List[Tuple[int, ...]],
    L: int = None,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """
    Empirical next-token probabilities for *all* k-gram position subsets
    (non-consecutive allowed) chosen from the C-length prefix.

    For each subset of positions S ⊂ {0,…,C-1} with |S|=k (0-based indices,
    where C-1 is the most recent token before the next-token), this computes
    P(x_{t} | x_{t-C:t-1}[S]) from the corpus and evaluates it for each query.

    Args:
        corpus (np.ndarray): 1D int array of tokens (vocab size L).
        queries (np.ndarray): 2D int array of shape (B, C) with query prefixes.
        combs (List[Tuple[int, ...]]): List of position tuples (in lexicographic order) used
                   to index the first dimension of probs.
        L (int): Vocabulary size (tokens in [0, L-1]).

    Returns:
        (np.ndarray, List[Tuple[int,...]]):
            probs: float array of shape (M, B, L), where M = comb(C, k).
                   probs[m, b] is the empirical distribution over next token
                   for query b using the position subset combs[m].
                   If a k-gram was unseen in the corpus, that row is np.nan.
            combs: list of the position tuples (in lexicographic order) used
                   to index the first dimension of probs.

    Raises:
        ValueError: On invalid inputs.
    """
    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array (B, C).")
    if corpus.ndim != 1:
        raise ValueError("corpus must be a 1D array.")
    B, C = queries.shape

    if L is None:
        # faster than np.unique for large arrays
        L = int(corpus.max()) + 1

    w = sliding_window_view(corpus, C + 1)           # (N-C, C+1)
    past = w[:, :C]                                   # (N-C, C)
    nxt  = w[:, -1]                                   # (N-C,)

    probs = np.full((len(combs), B, L), np.nan, dtype=float)

    # Helper: pack rows to bytes so we can unique/searchsorted quickly
    def _pack_rows(a: np.ndarray) -> np.ndarray:
        a_c = np.ascontiguousarray(a)
        return a_c.view(np.dtype((np.void, a_c.dtype.itemsize * a_c.shape[1]))).ravel()

    for m, comb in enumerate(combs):
        ctx = past[:, comb]                            # (N-C, k)
        qctx = queries[:, comb]                        # (B, k)

        # Deduplicate k-grams via packed bytes (much faster than axis=0 on large arrays)
        ctx_pack = _pack_rows(ctx)
        keys_pack, inv = np.unique(ctx_pack, return_inverse=True)  # keys sorted by bytes
        n_keys = keys_pack.shape[0]

        # Count next tokens per key
        counts = np.zeros((n_keys, L), dtype=np.int64)
        np.add.at(counts, (inv, nxt), 1)

        totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            probs_by_key = counts / totals

        # Vectorized query lookup using searchsorted on packed representation
        q_pack = _pack_rows(qctx)
        idx = np.searchsorted(keys_pack, q_pack)                # candidate indices
        in_range = idx < n_keys
        match = np.zeros(B, dtype=bool)
        match[in_range] = keys_pack[idx[in_range]] == q_pack[in_range]

        if match.any():
            valid_idx = idx[match]
            # Only keep keys that have at least one observation
            nonzero = (totals[valid_idx, 0] > 0)
            if nonzero.any():
                mb = np.flatnonzero(match)[nonzero]
                kv = valid_idx[nonzero]
                probs[m, mb] = probs_by_key[kv]

    return probs, combs


import torch
@torch.inference_mode()
def estimate_positionwise_marginals(
    model,
    token_sequences: torch.LongTensor,
    vocab_size: int,
    K_max: int,
    device: torch.device | str | None = None,
    batch_size: int = 1024,
    average_probs: bool = True
) -> List[torch.Tensor]:
    """
    Compute p(next | token_at_-k = t) for k = 1..K_max by attributing each batch's
    next-token distribution to the token appearing k steps before the end of the context.

    Args:
        model: Autoregressive LM; must accept (b, C) token IDs and return logits (b, V).
        token_sequences (LongTensor): Shape (N, C). Each row is a context to predict the next token from.
        vocab_size (int): Vocabulary size V expected from the model.
        K_max (int): Furthest offset to condition on (1 = last token).
        device (torch.device | str | None, optional): If set, inputs are moved here before the forward pass.
        batch_size (int, optional): Number of contexts per forward pass.

    Returns:
        List[torch.Tensor]: pos_conds where pos_conds[k] has shape (V, V) and row t is
            p(next | token_at_-k = t). Indices 0..K_max; index 0 is unused (None).
    """
    assert K_max >= 1, "K_max must be >= 1"
    V = int(vocab_size)

    # Resolve devices
    model_device = next(model.parameters()).device
    run_device = model_device if device is None else torch.device(device)

    N, C = token_sequences.shape
    if K_max > C:
        raise ValueError(f"K_max={K_max} exceeds context length C={C}")

    # Accumulators: counts[k][t] sums predicted next-token probs when token_at_-k == t
    counts: List[torch.Tensor] = [None] * (K_max + 1)
    for k in range(1, K_max + 1):
        counts[k] = torch.zeros(V, V, device=run_device, dtype=torch.float32)

    # Batched evaluation, iterate over batches of contexts
    for start in range(0, N, batch_size):
        ctx = token_sequences[start:start + batch_size].to(run_device, non_blocking=True)  # (b, C)
        logits = model(ctx)[:, -1, :]                                              # (b, V)
        if logits.size(-1) != V:
            raise ValueError(f"vocab_size mismatch: expected V={V}, got V={logits.size(-1)} from model")

        if average_probs:
            contrib = logits.softmax(dim=-1)                                              # (b, V)
        else:
            # One-hot rows for hard predictions
            pred = logits.argmax(dim=-1)                                                  # (b,)
            contrib = torch.zeros_like(logits, dtype=torch.float32)
            contrib.scatter_(1, pred.unsqueeze(1), 1.0)                                   # (b, V)

        # Attribute this batch's contribution to each offset k
        for k in range(1, K_max + 1):
            idx = ctx[:, -k]  # (b,)    
            ## Along index 0, at index idx, add the values of contrib                                                     
            counts[k].index_add_(0, idx, contrib) # (V,V) += (b,V)                                         

    # Normalize row-wise to obtain conditional distributions
    pos_conds: List[torch.Tensor] = [None] * (K_max + 1)
    for k in range(1, K_max + 1):
        row_sums = counts[k].sum(dim=1, keepdim=True)# (V,1); equals #occurrences of each t
        denom = row_sums.masked_fill_(row_sums == 0, 1.0) # Avoid div-by-zero
        pos_conds[k] = (counts[k] / denom).to("cpu") # (V,V)

    return pos_conds

def teacher_projected_markov_probs(train_tensor_np, probs, combs):
    """
    Build the KL-optimal Markov approximation of the teacher distribution for each comb.

    Args:
        train_tensor_np (np.ndarray): Shape (N, CONTEXT_LENGTH). Context windows as token ids.
        probs (np.ndarray): Shape (N, V). Teacher probs p(y | full context) for each window.
        combs (list[np.ndarray]): Each array contains context indices to condition on (your existing combs).

    Returns:
        list[np.ndarray]: probs_proj[k] has shape (N, V) with q_k(y | suffix) = E[p(y)|suffix].
    """
    N, V = probs.shape
    probs_proj = []

    for comb in combs:
        # suffix tokens used as the Markov state for this order/comb
        suffix = train_tensor_np[:, comb]  # shape (N, k)
        # group identical suffixes
        _, inv = np.unique(suffix, axis=0, return_inverse=True)
        G = inv.max() + 1

        # sum teacher distributions within each group
        sums = np.zeros((G, V), dtype=np.float32)
        np.add.at(sums, inv, probs.astype(np.float32))

        counts = np.bincount(inv).astype(np.float32)  # shape (G,)
        means = sums / counts[:, None]                # shape (G, V)

        # assign group mean back to each position
        probs_proj.append(means[inv])

    return probs_proj


def kl_sweep_and_marginal_improvement(probs_proj, probs, eps=1e-12):
    """
    Args:
        probs_proj (list[np.ndarray]): Each (N, V), Markov-projected q_k for each comb.
        probs (np.ndarray): (N, V), teacher p.
        eps (float): small constant for numerical stability

    Returns:
        all_kl_div (np.ndarray): (K, N) KL(p || q_k) per position
        Lk (np.ndarray): (K,) mean KL per k
        delta (np.ndarray): (K,) marginal improvement L_{k-1} - L_k, with delta[0]=nan
    """
    all_kl_div = []
    for q in probs_proj:
        # KL(p || q) per position
        all_kl_div.append(entropy(probs + eps, q + eps, axis=1))
    all_kl_div = np.array(all_kl_div)        # (K, N)

    Lk = all_kl_div.mean(axis=1)             # (K,)
    delta = np.empty_like(Lk)
    delta[0] = np.nan
    delta[1:] = Lk[:-1] - Lk[1:]             # marginal KL reduction

    return all_kl_div, Lk, delta



# def effective_order_from_delta(delta, k_values=None, rho=0.95):
#     """
#     Compute an "effective order" as the smallest k capturing a fraction rho of total marginal gain.

#     Args:
#         delta (np.ndarray): Shape (K,), marginal improvements. delta[0] may be nan.
#         k_values (np.ndarray or None): Shape (K,), the k corresponding to each entry. If None, uses 1..K.
#         rho (float): Target cumulative fraction (e.g., 0.95).

#     Returns:
#         dict: {
#             "k_eff": int,
#             "cum_frac": np.ndarray,
#             "total_gain": float,
#         }
#     """
#     if k_values is None:
#         k_values = np.arange(1, len(delta) + 1)

#     d = np.array(delta, dtype=np.float64)
#     d[~np.isfinite(d)] = 0.0
#     d = np.maximum(d, 0.0)

#     total = d.sum()
#     if total <= 0:
#         return {"k_eff": int(k_values[-1]), "cum_frac": np.zeros_like(d), "total_gain": float(total)}

#     cum = np.cumsum(d)
#     frac = cum / total
#     k_eff = int(k_values[np.searchsorted(frac, rho, side="left")])

#     return {"k_eff": k_eff, "cum_frac": frac, "total_gain": float(total)}


# def fit_exponential_decay(delta, k_values=None, min_frac_of_max=1e-3):
#     """
#     Fit log(delta_k) ≈ a - k/tau over the region where delta is above a noise floor.

#     Args:
#         delta (np.ndarray): Shape (K,), marginal improvements; delta[0] may be nan.
#         k_values (np.ndarray or None): Shape (K,), k corresponding to each entry. If None, uses 1..K.
#         min_frac_of_max (float): Keep points with delta >= max(delta)*min_frac_of_max.

#     Returns:
#         dict: {
#             "tau": float or np.nan,
#             "a": float or np.nan,
#             "used_k": np.ndarray,
#             "used_delta": np.ndarray,
#         }
#     """
#     if k_values is None:
#         k_values = np.arange(1, len(delta) + 1)

#     d = np.array(delta, dtype=np.float64)
#     d[~np.isfinite(d)] = 0.0
#     d = np.maximum(d, 0.0)

#     dmax = d.max()
#     if dmax <= 0:
#         return {"tau": np.nan, "a": np.nan, "used_k": np.array([]), "used_delta": np.array([])}

#     mask = d >= (dmax * float(min_frac_of_max))
#     # also require strictly positive for log
#     mask &= d > 0

#     k_use = np.array(k_values, dtype=np.float64)[mask]
#     d_use = d[mask]

#     if len(d_use) < 2:
#         return {"tau": np.nan, "a": np.nan, "used_k": k_use, "used_delta": d_use}

#     # Fit y = a + b*k where y = log(d), b = -1/tau
#     y = np.log(d_use)
#     X = np.vstack([np.ones_like(k_use), k_use]).T
#     a, b = np.linalg.lstsq(X, y, rcond=None)[0]

#     tau = np.inf if b >= 0 else (-1.0 / b)
#     return {"tau": float(tau), "a": float(a), "used_k": k_use, "used_delta": d_use}
