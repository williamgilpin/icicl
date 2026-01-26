"""
Utilities to estimate k-mer transition matrices from autoregressive models.

This module provides several algorithms to compute or approximate the transition
probabilities between k-length token blocks ("k-mers") using a language model
and a corpus of sliding-window contexts. Implementations trade off exactness,
speed, and memory usage, and include Monte Carlo sampling, exact aggregation
over grouped contexts, and faster variants with chunking and optimized indexing.
"""

import torch
import torch.nn.functional as F

@torch.no_grad()
def transition_probs_mc(
    test_tensor: torch.Tensor,       # [B, T] long
    kmers_unique: torch.Tensor,      # [K, k] long
    model,
    shift: int = None,
    n_input_samples: int = 100,
    n_samples_per_input: int = 10,
    temperature: float = 1.0,
    use_compile: bool = False,
    fix_dangling: bool = False,
    verbose: bool = False,
    ground_truth: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Monte Carlo estimator of disjoint k-block transitions.

    Procedure:
      1) Randomly sample contexts (rows) from test_tensor.
      2) Let K_in be the last k tokens of the sampled context.
      3) Autoregressively sample the next k tokens from the model (disjoint block K_out).
      4) Map (K_in, K_out) to indices in kmers_unique and increment a count matrix.
      5) Row-normalize counts to estimate P(K_out | K_in) restricted to kmers_unique.

    Args:
        test_tensor (torch.Tensor): Context windows [B, T] (token ids).
        kmers_unique (torch.Tensor): Candidate k-mers [K, k] (token ids).
        model: Autoregressive model returning logits of shape [N, S, V] for input [N, S].
        shift (int): Number of tokens to shift the sampled context by. Defaults to 
            a full shift of k symbols.
        n_input_samples (int): Number of distinct contexts to sample from test_tensor.
        n_samples_per_input (int): Number of rollouts to sample per chosen context.
        temperature (float): Sampling temperature (1.0 = standard).
        use_compile (bool): If True and supported, wraps model with torch.compile.
        fix_dangling (bool): If True, rows with no counts become uniform over K.
        normalize (bool): If True, row-normalize the transition matrix.

    Returns:
        torch.Tensor: FloatTensor[K, K] row-stochastic transition matrix estimate.
    """
    ## Torch set up and compile
    device = next(model.parameters()).device
    test_tensor = test_tensor.to(device, non_blocking=True)
    kmers_unique = kmers_unique.to(device, non_blocking=True)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    ## Check input sizes
    B, T = test_tensor.shape
    K, k = kmers_unique.shape
    assert k > 1 and T >= k
    assert n_input_samples > 0 and n_samples_per_input > 0

    # ---- hash-based mapping from kmer -> index in kmers_unique ----
    vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
    base = max(vocab_upper + 1, 1024)
    powv = (base ** torch.arange(k, device=device, dtype=torch.long)).view(1, k)
    keys = (kmers_unique.long() * powv).sum(dim=1)  # [K]
    keys_sorted, order = torch.sort(keys)

    def lookup_index(kmer_block: torch.Tensor) -> torch.Tensor:
        """
        Map each k-mer in kmer_block to the index of the closest row in kmers_unique
        under Euclidean distance.

        Args:
            kmer_block (LongTensor[N, k]): k-mers to map.

        Returns:
            LongTensor[N]: indices in [0..K-1] (always defined).
        """
        d = torch.cdist(kmer_block.to(torch.float32), kmers_unique.to(torch.float32))  # [N, K]
        return d.argmin(dim=1)

    # ---- sample input contexts ----
    n_input_samples = min(n_input_samples, B)
    ctx_idx = torch.randint(0, B, (n_input_samples,), device=device)
    ctx = test_tensor[ctx_idx]  # [N, T]
    k_in = ctx[:, -k:]          # [N, k]
    i_idx = lookup_index(k_in)  # [N]

    if shift is None:
        shift = k

    counts = torch.zeros((K, K), device=device, dtype=torch.float32)

    # We'll replicate each context n_samples_per_input times to batch the rollouts.
    N = n_input_samples
    R = n_samples_per_input
    total = N * R

    # Base sequences for rollouts
    seq = ctx.repeat_interleave(R, dim=0).clone()     # [total, T]
    i_rep = i_idx.repeat_interleave(R)               # [total]

    # Autoregressively sample k tokens; we keep length fixed by overwriting last token each step
    # via a sliding window on the right of length T. This respects block_size.
    # At step t: feed current seq, sample next token, then shift seq left and append token.
    out_tokens = torch.empty((total, k), device=device, dtype=torch.long)

    
    if ground_truth:
        ## Get a reference spectrum using exact matching to the context
        ## Build hash keys for prefixes in test_tensor (first k tokens)
        vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
        base = max(vocab_upper + 1, 1024)
        powv = (base ** torch.arange(k, device=device, dtype=torch.long)).view(1, k)
        prefix = test_tensor[:, :k]  # [B, k]
        prefix_key = (prefix.long() * powv).sum(dim=1)  # [B]
        prefix_key_sorted, sort_idx = torch.sort(prefix_key)  # [B], [B]
        # Query keys: input k-grams (use tail of each input seq)
        qin = seq[:, -k:]  # [total, k]
        qkey = (qin.long() * powv).sum(dim=1)  # [total]
        # For each query, find the contiguous matching range in the sorted prefix keys
        left = torch.searchsorted(prefix_key_sorted, qkey, right=False)
        right = torch.searchsorted(prefix_key_sorted, qkey, right=True)
        cnt = right - left  # [total]
        # Randomly pick one matching row per query; fallback to random row if none
        pick = torch.empty_like(left)
        has = cnt > 0
        if has.any():
            offs = (torch.rand(int(has.sum().item()), device=device) * cnt[has].to(torch.float32)).to(torch.long)
            pick[has] = left[has] + offs
        if (~has).any():
            pick[~has] = torch.randint(0, B, (int((~has).sum().item()),), device=device)
        rows = sort_idx[pick]            # [total] indices into test_tensor
        out_tokens[:, :] = test_tensor[rows, k:2 * k]  # [total, k]

    else:
        ## Autoregressively sample k tokens for all inputs
        for t in range(shift):
            logits = model(seq)[:, -1, :]  # [total, V]
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(1)  # [total]
            out_tokens[:, t] = nxt
            # slide window: drop first token, append nxt
            seq = torch.cat([seq[:, 1:], nxt.view(-1, 1)], dim=1)
    
    ## append out_tokens to k_in to form a full sequence
    out_tokens = torch.cat([k_in, out_tokens], dim=1)[:, shift:shift+k]

    j_idx = lookup_index(out_tokens)  # [total]

    ## Increment counts for all valid pairs (i,j)
    valid = (i_rep >= 0) & (j_idx >= 0)
    if valid.any():
        ii = i_rep[valid]
        jj = j_idx[valid]
        counts.index_put_((ii, jj), torch.ones_like(ii, dtype=counts.dtype), accumulate=True)

    # Row-normalize to get probabilities
    row_sum = counts.sum(dim=-1, keepdim=True)
    if fix_dangling:
        dangling = (row_sum.squeeze(-1) == 0)
        probs_out = torch.empty_like(counts)
        if dangling.any():
            probs_out[dangling] = 1.0 / K
        non = ~dangling
        probs_out[non] = counts[non] / row_sum[non]
        return probs_out

    if normalize:
        row_sum = row_sum.clamp_min(1e-30)
        return counts / row_sum
    else:
        return counts



@torch.no_grad()
def transition_probs_mc_greedy_one_step(
    test_tensor: torch.Tensor,       # [B, T] long
    kmers_unique: torch.Tensor,      # [K, k] long
    model,
    temperature: float = 0.0,
    use_compile: bool = False,
    fix_dangling: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """
    A an accelerated version of transition_probs_mc that only samples the next step
    at zero temperature.
    """
    ## Torch set up and compile
    device = next(model.parameters()).device
    test_tensor = test_tensor.to(device, non_blocking=True)
    kmers_unique = kmers_unique.to(device, non_blocking=True)
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    ## Check input sizes
    B, T = test_tensor.shape
    K, k = kmers_unique.shape
    assert k > 1 and T >= k

    next_tokens = model(test_tensor[:, -k:])[:, -1, :]
    next_tokens = next_tokens.argmax(dim=-1)
    prefixes = test_tensor[:, -k:]
    suffixes = torch.cat([prefixes[:, 1:], next_tokens.view(-1, 1)], dim=1)[:, -k:]
    
    # # ---- hash-based mapping from kmer -> index in kmers_unique ----
    # vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
    # base = max(vocab_upper + 1, 1024)
    # powv = (base ** torch.arange(k, device=device, dtype=torch.long)).view(1, k)
    # keys = (kmers_unique.long() * powv).sum(dim=1)  # [K]
    # keys_sorted, order = torch.sort(keys)

    def lookup_index(kmer_block: torch.Tensor) -> torch.Tensor:
        """
        Map each k-mer in kmer_block to the index of the closest row in kmers_unique
        under Euclidean distance.

        Args:
            kmer_block (LongTensor[N, k]): k-mers to map.

        Returns:
            LongTensor[N]: indices in [0..K-1] (always defined).
        """
        d = torch.cdist(kmer_block.to(torch.float32), kmers_unique.to(torch.float32))  # [N, K]
        return d.argmin(dim=1)

    counts = torch.zeros((K, K), device=device, dtype=torch.float32)
    i_idx = lookup_index(prefixes)
    j_idx = lookup_index(suffixes)
    valid = (i_idx >= 0) & (j_idx >= 0)
    ii = i_idx[valid]
    jj = j_idx[valid]
    counts.index_put_((ii, jj), torch.ones_like(ii, dtype=counts.dtype), accumulate=True)

    # Row-normalize to get probabilities
    row_sum = counts.sum(dim=-1, keepdim=True)
    if fix_dangling:
        dangling = (row_sum.squeeze(-1) == 0)
        probs_out = torch.empty_like(counts)
        if dangling.any():
            probs_out[dangling] = 1.0 / K
        non = ~dangling
        probs_out[non] = counts[non] / row_sum[non]
        return probs_out

    if normalize:
        row_sum = row_sum.clamp_min(1e-30)
        return counts / row_sum
    else:
        return counts



@torch.no_grad()
def transition_probs2(
    test_tensor: torch.Tensor,      # [B, T] long
    kmers_unique: torch.Tensor,     # [K, k] long
    model,
    use_compile: bool = False,
    fix_dangling: bool = False,
) -> torch.Tensor:
    """
    Compute transition probabilities by grouping identical tail contexts and
    averaging model next-token distributions for each group.

    Briefly: this groups rows of `test_tensor` by their last k tokens, computes
    the mean next-token distribution per group using the model, and then maps
    those token probabilities to k-mer transitions that are reachable by a
    one-step shift.

    In more detail:
      1) Extract the last k tokens of every context row (the "tail" k-mer) and
         group identical tails with `torch.unique`, producing groups of rows.
      2) Build a fast hash-based join from `kmers_unique` to the tail groups, so
         each start k-mer index i either maps to a group id or is marked missing.
      3) Precompute a mapping from each start k-mer i to the set of next k-mers
         j that are reachable by a one-token shift, i.e. prefix(kmer_j) ==
         suffix(kmer_i). This is implemented via hashing of (k-1)-length
         prefixes/suffixes to avoid O(K^2) comparisons.
      4) For each start k-mer i with matching contexts, run the model on all
         rows in the corresponding group, average the resulting token
         probabilities, and assign those probabilities to the reachable next
         k-mers j using their last token.
      5) Optionally normalize rows and/or fix dangling rows.

    Args:
        test_tensor: Sliding-window inputs ending with a k-mer (token ids), shape [B, T].
        kmers_unique: Unique k-mers (length k), shape [K, k].
        model: Autoregressive model; logits = model(x) with logits[..., vocab].
        use_compile: Optional torch.compile.
        fix_dangling: If True, rows with no matching contexts become uniform over K.

    Returns:
        FloatTensor[K, K]: Row i = starting k-mer; col j = P(next_kmer=j | current_kmer=i).
        Only k-mers reachable by 1-step shift can get nonzero probability mass.
    """
    device = next(model.parameters()).device
    test_tensor = test_tensor.to(device, non_blocking=True)
    kmers_unique = kmers_unique.to(device, non_blocking=True)

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    B, T = test_tensor.shape
    K, k = kmers_unique.shape
    assert k > 0 and T >= k

    tails = test_tensor[:, -k:]  # [B, k]
    uniq_tails, inv, counts = torch.unique(
        tails, dim=0, return_inverse=True, return_counts=True
    )  # uniq_tails [U, k], inv [B], counts [U]
    U = uniq_tails.size(0)

    # --- build a mapping: exact kmer -> index i in [0..K-1] ---
    # Use a safer matching approach than power-hash if k can be moderate/large.
    # Here we still use a hash but with int64 overflow avoidance by using two mod primes.
    # (If k is small, you can simplify.)
    vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
    base = max(vocab_upper + 1, 1024)

    # two mod primes for low collision risk
    p1 = 2_305_843_009_213_693_951  # < 2^63
    p2 = 2_190_434_095_805_586_177

    powv = base ** torch.arange(k, device=device, dtype=torch.long)  # [k]
    def hash2(x: torch.Tensor):
        # x: [N, k] long
        h1 = (x * powv).remainder(p1).sum(dim=1).remainder(p1)
        h2 = (x * powv).remainder(p2).sum(dim=1).remainder(p2)
        return h1, h2

    h1_u, h2_u = hash2(uniq_tails.long())       # [U]
    h1_k, h2_k = hash2(kmers_unique.long())     # [K]

    # map uniq_tails group id -> starting kmer index i (or -1 if not in kmers_unique)
    # do this by sorting uniq hashes and searching kmers_unique hashes
    # (still a hash-based join; collisions extremely unlikely with two primes)
    hu = torch.stack([h1_u, h2_u], dim=1)  # [U,2]
    hk = torch.stack([h1_k, h2_k], dim=1)  # [K,2]

    hu_sorted, order_u = torch.sort(hu[:, 0])  # sort by first component
    hu2_sorted = hu[order_u, 1]

    # for each kmer_unique i, find matching uniq_tails group gid
    pos = torch.searchsorted(hu_sorted, hk[:, 0])
    inb = pos < U
    # candidate matches where first hash matches
    cand = torch.full((K,), -1, device=device, dtype=torch.long)
    ok = inb & (hu_sorted[pos.clamp(max=U-1)] == hk[:, 0])
    # confirm second hash matches too
    pos_ok = pos[ok]
    ok2 = hu2_sorted[pos_ok] == hk[ok, 1]
    cand[ok.nonzero().view(-1)[ok2]] = order_u[pos_ok[ok2]]

    # Now cand[i] is the group id in uniq_tails for starting kmer i (or -1).
    group_id_for_start_kmer = cand  # [K]

    # --- prepare grouped row indices for each uniq tail group ---
    row_ids = torch.arange(B, device=device, dtype=torch.long)
    sort_idx = torch.argsort(inv)
    row_ids_grouped = row_ids[sort_idx]

    seg_starts = torch.empty(U + 1, device=device, dtype=torch.long)
    seg_starts[0] = 0
    seg_starts[1:] = torch.cumsum(counts.to(torch.long), dim=0)

    # --- precompute mapping: (suffix = kmer[1:]) + token -> next kmer index ---
    # For each start kmer i and each possible next token x, next_kmer = [suffix_i, x].
    # We can implement this as: for each i, among kmers_unique, the ones that share prefix==suffix_i
    # and their last token gives the token->j map.
    suffix = kmers_unique[:, 1:]               # [K, k-1]
    prefix = kmers_unique[:, :-1]              # [K, k-1]
    last_tok = kmers_unique[:, -1]             # [K]

    # For each i, we want all j such that prefix[j] == suffix[i], and then map last_tok[j] -> j.
    # Build a dictionary-like structure via hashing k-1 sequences.
    if k == 1:
        # special case: suffix/prefix empty; next kmer is just token itself
        # so mapping is token->j directly where kmers_unique[j,0]=token
        token_to_j = torch.full((int(vocab_upper),), -1, device=device, dtype=torch.long)
        token_to_j[last_tok] = torch.arange(K, device=device)
        # transition row i uses same token_to_j for all i
    else:
        powv2 = base ** torch.arange(k-1, device=device, dtype=torch.long)
        def hash2_km1(x: torch.Tensor):
            h1 = (x * powv2).remainder(p1).sum(dim=1).remainder(p1)
            h2 = (x * powv2).remainder(p2).sum(dim=1).remainder(p2)
            return h1, h2

        h1_pref, h2_pref = hash2_km1(prefix.long())   # [K]
        h1_suf,  h2_suf  = hash2_km1(suffix.long())   # [K]

        # group kmers by prefix hash, then for each suffix hash look up the matching group
        hp = torch.stack([h1_pref, h2_pref], dim=1)
        hs = torch.stack([h1_suf,  h2_suf ], dim=1)

        hp_sorted, order_p = torch.sort(hp[:, 0])
        hp2_sorted = hp[order_p, 1]
        pos2 = torch.searchsorted(hp_sorted, hs[:, 0])
        ok = (pos2 < K) & (hp_sorted[pos2.clamp(max=K-1)] == hs[:, 0])
        i_match = torch.full((K,), -1, device=device, dtype=torch.long)
        pos_ok = pos2[ok]
        ok2 = hp2_sorted[pos_ok] == hs[ok, 1]
        i_match[ok.nonzero().view(-1)[ok2]] = order_p[pos_ok[ok2]]
        # i_match[i] gives *one* j with matching prefix; but multiple j share that prefix.
        # We'll instead build per-prefix buckets by sorting hp (not just one element).

        # Build buckets for each distinct prefix hash pair
        # Sort by (h1,h2) lexicographically
        hp_lex = hp[:, 0] * 0 + hp[:, 0]  # alias
        # Use stable sort on h2 within equal h1 by doing a combined key with large multiplier mod
        # Simpler: use argsort on two keys
        order_lex = torch.argsort(hp[:, 0] * 1_000_000_007 + hp[:, 1])
        hp_sorted2 = hp[order_lex]  # [K,2]
        last_tok_sorted = last_tok[order_lex]
        j_sorted = torch.arange(K, device=device)[order_lex]

        # Identify segment starts for each distinct hp pair
        change = torch.ones(K, device=device, dtype=torch.bool)
        change[1:] = (hp_sorted2[1:, 0] != hp_sorted2[:-1, 0]) | (hp_sorted2[1:, 1] != hp_sorted2[:-1, 1])
        seg_idx = torch.cumsum(change.to(torch.long), dim=0) - 1  # [K], segment id
        nseg = int(seg_idx.max().item()) + 1

        seg_start = torch.full((nseg + 1,), K, device=device, dtype=torch.long)
        seg_start[0] = 0
        # starts: first index where seg id == s
        starts = torch.nonzero(change).view(-1)
        seg_start[: starts.numel()] = starts
        seg_start = torch.cat([starts, torch.tensor([K], device=device)])  # [nseg+1]

        # Map suffix hash pair -> segment id (or -1)
        # Build lookup by sorting unique segment keys
        seg_keys = hp_sorted2[starts]  # [nseg,2]
        seg_h1 = seg_keys[:, 0]
        seg_h2 = seg_keys[:, 1]
        seg_h1_sorted, seg_order = torch.sort(seg_h1)
        seg_h2_sorted = seg_h2[seg_order]
        seg_keys_sorted = seg_keys[seg_order]

        pos3 = torch.searchsorted(seg_h1_sorted, hs[:, 0])
        ok = (pos3 < nseg) & (seg_h1_sorted[pos3.clamp(max=nseg-1)] == hs[:, 0])
        seg_for_i = torch.full((K,), -1, device=device, dtype=torch.long)
        pos_ok = pos3[ok]
        ok2 = seg_h2_sorted[pos_ok] == hs[ok, 1]
        seg_for_i[ok.nonzero().view(-1)[ok2]] = seg_order[pos_ok[ok2]]

    # --- compute transition probabilities ---
    out_probs = torch.zeros((K, K), device=device, dtype=torch.float32)

    for i in range(K):
        gid = int(group_id_for_start_kmer[i].item())
        if gid < 0:
            continue

        m0 = int(seg_starts[gid].item())
        m1 = int(seg_starts[gid + 1].item())
        rows = row_ids_grouped[m0:m1]
        M = rows.numel()
        if M == 0:
            continue

        logits = model(test_tensor[rows])[:, -1]           # [M, V]
        p_tok = F.softmax(logits, dim=-1)                  # [M, V]
        p_tok_mean = p_tok.mean(dim=0)                     # [V] consensus over examples

        if k == 1:
            # next kmer index j is token itself if present in kmers_unique
            j = token_to_j
            valid = j >= 0
            out_probs[i, j[valid]] = p_tok_mean[valid]
        else:
            s = int(seg_for_i[i].item())
            if s < 0:
                continue
            a = int(seg_start[s].item())
            b = int(seg_start[s + 1].item())
            # candidate next-kmers j in this bucket
            j_cand = j_sorted[a:b]               # [Ncand]
            tok_cand = last_tok_sorted[a:b]      # [Ncand]
            out_probs[i, j_cand] = p_tok_mean[tok_cand]

    if fix_dangling:
        row_sum = out_probs.sum(dim=-1, keepdim=True)
        dangling = (row_sum.squeeze(-1) == 0)
        if dangling.any():
            out_probs[dangling] = 1.0 / K
        else:
            out_probs = out_probs / row_sum
        return out_probs

    # normalize rows (should already sum to <=1; exactly 1 if all reachable kmers are included)
    row_sum = out_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return out_probs / row_sum

# @torch.no_grad()
# def transition_probs2(
#     test_tensor: torch.Tensor,      # [B, T] long
#     kmers_unique: torch.Tensor,     # [K, k] long
#     model,
#     use_compile: bool = False,
#     fix_dangling: bool = False,
# ) -> torch.Tensor:
#     """
#     Args:
#         test_tensor: Sliding-window inputs ending with a k-mer (token ids), shape [B, T].
#         kmers_unique: Unique k-mers (length k), shape [K, k].
#         model: Autoregressive model; logits = model(x) with logits[..., vocab].
#         use_compile: Optional torch.compile.
#         fix_dangling: If True, rows with no matching contexts become uniform over K.

#     Returns:
#         FloatTensor[K, K]: Row i = starting k-mer; col j = P(next_kmer=j | current_kmer=i).
#         Only k-mers reachable by 1-step shift can get nonzero probability mass.
#     """
#     device = next(model.parameters()).device
#     test_tensor = test_tensor.to(device, non_blocking=True)
#     kmers_unique = kmers_unique.to(device, non_blocking=True)

#     if use_compile and hasattr(torch, "compile"):
#         model = torch.compile(model)

#     B, T = test_tensor.shape
#     K, k = kmers_unique.shape
#     assert k > 1 and T >= k

#     tails = test_tensor[:, -k:]  # [B, k]
#     uniq_tails, inv, counts = torch.unique(
#         tails, dim=0, return_inverse=True, return_counts=True
#     )  # uniq_tails [U, k], inv [B], counts [U]
#     U = uniq_tails.size(0)

#     # --- build a mapping: exact kmer -> index i in [0..K-1] ---
#     # Use a safer matching approach than power-hash if k can be moderate/large.
#     # Here we still use a hash but with int64 overflow avoidance by using two mod primes.
#     # (If k is small, you can simplify.)
#     vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
#     base = max(vocab_upper + 1, 1024)

#     # two mod primes for low collision risk
#     p1 = 2_305_843_009_213_693_951  # < 2^63
#     p2 = 2_190_434_095_805_586_177

#     powv = base ** torch.arange(k, device=device, dtype=torch.long)  # [k]
#     def hash2(x: torch.Tensor):
#         # x: [N, k] long
#         h1 = (x * powv).remainder(p1).sum(dim=1).remainder(p1)
#         h2 = (x * powv).remainder(p2).sum(dim=1).remainder(p2)
#         return h1, h2

#     h1_u, h2_u = hash2(uniq_tails.long())       # [U]
#     h1_k, h2_k = hash2(kmers_unique.long())     # [K]

#     # map uniq_tails group id -> starting kmer index i (or -1 if not in kmers_unique)
#     # do this by sorting uniq hashes and searching kmers_unique hashes
#     # (still a hash-based join; collisions extremely unlikely with two primes)
#     hu = torch.stack([h1_u, h2_u], dim=1)  # [U,2]
#     hk = torch.stack([h1_k, h2_k], dim=1)  # [K,2]

#     hu_sorted, order_u = torch.sort(hu[:, 0])  # sort by first component
#     hu2_sorted = hu[order_u, 1]

#     # for each kmer_unique i, find matching uniq_tails group gid
#     pos = torch.searchsorted(hu_sorted, hk[:, 0])
#     inb = pos < U
#     # candidate matches where first hash matches
#     cand = torch.full((K,), -1, device=device, dtype=torch.long)
#     ok = inb & (hu_sorted[pos.clamp(max=U-1)] == hk[:, 0])
#     # confirm second hash matches too
#     pos_ok = pos[ok]
#     ok2 = hu2_sorted[pos_ok] == hk[ok, 1]
#     cand[ok.nonzero().view(-1)[ok2]] = order_u[pos_ok[ok2]]

#     # Now cand[i] is the group id in uniq_tails for starting kmer i (or -1).
#     group_id_for_start_kmer = cand  # [K]

#     # --- prepare grouped row indices for each uniq tail group ---
#     row_ids = torch.arange(B, device=device, dtype=torch.long)
#     sort_idx = torch.argsort(inv)
#     row_ids_grouped = row_ids[sort_idx]

#     seg_starts = torch.empty(U + 1, device=device, dtype=torch.long)
#     seg_starts[0] = 0
#     seg_starts[1:] = torch.cumsum(counts.to(torch.long), dim=0)

#     # --- precompute mapping: (suffix = kmer[1:]) + token -> next kmer index ---
#     # For each start kmer i and each possible next token x, next_kmer = [suffix_i, x].
#     # We can implement this as: for each i, among kmers_unique, the ones that share prefix==suffix_i
#     # and their last token gives the token->j map.
#     suffix = kmers_unique[:, 1:]               # [K, k-1]
#     prefix = kmers_unique[:, :-1]              # [K, k-1]
#     last_tok = kmers_unique[:, -1]             # [K]

#     # For each i, we want all j such that prefix[j] == suffix[i], and then map last_tok[j] -> j.
#     # Build a dictionary-like structure via hashing k-1 sequences.
#     powv2 = base ** torch.arange(k-1, device=device, dtype=torch.long)
#     def hash2_km1(x: torch.Tensor):
#         h1 = (x * powv2).remainder(p1).sum(dim=1).remainder(p1)
#         h2 = (x * powv2).remainder(p2).sum(dim=1).remainder(p2)
#         return h1, h2

#     h1_pref, h2_pref = hash2_km1(prefix.long())   # [K]
#     h1_suf,  h2_suf  = hash2_km1(suffix.long())   # [K]

#     # group kmers by prefix hash, then for each suffix hash look up the matching group
#     hp = torch.stack([h1_pref, h2_pref], dim=1)
#     hs = torch.stack([h1_suf,  h2_suf ], dim=1)

#     hp_sorted, order_p = torch.sort(hp[:, 0])
#     hp2_sorted = hp[order_p, 1]
#     pos2 = torch.searchsorted(hp_sorted, hs[:, 0])
#     ok = (pos2 < K) & (hp_sorted[pos2.clamp(max=K-1)] == hs[:, 0])
#     i_match = torch.full((K,), -1, device=device, dtype=torch.long)
#     pos_ok = pos2[ok]
#     ok2 = hp2_sorted[pos_ok] == hs[ok, 1]
#     i_match[ok.nonzero().view(-1)[ok2]] = order_p[pos_ok[ok2]]
#     # i_match[i] gives *one* j with matching prefix; but multiple j share that prefix.
#     # We'll instead build per-prefix buckets by sorting hp (not just one element).

#     # Build buckets for each distinct prefix hash pair
#     # Sort by (h1,h2) lexicographically
#     hp_lex = hp[:, 0] * 0 + hp[:, 0]  # alias
#     # Use stable sort on h2 within equal h1 by doing a combined key with large multiplier mod
#     # Simpler: use argsort on two keys
#     order_lex = torch.argsort(hp[:, 0] * 1_000_000_007 + hp[:, 1])
#     hp_sorted2 = hp[order_lex]  # [K,2]
#     last_tok_sorted = last_tok[order_lex]
#     j_sorted = torch.arange(K, device=device)[order_lex]

#     # Identify segment starts for each distinct hp pair
#     change = torch.ones(K, device=device, dtype=torch.bool)
#     change[1:] = (hp_sorted2[1:, 0] != hp_sorted2[:-1, 0]) | (hp_sorted2[1:, 1] != hp_sorted2[:-1, 1])
#     seg_idx = torch.cumsum(change.to(torch.long), dim=0) - 1  # [K], segment id
#     nseg = int(seg_idx.max().item()) + 1

#     seg_start = torch.full((nseg + 1,), K, device=device, dtype=torch.long)
#     seg_start[0] = 0
#     # starts: first index where seg id == s
#     starts = torch.nonzero(change).view(-1)
#     seg_start[: starts.numel()] = starts
#     seg_start = torch.cat([starts, torch.tensor([K], device=device)])  # [nseg+1]

#     # Map suffix hash pair -> segment id (or -1)
#     # Build lookup by sorting unique segment keys
#     seg_keys = hp_sorted2[starts]  # [nseg,2]
#     seg_h1 = seg_keys[:, 0]
#     seg_h2 = seg_keys[:, 1]
#     seg_h1_sorted, seg_order = torch.sort(seg_h1)
#     seg_h2_sorted = seg_h2[seg_order]
#     seg_keys_sorted = seg_keys[seg_order]

#     pos3 = torch.searchsorted(seg_h1_sorted, hs[:, 0])
#     ok = (pos3 < nseg) & (seg_h1_sorted[pos3.clamp(max=nseg-1)] == hs[:, 0])
#     seg_for_i = torch.full((K,), -1, device=device, dtype=torch.long)
#     pos_ok = pos3[ok]
#     ok2 = seg_h2_sorted[pos_ok] == hs[ok, 1]
#     seg_for_i[ok.nonzero().view(-1)[ok2]] = seg_order[pos_ok[ok2]]




#     for i in range(K):
#         gid = int(group_id_for_start_kmer[i].item())
#         if gid < 0:
#             continue

#         m0 = int(seg_starts[gid].item())
#         m1 = int(seg_starts[gid + 1].item())
#         rows = row_ids_grouped[m0:m1]
#         M = rows.numel()
#         if M == 0:
#             continue

#         logits = model(test_tensor[rows])[:, -1]           # [M, V]
#         p_tok = F.softmax(logits, dim=-1)                  # [M, V]
#         p_tok_mean = p_tok.mean(dim=0)                     # [V] consensus over examples

#         s = int(seg_for_i[i].item())
#         if s < 0:
#             continue
#         a = int(seg_start[s].item())
#         b = int(seg_start[s + 1].item())
#         # candidate next-kmers j in this bucket
#         j_cand = j_sorted[a:b]               # [Ncand]
#         tok_cand = last_tok_sorted[a:b]      # [Ncand]
#         out_probs[i, j_cand] = p_tok_mean[tok_cand]

#         # kmer_meta_index = np.argmin(cdist(kmers_predicted, centroid_kgrams), axis=1)


#     if fix_dangling:
#         row_sum = out_probs.sum(dim=-1, keepdim=True)
#         dangling = (row_sum.squeeze(-1) == 0)
#         if dangling.any():
#             out_probs[dangling] = 1.0 / K
#         else:
#             out_probs = out_probs / row_sum
#         return out_probs

#     # normalize rows (should already sum to <=1; exactly 1 if all reachable kmers are included)
#     row_sum = out_probs.sum(dim=-1, keepdim=True).clamp_min(1e-30)
#     return out_probs / row_sum



@torch.inference_mode()
def transition_probs_fast(
    test_tensor: torch.Tensor,
    kmers_unique: torch.Tensor,
    model,
    use_compile: bool = False,
    fix_dangling: bool = False,
    chunk_k: int | None = None,   # optional: split K to reduce peak memory
) -> torch.Tensor:
    """
    Faster variant of `transition_probs` that scores all candidate next k-mers
    per start k-mer using vectorized log-prob accumulation.

    Briefly: for each start k-mer, it finds all matching context rows, then
    computes the log-probability of every candidate next k-mer by running the
    model on appropriately constructed inputs and accumulating token log-probs.
    It uses `inference_mode`, avoids arange-based fancy indexing, and can chunk
    over K to cap memory.

    In more detail:
      1) Extract the last L tokens from each row of `test_tensor` and group
         identical tails. A hash join maps each `kmers_unique` row to a tail
         group id (or -1 if unseen).
      2) For each start k-mer i, gather the rows whose tail equals that k-mer.
         These rows provide the contexts for estimating P(next_kmer | kmer_i).
      3) For each candidate next k-mer j (optionally in chunks), construct model
         inputs for each position t=0..L-1 by splicing the context suffix with
         the length-t prefix of k-mer j. This produces inputs that correspond to
         autoregressive scoring of the k-mer tokens.
      4) Run the model to obtain logits, take log-softmax, and gather the
         log-prob for the required token at each position t. Sum across t to get
         log P(kmer_j | context row).
      5) Aggregate across context rows using log-mean-exp to obtain the final
         log probability for (i, j), then softmax across j to form a row of the
         transition matrix. Optional `fix_dangling` replaces empty rows with
         uniform probabilities.

    Args:
        test_tensor: Sliding-window inputs ending with a k-mer (token ids), shape [B, T].
        kmers_unique: Unique k-mers (length k), shape [K, k].
        model: Autoregressive next-token model: logits = model(x) with logits[..., vocab].
        use_compile: If True and supported, wraps the model with torch.compile.
        fix_dangling: If True, replace rows with no matched contexts (all -inf logits)
            by a uniform distribution over kmers.
        chunk_k: Optional: split K to reduce peak memory.
        
    Returns:
        (FloatTensor[K, K]): Row k -> starting kmer index; column j -> prob of appending 
            kmers_unique[j].
    """
    device = next(model.parameters()).device
    test_tensor = test_tensor.to(device, non_blocking=True)
    kmers_unique = kmers_unique.to(device, non_blocking=True)

    model.eval()
    if use_compile and hasattr(torch, "compile"):
        # Prefer compiling ONCE outside this function in your code if called repeatedly.
        model = torch.compile(model)

    B, T = test_tensor.shape
    K, L = kmers_unique.shape
    assert L > 0

    tails = test_tensor[:, -L:]  # [B, L]
    uniq_tails, inv, counts = torch.unique(
        tails, dim=0, return_inverse=True, return_counts=True
    )  # uniq_tails: [U, L]

    # Hash tails and kmers to match which kmers appear as tails.
    vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
    base = max(vocab_upper + 1, 1024)
    powv = (base ** torch.arange(L, device=device, dtype=torch.long)).view(1, L)
    key_uniq = (uniq_tails.long() * powv).sum(dim=1)     # [U]
    key_query = (kmers_unique.long() * powv).sum(dim=1)  # [K]

    key_uniq_sorted, order = torch.sort(key_uniq)
    pos = torch.searchsorted(key_uniq_sorted, key_query)
    in_bounds = (pos < key_uniq_sorted.numel()) & (key_uniq_sorted[pos] == key_query)
    group_id_for_kmer = torch.full((K,), -1, device=device, dtype=torch.long)
    group_id_for_kmer[in_bounds] = order[pos[in_bounds]]

    # Group row indices by inv (tail id)
    row_ids = torch.arange(B, device=device, dtype=torch.long)
    sort_idx = torch.argsort(inv)
    U = uniq_tails.size(0)
    seg_starts = torch.empty(U + 1, device=device, dtype=torch.long)
    seg_starts[0] = 0
    seg_starts[1:] = torch.cumsum(counts.to(torch.long), dim=0)
    row_ids_grouped = row_ids[sort_idx]

    out_logprobs = torch.full((K, K), float("-inf"), device=device, dtype=torch.float32)

    # Helper to score one group (rows share the same tail context)
    def score_group(rows: torch.Tensor) -> torch.Tensor:
        M = rows.numel()
        if M == 0:
            return torch.full((K,), float("-inf"), device=device, dtype=torch.float32)

        # accumulate log p over t=0..L-1 for each (row, next-kmer)
        pair_logp = torch.zeros((M, K), device=device, dtype=torch.float32)

        # chunk over K if desired
        k_splits = [(0, K)] if not chunk_k else [
            (s, min(K, s + chunk_k)) for s in range(0, K, chunk_k)
        ]

        for ks, ke in k_splits:
            Kc = ke - ks
            pair_logp_c = torch.zeros((M, Kc), device=device, dtype=torch.float32)

            for t in range(L):
                if t == 0:
                    # inputs: [M*Kc, T]
                    row_part = test_tensor[rows]  # [M, T]
                    inputs_t = row_part.unsqueeze(1).expand(M, Kc, T).reshape(M * Kc, T)
                else:
                    rp = test_tensor[rows, t:]  # [M, T-t]
                    pf = kmers_unique[ks:ke, :t]  # [Kc, t]
                    inputs_t = torch.cat(
                        [
                            rp.unsqueeze(1).expand(M, Kc, T - t).reshape(M * Kc, T - t),
                            pf.unsqueeze(0).expand(M, Kc, t).reshape(M * Kc, t),
                        ],
                        dim=1,
                    )  # [M*Kc, T]

                logits = model(inputs_t)[:, -1]  # [M*Kc, V]
                logp = F.log_softmax(logits, dim=-1)

                # gather token probs without arange-based fancy indexing
                tok = kmers_unique[ks:ke, t].to(torch.long)  # [Kc]
                logp = logp.view(M, Kc, -1).gather(
                    2, tok.view(1, Kc, 1).expand(M, Kc, 1)
                ).squeeze(-1)  # [M, Kc]

                pair_logp_c += logp.to(torch.float32)

            # average over rows (log-mean-exp)
            lme_c = torch.logsumexp(pair_logp_c, dim=0) - torch.log(
                torch.tensor(M, device=device, dtype=torch.float32)
            )
            pair_logp[:, ks:ke] = pair_logp_c  # optional, but keeps structure if you debug
            out = lme_c
            # assemble into full length K vector
            full = torch.full((K,), float("-inf"), device=device, dtype=torch.float32)
            full[ks:ke] = out
            return full

        raise RuntimeError("unreachable")

    # Main loop over starting kmer index (still needed because each k selects different rows)
    # But now inner token selection is faster.
    for k in range(K):
        gid = int(group_id_for_kmer[k].item())
        if gid < 0:
            continue
        m_start, m_end = int(seg_starts[gid].item()), int(seg_starts[gid + 1].item())
        rows = row_ids_grouped[m_start:m_end]
        out_logprobs[k] = score_group(rows)

    if fix_dangling:
        row_all_neginf = torch.isneginf(out_logprobs).all(dim=-1)
        probs = out_logprobs.softmax(dim=-1)
        if row_all_neginf.any():
            probs[row_all_neginf] = 1.0 / K
        return probs

    return out_logprobs.softmax(dim=-1)



@torch.no_grad()
def transition_probs(
    test_tensor: torch.Tensor,
    kmers_unique: torch.Tensor,
    model,
    use_compile: bool = False,
    fix_dangling: bool = False,
) -> torch.Tensor:
    """
    Given a sliding window of tokens, compute the empirical probability of each kmer in 
    kmers_unique given the previous kmer.

    Args:
        test_tensor (LongTensor[batch, T]): Sliding window inputs (token ids).
        kmers_unique (LongTensor[K, L]): Set of "kmer" patterns (assumed fixed L).
        model: Autoregressive next-token model: logits = model(x) with logits[..., vocab].
        use_compile (bool): If True and supported, wraps the model with torch.compile.
        fix_dangling (bool): If True, replace rows with no matched contexts (all -inf logits)
            by a uniform distribution over kmers.

    Returns:
        (FloatTensor[K, K]): Row k -> starting kmer index; column j -> prob of appending kmers_unique[j].
    """
    device = next(model.parameters()).device
    test_tensor = test_tensor.to(device, non_blocking=True)
    kmers_unique = kmers_unique.to(device, non_blocking=True)

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    B, T = test_tensor.shape
    K, L = kmers_unique.shape
    assert L > 0, "kmer length must be > 0"

    tails = test_tensor[:, -L:]  # [B, L]
    uniq_tails, inv, counts = torch.unique(tails, dim=0, return_inverse=True, return_counts=True)

    vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
    base = max(vocab_upper + 1, 1024)
    powv = (base ** torch.arange(L, device=device, dtype=torch.long)).view(1, L)
    key_uniq = (uniq_tails.long() * powv).sum(dim=1)     # [U]
    key_query = (kmers_unique.long() * powv).sum(dim=1)  # [K]

    key_uniq_sorted, order = torch.sort(key_uniq)
    pos = torch.searchsorted(key_uniq_sorted, key_query)
    in_bounds = (pos < key_uniq_sorted.numel()) & (key_uniq_sorted[pos] == key_query)
    group_id_for_kmer = torch.full((K,), -1, device=device, dtype=torch.long)
    group_id_for_kmer[in_bounds] = order[pos[in_bounds]]

    row_ids = torch.arange(B, device=device, dtype=torch.long)
    sort_idx = torch.argsort(inv)
    U = uniq_tails.size(0)
    seg_starts = torch.empty(U + 1, device=device, dtype=torch.long)
    seg_starts[0] = 0
    seg_starts[1:] = torch.cumsum(counts.to(torch.long), dim=0)
    row_ids_grouped = row_ids[sort_idx]

    out_logprobs = torch.full((K, K), float("-inf"), device=device)

    for k in range(K):
        gid = int(group_id_for_kmer[k].item())
        if gid < 0:
            continue

        m_start, m_end = int(seg_starts[gid].item()), int(seg_starts[gid + 1].item())
        rows = row_ids_grouped[m_start:m_end]  # [M]
        M = rows.numel()
        if M == 0:
            continue

        pair_logp = torch.zeros(M, K, device=device)

        for t in range(L):
            # last (T - t) tokens
            if t == 0:
                row_part = test_tensor[rows]              # [M, T]
                inputs_t = row_part.repeat_interleave(K, dim=0)  # [M*K, T]
            else:
                row_part = test_tensor[rows, t:]          # [M, T - t]
                prefix = kmers_unique[:, :t]              # [K, t]
                rp = row_part.repeat_interleave(K, dim=0) # [M*K, T - t]
                pf = prefix.repeat(M, 1)                  # [M*K, t]
                inputs_t = torch.cat([rp, pf], dim=1)     # [M*K, T]

            logits_t = model(inputs_t)[:, -1]             # [M*K, V]
            logp_t = F.log_softmax(logits_t, dim=-1)      # [M*K, V]
            tok_t = kmers_unique[:, t].repeat(M)          # [M*K]
            # pair_logp.view(-1).add_(logp_t[torch.arange(M * K, device=device), tok_t])
            pair_logp += logp_t[torch.arange(M*K, device=device), tok_t].view(M, K)


        lme = torch.logsumexp(pair_logp, dim=0) - torch.log(
            torch.tensor(M, device=device, dtype=pair_logp.dtype)
        )
        # lme = pair_logp.mean(dim=0)
        out_logprobs[k] = lme

    if fix_dangling:
        # Rows with no evidence remain all -inf; set to uniform in prob space.
        row_all_neginf = torch.isneginf(out_logprobs).all(dim=-1)  # [K]
        probs = out_logprobs.softmax(dim=-1)
        if row_all_neginf.any():
            probs[row_all_neginf] = 1.0 / K
        return probs

    return out_logprobs.softmax(dim=-1)



@torch.no_grad()
def transition_probs_chunked(
    test_tensor: torch.Tensor,
    kmers_unique: torch.Tensor,
    model,
    max_pairs_per_batch: int = 200_000,
    use_autocast: bool = True,
    fix_dangling: bool = False,
) -> torch.Tensor:
    """
    Given a sliding window of tokens, compute the empirical probability of each kmer in kmers_unique given the previous kmer.

    Args:
        test_tensor (LongTensor[batch, T]): Sliding-window token ids.
        kmers_unique (LongTensor[K, L]): Distinct kmers (assumes fixed L).
        model: Next-token LM; returns logits = model(x) with logits[..., vocab].
        max_pairs_per_batch (int): Upper bound on (#rows_matched x #kmer2_chunk).
        use_autocast (bool): Enable torch.autocast("cuda") for activation savings.
        fix_dangling (bool): If True, replace rows with no matched contexts by uniform distribution.

    Returns:
        (FloatTensor[K, K]): Row k -> start kmer index; col j -> P(kmer2=j | start=k).
    """
    device = next(model.parameters()).device
    test_tensor = test_tensor.to(device, non_blocking=True)
    kmers_unique = kmers_unique.to(device, non_blocking=True)

    B, T = test_tensor.shape
    K, L = kmers_unique.shape
    assert L > 0

    tails = test_tensor[:, -L:]
    uniq_tails, inv, counts = torch.unique(tails, dim=0, return_inverse=True, return_counts=True)

    vocab_upper = int(max(test_tensor.max().item(), kmers_unique.max().item())) + 1
    base = max(vocab_upper + 1, 1024)
    powv = (base ** torch.arange(L, device=device, dtype=torch.long)).view(1, L)
    key_uniq = (uniq_tails.long() * powv).sum(dim=1)
    key_query = (kmers_unique.long() * powv).sum(dim=1)

    key_uniq_sorted, order = torch.sort(key_uniq)
    pos = torch.searchsorted(key_uniq_sorted, key_query)
    in_bounds = (pos < key_uniq_sorted.numel()) & (key_uniq_sorted[pos] == key_query)
    group_id_for_kmer = torch.full((K,), -1, device=device, dtype=torch.long)
    group_id_for_kmer[in_bounds] = order[pos[in_bounds]]

    row_ids = torch.arange(B, device=device, dtype=torch.long)
    sort_idx = torch.argsort(inv)
    U = uniq_tails.size(0)
    seg_starts = torch.empty(U + 1, device=device, dtype=torch.long)
    seg_starts[0] = 0
    seg_starts[1:] = torch.cumsum(counts.to(torch.long), dim=0)
    row_ids_grouped = row_ids[sort_idx]

    out_logprobs = torch.full((K, K), float("-inf"), device=device)

    amp_ctx = (
        torch.autocast("cuda")
        if (use_autocast and device.type == "cuda")
        # else torch.cuda.amp.autocast(enabled=False)
        else torch.amp.autocast('cuda', enabled=False)
    )

    for k in range(K):
        gid = int(group_id_for_kmer[k].item())
        if gid < 0:
            continue

        m0, m1 = int(seg_starts[gid].item()), int(seg_starts[gid + 1].item())
        rows = row_ids_grouped[m0:m1]
        M = rows.numel()
        if M == 0:
            continue

        pair_logp = torch.zeros(M, K, device=device)

        for t in range(L):
            row_part = test_tensor[rows] if t == 0 else test_tensor[rows, t:]

            K_chunk = max(1, min(K, max_pairs_per_batch // max(1, M)))

            for j0 in range(0, K, K_chunk):
                j1 = min(K, j0 + K_chunk)
                Kc = j1 - j0

                if t == 0:
                    inputs_t = row_part.repeat_interleave(Kc, dim=0)
                else:
                    prefix = kmers_unique[j0:j1, :t]
                    rp = row_part.repeat_interleave(Kc, dim=0)
                    pf = prefix.repeat(M, 1)
                    inputs_t = torch.cat([rp, pf], dim=1)

                with amp_ctx:
                    logits_t = model(inputs_t)[:, -1]

                # safer selective log-softmax (avoid inf-inf under autocast by doing reductions in fp32)
                z = torch.logsumexp(logits_t.float(), dim=-1)  # [M*Kc]
                tok_t = kmers_unique[j0:j1, t].repeat(M)
                idx = torch.arange(M * Kc, device=device)
                s = logits_t[idx, tok_t].float() - z
                pair_logp[:, j0:j1] += s.view(M, Kc)

        out_logprobs[k] = torch.logsumexp(pair_logp, dim=0) - torch.log(
            torch.tensor(M, device=device, dtype=pair_logp.dtype)
        )

    probs = out_logprobs.softmax(dim=-1)
    if fix_dangling:
        row_all_neginf = torch.isneginf(out_logprobs).all(dim=-1)
        if row_all_neginf.any():
            probs[row_all_neginf] = 1.0 / K
    return probs
