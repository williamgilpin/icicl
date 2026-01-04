import torch
import torch.nn.functional as F

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
