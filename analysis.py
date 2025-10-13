import torch
from typing import List, Literal, Optional

def attention_flow(
    attns: List[torch.Tensor],
    *,
    head_reduce: Literal["mean","sum","max","weighted"]="mean",
    head_weights: Optional[torch.Tensor]=None,
    add_residual: float = 1.0,
    renormalize_rows: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute attention flow (Abnar & Zuidema, 2020) from a list of per-layer attention maps.

    Args:
        attns (list[Tensor]): List of L attention tensors, each of shape:
            - [B, H, T, T] or [H, T, T] or [T, T].
            Entries are row-stochastic over keys (last dim).
        head_reduce ({"mean","sum","max","weighted"}): How to combine heads per layer.
        head_weights (Tensor, optional): If head_reduce=="weighted", weights of shape [B,H] or [H].
        add_residual (float): α in Ã = α·I + A_head_reduced. Use 1.0 to include the residual path.
        renormalize_rows (bool): If True, row-normalize Ã to keep it stochastic after adding I.
        eps (float): Numerical floor for normalization.

    Returns:
        Tensor: Flow matrix F of shape [B, T, T] (or [T, T] if inputs were batched-less),
            equal to the left-to-right product of per-layer Ã matrices.
    """
    # Normalize inputs to [B,H,T,T]
    proc = []
    B = None
    for A in attns:
        if A.dim() == 2:  # [T,T]
            A = A.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        elif A.dim() == 3:  # [H,T,T]
            A = A.unsqueeze(0)               # [1,H,T,T]
        elif A.dim() != 4:
            raise ValueError("Each attention map must be [B,H,T,T], [H,T,T], or [T,T].")
        proc.append(A)

    B = proc[0].shape[0]
    T = proc[0].shape[-1]

    def reduce_heads(A: torch.Tensor) -> torch.Tensor:
        # A: [B,H,T,T]
        if head_reduce == "mean":
            return A.mean(dim=1)              # [B,T,T]
        if head_reduce == "sum":
            return A.sum(dim=1)
        if head_reduce == "max":
            return A.max(dim=1).values
        if head_reduce == "weighted":
            if head_weights is None:
                raise ValueError("head_weights must be provided for 'weighted'.")
            w = head_weights
            if w.dim() == 1:  # [H] -> [B,H]
                w = w.unsqueeze(0).expand(A.size(0), -1)
            w = w / (w.sum(dim=1, keepdim=True) + eps)
            return (A * w[:, :, None, None]).sum(dim=1)
        raise ValueError("Unknown head_reduce.")

    # Build per-layer augmented matrices Ã_l
    A_tildes = []
    I = torch.eye(T, device=proc[0].device, dtype=proc[0].dtype).expand(B, T, T)
    for A in proc:
        Abar = reduce_heads(A)                # [B,T,T]
        Atil = add_residual * I + Abar        # [B,T,T]
        if renormalize_rows:
            row_sum = Atil.sum(dim=-1, keepdim=True).clamp_min(eps)
            Atil = Atil / row_sum
        A_tildes.append(Atil)

    # Left-to-right chain product: F = Ã_L · Ã_{L-1} · ... · Ã_1  (token mixing forward)
    F = torch.eye(T, device=proc[0].device, dtype=proc[0].dtype).expand(B, T, T).clone()
    for Atil in A_tildes:
        F = torch.bmm(Atil, F)
    return F.squeeze(0) if F.shape[0] == 1 else F


# --- Example ---
# attns = [layer0_attn, layer1_attn, ...] each [B,H,T,T] with rows ~softmax over T
# F = attention_flow(attns, head_reduce="mean", add_residual=1.0)
# For CLS-to-token attribution, take F[:, cls_idx, :] or per-query row(s).


def attention_rollout(attn_list, add_residual: bool = True):
    """
    Compute attention rollout (Chefer et al. / Abnar & Zuidema) for a *causal* stack.

    Args:
        attn_list : list of (B,T,T) tensors
        add_residual : bool
            If True, augment each layer's attention with the identity and renormalize.

    Returns:
        rollout (B,T,T) tensor: Row-stochastic matrix mapping input tokens
                (cols) -> output tokens (rows).
    """
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






### Dimensionality Estimators

def _stable_singular_values(M: torch.Tensor, center=True, ridge=1e-8, to_double=True):
    """
    Compute singular values of M via symmetric Gram matrix eigendecomposition
    to avoid SVD non-convergence.

    Args:
        M: (..., m, n) tensor
        center (bool): right-multiply by J = I - 1/n 1 1^T (row-centering for rollout)
        ridge (float): adds ridge*I to Gram for numerical stability
        to_double (bool): upcast to float64 during eigensolve
    
    Returns:
        S: (..., min(m,n)) singular values (nonnegative, descending)
    """
    # sanitize
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    if to_double and M.dtype != torch.float64:
        M = M.double()

    # optional row-centering (helps with row-stochastic rollout matrices)
    if center:
        n = M.size(-1)
        I = torch.eye(n, device=M.device, dtype=M.dtype)
        J = I - (1.0 / n) * I.fill_diagonal_(1.0)  # cheaper than 11^T
        # reset I (fill_diagonal_ modified it); rebuild J properly
        I = torch.eye(n, device=M.device, dtype=M.dtype)
        J = I - (1.0/n) * torch.ones(n, n, device=M.device, dtype=M.dtype)
        M = M @ J

    # Gram matrix (… , m, m), symmetrized
    G = M @ M.transpose(-1, -2)
    G = 0.5 * (G + G.transpose(-1, -2))
    if ridge and ridge > 0:
        m = G.size(-1)
        G = G + ridge * torch.eye(m, device=G.device, dtype=G.dtype)

    # Eigenvalues of symmetric PSD G; singular values are sqrt(eigs)
    evals = torch.linalg.eigvalsh(G)            # (..., m), ascending
    evals = torch.clamp(evals, min=0.0)
    S = torch.sqrt(evals.flip(-1))              # descending-like order
    return S


def _safe_normalized_spectrum(S: torch.Tensor, eps=1e-12):
    """Normalize singular values and guard against degenerate spectra."""
    # S: (..., r)
    sums = S.sum(-1, keepdim=True)
    mask_zero = (sums <= eps)
    sums = torch.where(mask_zero, torch.ones_like(sums), sums)
    p = S / sums
    # if all singular values ≈0, define p as one-hot at index 0 to give eRank=1, PR=1
    if mask_zero.any():
        p = p.masked_fill(mask_zero.expand_as(p), 0.0)
        # put probability mass at first coordinate
        idx0 = [slice(None)] * p.ndim
        idx0[-1] = 0
        p[tuple(idx0)] = torch.where(mask_zero.squeeze(-1), torch.tensor(1.0, device=p.device, dtype=p.dtype), p[tuple(idx0)])
    return p


@torch.no_grad()
def erank(A_roll: torch.Tensor, ridge=1e-8, eps=1e-12, center=True):
    """
    Effective rank (entropy) and participation ratio from rollout matrices, robust to SVD failures.

    Args:
        A_roll: (B, T, T) rollout matrix (row-stochastic typical)
        ridge: small ridge added to Gram for stability
        eps: numerical epsilon
        center: apply right-centering to remove rank-1 simplex component

    Returns:
        e_rank: (B,)  = exp( -∑ p_i log p_i )
        pr:     (B,)  = 1 / ∑ p_i^2
    """
    S = _stable_singular_values(A_roll, center=center, ridge=ridge, to_double=True)  # (B, T)
    p = _safe_normalized_spectrum(S, eps=eps)

    H = -(p * torch.clamp(p, min=eps).log()).sum(-1)  # (B,)
    e_rank = torch.exp(H)

    pr = 1.0 / (torch.clamp((p**2).sum(-1), min=eps))

    # Return as float32 on original device
    return e_rank.to(A_roll.dtype), pr.to(A_roll.dtype)

def participation_ratio(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Spectral participation ratio PR = (∑σ)^2 / ∑σ^2,
    computed via a numerically stable eigendecomposition of the Gram matrix.

    Args:
        M: (..., m, n) tensor
        eps: small constant for numerical stability

    Returns:
        (...,) tensor of participation ratios
    """
    # Sanitize inputs and upcast for the solve
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    out_dtype = M.dtype
    if M.dtype != torch.float64:
        M = M.double()

    # Gram matrix (symmetric PSD), with tiny ridge for stability
    G = M @ M.transpose(-1, -2)
    G = 0.5 * (G + G.transpose(-1, -2))
    if eps > 0:
        I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)
        G = G + eps * I

    # Singular values via eigs of Gram: σ_i = sqrt(λ_i(G))
    evals = torch.linalg.eigvalsh(G)                  # (..., m), ascending
    evals = torch.clamp(evals, min=0.0)
    S = torch.sqrt(evals)                             # order irrelevant for sums

    # PR = (∑σ)^2 / ∑σ^2 with safe guards
    S = torch.clamp(S, min=0.0)
    num = (S.sum(-1))**2
    den = (S.square()).sum(-1)
    pr = num / torch.clamp(den, min=eps)

    return pr.to(out_dtype)


def participation_ratio_1d(vals: torch.Tensor):
    vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
    vals = vals / vals.sum()
    return 1.0 / vals.sum()**2