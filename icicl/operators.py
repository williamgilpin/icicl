import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy

def predictive_information(P, base=np.e):
    """
    Compute predictive information I(S_t; S_{t+1}) from a row-stochastic matrix.

    Args:
        P (array_like): Row-stochastic transition matrix of shape (n, n).
        base (float, optional): Logarithm base. Use np.e for nats, 2 for bits.

    Returns:
        float: Predictive information I(S_t; S_{t+1}) in units set by `base`.
    """
    
    n = P.shape[0]
    pi = invariant_distribution(P)
    row_sums = P.sum(axis=1)

    P = np.clip(P, 0.0, 1.0)

    # Next-state marginal q = pi P
    q = pi @ P

    I_nats = entropy(q) - np.sum([pi[i] * entropy(P[i, :]) for i in range(n)])
    I = I_nats / np.log(base)
    # Numerical guard against tiny negative due to floating error
    return float(max(I, -1e-12))


# def entropy_rate(P, base=np.e):
#     """
#     Compute the entropy rate h of a Markov chain from a row-stochastic transition matrix.

#     Args:
#         P (np.ndarray): Row-stochastic transition matrix (n, n).
#         base (float, optional): Log base (np.e for nats, 2 for bits).

#     Returns:
#         float: Entropy rate h.
#     """
#     P = np.asarray(P, dtype=float)
#     pi = invariant_distribution(P)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         logP = np.log(P) / np.log(base)
#     row_ent = -np.nansum(P * logP, axis=1)  # 0*log 0 treated as nan -> ignored
#     return float(pi @ row_ent)
def entropy_rate(P, base=np.e):
    """
    Args:
        P (np.ndarray): Row-stochastic transition matrix (n, n).
        base (float, optional): Log base (np.e for nats, 2 for bits).

    Returns:
        float: Entropy rate h.
    """
    P = np.asarray(P, dtype=float)
    pi = invariant_distribution(P)

    log_base = np.log(base)
    row_H = -np.sum(np.where(P > 0, P * (np.log(P) / log_base), 0.0), axis=1)

    return float(pi @ row_H)

def stationary_entropy(P):
    """
    Compute the stationary entropy H(S) of a Markov chain from a row-stochastic transition matrix.

    Args:
        P (np.ndarray): Row-stochastic transition matrix (n, n).

    Returns:
        float: Stationary entropy H(S).
    """
    return entropy(invariant_distribution(P))

def invariant_distribution(P, tol=1e-12, max_iter=10_000, fix_dangling=True, seed=None):
    """
    Given a row-stochastic transition matrix P, returns a stationary (invariant) 
    distribution π satisfying π^T P = π^T, sum(π)=1, π>=0. This can be used to compute 
    the long-term behavior of a Markov chain.
    Uses power iteration on P^T. If P has zero-rows (no outgoing transitions), optionally
    makes them uniform (common 'dangling' fix) to ensure stochasticity.

    Args:
        P (array-like): row-stochastic transition matrix (K,K)
        tol (float): convergence tolerance on ||π_{k+1}-π_k||_1.
        max_iter (int): Maximum number of iterations.
        fix_dangling (bool): If True, replace any all-zero row with uniform distribution.
        seed (int or None): Random seed for initialization.

    Returns:
        pi (array-like): Stationary distribution (K,)
    """
    P = np.asarray(P, dtype=float)
    K = P.shape[0]
    if P.shape != (K, K):
        raise ValueError("P must be square")

    # Handle dangling states (zero-outgoing-prob rows)
    if fix_dangling:
        row_sums = P.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze() == 0)
        if np.any(zero_rows):
            P = P.copy()
            P[zero_rows, :] = 1.0 / K

    # Power iteration on P^T
    rng = np.random.default_rng(seed)
    pi = rng.random(K)
    pi /= pi.sum()
    Pt = P.T
    for _ in range(max_iter):
        new = Pt @ pi
        new_sum = new.sum()
        if new_sum == 0 or not np.isfinite(new_sum):
            raise FloatingPointError("Iteration produced non-finite or zero vector")
        new /= new_sum
        if np.linalg.norm(new - pi, 1) < tol:
            pi = new
            break
        pi = new
    else:
        # Not converged; still return best iterate
        pass
    # Ensure nonnegative and normalized
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi

def leading_transients(P, m=3, tol=1e-10, max_iter=20_000, seed=None):
    """
    Sequentially compute the longest-lived transient *left* modes of a Markov chain
    using simple deflation: at each stage subtract 1*u^T so that the found left
    eigenvector u^T is annihilated in subsequent searches.

    Args:
        P (array-like): Row-stochastic transition matrix (N, N).
        m (int): Number of transient modes to compute (excludes the stationary mode).
        tol (float): L2 tolerance for convergence of each mode.
        max_iter (int): Maximum iterations per mode.
        seed (int | None): RNG seed.

    Returns:
        evals (ndarray): Length-m array of eigenvalue estimates |λ| in non-increasing order.
        evecs (ndarray): (m, N) array; evecs[j] is the j-th transient left mode (L2-normalized, sum≈0).
        pi (ndarray): Stationary distribution (N,).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if P.shape != (n, n): raise ValueError("P must be square")
    if not (1 <= m <= n - 1): raise ValueError("m must be in [1, N-1]")

    # 1) Fix dangling and get π
    pi = invariant_distribution(P, fix_dangling=True, seed=seed)

    # 2) Build deflated operator that kills π; then iteratively deflate found transients
    Pdef = deflated_operator(P, pi)  # P - 1*pi^T
    Pt = P.T

    rng = np.random.default_rng(seed)
    evals, modes = [], []

    ones = np.ones(n)

    def project_stable(v):
        # Remove any leakage back into π and enforce zero-sum (left transient modes satisfy u^T 1 = 0)
        v = v - (v @ pi) * pi
        v = v - (v.sum() / n) * ones
        return v

    for _ in range(m):
        # Initialize in the zero-sum, π-orthogonal subspace
        v = rng.normal(size=n)
        v = project_stable(v)
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv): v = rng.normal(size=n); v = project_stable(v); nv = np.linalg.norm(v)
        v /= nv

        # Power iteration on (Pdef)^T to find current dominant transient left mode
        Pt_def = Pdef.T
        prev = v.copy()
        for _it in range(max_iter):
            w = Pt_def @ v
            nw = np.linalg.norm(w)
            if nw == 0 or not np.isfinite(nw): raise FloatingPointError("Deflated power iteration became singular")
            v = w / nw
            v = project_stable(v)
            nv = np.linalg.norm(v)
            if nv == 0 or not np.isfinite(nv): raise FloatingPointError("Projection produced degenerate vector")
            v /= nv
            if np.linalg.norm(v - prev) < tol: break
            prev = v

        # Stabilize sign (deterministic)
        imax = np.argmax(np.abs(v))
        if v[imax] < 0: v = -v

        # Eigenvalue estimate from original P (Rayleigh quotient for left vector)
        lam = float(v @ (Pt @ v))

        evals.append(lam)
        modes.append(v.copy())

        # 3) Deflate this mode so the next search is self-consistent
        Pdef = deflated_operator(Pdef, v)  # cumulative: P - 1*pi^T - sum 1*u_i^T

    return np.asarray(evals), np.vstack(modes), pi

def deflated_operator(P, pi):
    """
    A deflated operator that removes a trivial mode from a row-stochastic matrix

    Args:
        P: (N, N) array-like, row-stochastic ideally
        pi: (N,) np.ndarray, stationary distribution

    Returns:
        Ptil: (N, N) sparse.csr_matrix, deflated operator
    """
    n = P.shape[0]
    return P - np.outer(np.ones(n), pi)


def transition_matrix(seq, vocab_size=None, tau=1, normalize=True, dtype=float, remap=True):
    """
    Estimate the lag-τ Markov transition matrix P where
    P[i, j] = Pr(s_t = j | s_{t-τ} = i) from an integer sequence.

    Args:
        seq (Sequence[int]): Observed states.
        vocab_size (int | None): Number of possible states. If None and remap=True, inferred as
            number of unique states in seq after remapping. If None and remap=False, inferred as
            max(seq)+1.
        tau (int, optional): Lag between conditioning and target (τ ≥ 1).
        normalize (bool, optional): If True, return probabilities; if False, return raw counts.
        dtype (type, optional): dtype for the returned matrix when normalize=True.
        remap (bool, optional): If True, remap arbitrary integer labels to 0..K-1.

    Returns:
        np.ndarray: (vocab_size, vocab_size) transition matrix (row-stochastic if normalize=True).
    """
    if tau < 1:
        raise ValueError("tau must be ≥ 1")

    seq = np.asarray(seq, dtype=np.int64)
    n = seq.size
    if n <= tau:
        vs = 0 if vocab_size is None else int(vocab_size)
        out_dtype = dtype if normalize else np.int64
        return np.zeros((vs, vs), dtype=out_dtype)

    if remap:
        # Map arbitrary labels (e.g., {7, 241, 999}) -> {0, 1, 2}
        _, seq = np.unique(seq, return_inverse=True)
        if vocab_size is None:
            vocab_size = int(seq.max()) + 1
    else:
        if vocab_size is None:
            vocab_size = int(seq.max()) + 1

    vocab_size = int(vocab_size)

    if seq.min() < 0 or seq.max() >= vocab_size:
        raise ValueError(f"State index out of range: min={seq.min()}, max={seq.max()}, vocab_size={vocab_size}")

    src = seq[:-tau]
    dst = seq[tau:]

    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(counts, (src, dst), 1)

    if not normalize:
        return counts

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = counts / row_sums
    P[row_sums.squeeze() == 0] = 0
    return P.astype(dtype)


# def transition_matrix(seq, vocab_size, tau=1, normalize=True, dtype=float):
#     """
#     Estimate the lag-τ Markov transition matrix P where
#     P[i, j] = Pr(s_t = j | s_{t-τ} = i) from an integer sequence.

#     Args:
#         seq (Sequence[int]): Observed states, each in [0, vocab_size).
#         vocab_size (int): Number of possible states.
#         tau (int, default=1): Lag between conditioning and target (τ ≥ 1).
#         normalize (bool, default=True): If True, return probabilities; if False, return raw counts.
#         dtype (type, default=float): dtype for the returned matrix when normalize=True.

#     Returns:
#         P (np.ndarray, shape (vocab_size, vocab_size)): Row-stochastic transition matrix
#             (or counts if normalize=False). Rows with no outgoing observations are all zeros.
#     """
#     # if tau < 1:
#     #     raise ValueError("tau must be ≥ 1")
#     seq = np.asarray(seq, dtype=np.int64)
#     n = len(seq)
#     if n <= tau:
#         return np.zeros((vocab_size, vocab_size), dtype=(dtype if normalize else np.int64))

#     src = seq[:-tau]   # s_{t-τ}
#     dst = seq[tau:]    # s_t

#     counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
#     np.add.at(counts, (src, dst), 1)  # accumulate counts for (src -> dst)

#     if not normalize:
#         return counts

#     row_sums = counts.sum(axis=1, keepdims=True)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         P = counts / row_sums
#     P[row_sums.squeeze() == 0] = 0  # keep rows with no evidence as zeros
#     return P.astype(dtype)


def transitions_from_time_labels(T, labels, K, tau_time, normalize=True, dtype=float):
    """
    Compute τ-time transitions using explicit timestamps by mapping each index i
    to the first index j with T[j] >= T[i] + tau_time.

    Args:
        T (array-like): Monotonic time stamps (1D, real-valued).
        labels (array-like): Discrete state labels aligned with T, each in [0, K).
        K (int): Number of states.
        tau_time (float): Positive lag in the same units as T.
        normalize (bool, default=True): If True, return probabilities; else raw counts.
        dtype (type, default=float): dtype for probabilities when normalize=True.

    Returns:
        (np.ndarray, shape (K, K)): Transition matrix (row-stochastic if normalize=True).
    """
    T = np.asarray(T, dtype=float).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if T.ndim != 1:
        raise ValueError("T must be 1D.")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D.")
    if len(T) != len(labels):
        raise ValueError("T and labels must have the same length.")
    if not (np.isfinite(tau_time) and tau_time > 0):
        raise ValueError("tau_time must be positive and finite.")
    if not (isinstance(K, int) and K >= 1):
        raise ValueError("K must be a positive integer.")

    counts = np.zeros((K, K), dtype=np.int64)
    idx_after = np.searchsorted(T, T + tau_time, side="left")
    valid = idx_after < len(T)
    src = labels[valid]
    dst = labels[idx_after[valid]]
    counts += np.bincount(src * K + dst, minlength=K * K).reshape(K, K)

    if not normalize:
        return counts
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(row_sums > 0, counts / row_sums, 0.0)
    return P.astype(dtype)


def reduce_markov_chain(P, assignment, weights=None, renormalize=True, atol=1e-12):
    """
    Given a row-stochastic transition matrix P and a mapping of microstates to metastates,
    reduce the matrix by averaging transitions within each metastate.
    
    Args:
        P (np.ndarray): Row-stochastic transition matrix of shape (N, N).
        assignment (array_like): Length-N vector mapping each microstate to a metastate
            (integers; need not be contiguous).
        weights (array_like | None): Optional nonnegative length-N weights for averaging
            rows within each metastate. If None, uses uniform weights within each metastate.
            A common alternative is the stationary distribution of P.
        renormalize (bool): If True, renormalize each row of the reduced matrix to sum to 1.
        atol (float): Small threshold used for numerical safety.

    Returns:
        np.ndarray: Reduced row-stochastic transition matrix of shape (M, M).
            Metastates with zero or near-zero total weight are assigned self-transitions
            (P[a,a] = 1, P[a,b] = 0 for b≠a), representing isolated/absorbing states.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square (N, N) array.")
    N = P.shape[0]

    a = np.asarray(assignment)
    if a.shape[0] != N:
        raise ValueError("assignment must have length N to match P.")

    # Map arbitrary labels -> {0,1,...,M-1}
    labels, inv = np.unique(a, return_inverse=True)
    M = labels.size

    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != N:
            raise ValueError("weights must have length N to match P.")
        if np.any(w < -atol):
            raise ValueError("weights must be nonnegative.")
        w = np.maximum(w, 0.0)

    # Build membership matrix S: S[i, m] = 1 if microstate i is in metastate m
    S = np.zeros((N, M), dtype=float)
    S[np.arange(N), inv] = 1.0

    # For each microstate i and metastate b: Q[i,b] = sum_{j in b} P[i,j]
    Q = P @ S  # shape (N, M)

    # Weighted averaging of rows within each metastate:
    # P_red[a,b] = (sum_{i in a} w_i * Q[i,b]) / (sum_{i in a} w_i)
    P_reduced_num = (w[:, None] * Q).T @ S  # (M,N)@(N,M) -> (M,M) but via transpose trick
    # The above line is a bit opaque; clearer equivalent:
    # P_reduced_num = np.zeros((M, M))
    # for i in range(N): P_reduced_num[inv[i], :] += w[i] * Q[i, :]

    # Initialize the reduced matrix to the identity matrix
    P_reduced = np.identity(M)

    meta_mass = (w[:, None] * S).sum(axis=0)  # length M, meta_mass[a] = sum_{i in a} w_i
    # Identify metastates with zero or near-zero weight (isolated states)
    isolated = meta_mass <= atol
    # Handle normal metastates (those with positive weight)
    valid_mask = ~isolated
    if np.any(valid_mask):
        # Division is safe since we only divide for valid metastates
        P_reduced[valid_mask, :] = P_reduced_num[valid_mask, :] / meta_mass[valid_mask, None]
    

    if renormalize:
        row_sums = P_reduced.sum(axis=1, keepdims=True)
        # If numerical drift produces tiny row_sums, protect division
        row_sums = np.where(row_sums <= atol, 1.0, row_sums)
        P_reduced = P_reduced / row_sums

    return P_reduced


from clustering import UniformGridClusterer

class SymbolicMarkovChain:
    """
    Partition a multivariate time series into K symbols with k-means,
    then estimate a tau-lag Markov transition matrix between symbols.

    Parameters:
        K (int): Number of clusters (symbols).
        tau (float): Time lag. For each t_i, transitions are computed to the first index j
            with T_j >= T_i + tau (if such j exists).
        T (array-like, optional): Time stamps (need not be uniform; any real-valued, sortable type).
        D (array-like, optional): Multivariate observations aligned with T.
        clustering_method (str, optional): Clustering method to use. "kmeans" or "uniform".

    Attributes:
        K (int): Number of clusters (symbols).
        tau (float): Time lag. For each t_i, transitions are computed to the first index j
            with T_j >= T_i + tau (if such j exists).
        kmeans (KMeans): Trained k-means object.
        labels_ (np.ndarray, shape [N,]): Symbol of each sample (time-sorted).
        counts_ (np.ndarray, shape [K, K]): Tau-lag transition counts.
        P_ (np.ndarray, shape [K, K]): Row-stochastic transition probabilities.
        order_ (np.ndarray, shape [N,]): Indices that sort the input times.
        T_sorted_ (np.ndarray, shape [N,]): Sorted times.

    Examples:
        >>> from icicl.operators import SymbolicMarkovChain
        >>> chain = SymbolicMarkovChain(K=10, tau=1)
        >>> chain.fit(D, K, tau, T)
        >>> labels = chain.predict(D)
        >>> counts = chain.counts_
        >>> P = chain.P_
        
    """

    def fit(self, X, K, tau, T=None, clustering_method="kmeans"):
        """
        Fit a SymbolicMarkovChain to a multivariate time series.

        Args:
            X (array-like): Multivariate observations aligned with T.
            K (int): Number of clusters (symbols).
            tau (float): Time lag. For each t_i, transitions are computed to the first index j
                with T_j >= T_i + tau (if such j exists).
            T (array-like, optional): Time stamps (need not be uniform; any real-valued, sortable type).
            clustering_method (str, optional): Clustering method to use. "kmeans" or "uniform".

        Returns:
            self (SymbolicMarkovChain): The fitted SymbolicMarkovChain.
        """
        if T is not None:
            # Validate and sort by time
            T = np.asarray(T).astype(float).ravel()
            if T.ndim != 1:
                raise ValueError("T must be 1D.")
            if len(T) != len(X):
                raise ValueError("T and D must have the same length.")
            if not (isinstance(K, int) and K >= 1 and K <= len(T)):
                raise ValueError("K must be an integer in [1, N].")
            order = np.argsort(T)
            T, X = T[order], X[order]
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, d].")
        if not (np.isfinite(tau) and tau > 0):
            raise ValueError("tau must be a positive finite number.")

        self.K = int(K)
        self.tau = float(tau)
        self.clustering_method = clustering_method


        ## Cluster the data to define symbols
        if self.clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=self.K, n_init=10, random_state=0)
        elif self.clustering_method == "uniform":
            clusterer = UniformGridClusterer(K=self.K)
        else:
            raise ValueError(f"Invalid clustering method: {self.clustering_method}")
        clusterer.fit(X)
        labels = np.asarray(clusterer.labels_)
        self.clusterer = clusterer
        self.labels_ = labels

        ## If times are provided, use them to compute transitions among symbols
        if T is not None:
            P = transitions_from_time_labels(T, labels, self.K, self.tau, normalize=True, dtype=float)
        else:
            ## Otherwise, compute transitions among observations directly
            P = transition_matrix(labels, self.K, tau=int(self.tau), normalize=True, dtype=float)

        self.P_ = P
        return self

    def predict(self, X):
        """
        Predict the labels for a new set of observations.
        """
        return self.clusterer.predict(X)

    def fit_predict(self, X, *args, **kwargs):
        """
        Fit the SymbolicMarkovChain and predict the labels for a new set of observations.
        """
        self.fit(X, *args, **kwargs)
        return self.predict(X)
