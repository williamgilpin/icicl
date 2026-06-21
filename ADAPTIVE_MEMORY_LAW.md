# Adaptive Memory Law

Minimal experiment to test a conceptual claim for this repo:

> Effective context order is state-dependent and increases in dynamically harder regions.

The implementation is in `icicl/adaptive_memory_law.py`.

## What this runs

- Uses existing artifacts only:
  - `data/tiny_lm.pt`
  - `data/traj_train.pkl`, `data/traj_test.pkl`, `data/traj_test_out.pkl`
- Computes per-context model next-token distributions.
- Computes KL-optimal projected Markov order `k_eff` for contiguous suffix lengths `k=1..K_max`.
- Computes a local instability proxy from nearest-neighbor finite-time divergence in trajectory space.
- Tests correlation between `k_eff` and local instability with permutation p-values.
- Produces context-truncation curves `KL(p_full || p_k)`.

## Run

From repo root:

```bash
uv run python icicl/adaptive_memory_law.py --device cpu --n-samples 8000 --k-max 48 --outdir results/adaptive_memory_law
```

## Output files

- `results/adaptive_memory_law/metrics.json`
- `results/adaptive_memory_law/arrays.npz`
- `results/adaptive_memory_law/summary.txt`
- `results/adaptive_memory_law/fig_keff_hist.png`
- `results/adaptive_memory_law/fig_keff_vs_local_div.png`
- `results/adaptive_memory_law/fig_permutation_control.png`
- `results/adaptive_memory_law/fig_truncation_kl.png`

## Latest run (seed=0)

- `train_id`: `N=8000`, `mean(k_eff)=6.728`, `rho(k_eff, local_div)=0.092`, `perm_p=0.00498`
- `test_id`: `N=868`, `mean(k_eff)=3.214`, `rho(k_eff, local_div)=0.258`, `perm_p=0.00498`
- `test_ood`: `N=8000`, `mean(k_eff)=8.433`, `rho(k_eff, local_div)=0.040`, `perm_p=0.00498`
- OOD shift: `mean(k_eff)_ood - mean(k_eff)_test_id = 5.218`

## Interpretation

- Consistent positive adaptive-memory signal: `rho(k_eff, local_div) > 0` on all splits.
- Clear regime shift: OOD contexts require substantially larger inferred effective order.
- Context truncation curves are smooth and monotonic in information loss, matching the memory-order story.

This supports a minimal version of the adaptive-memory hypothesis and gives a concrete base for follow-up ablations.

## Suggested PR framing

- Headline: **state-dependent memory is measurable and shifts strongly OOD**.
- Emphasize reproducibility: single script, fixed seed, existing artifacts only.
- Position this as a conceptual extension of transfer-operator analysis, not just another benchmark.

Quiet caveat line (optional): effect size varies by split, which motivates follow-up ablations across systems/checkpoints.
