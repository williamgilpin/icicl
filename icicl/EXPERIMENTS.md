# Experiments

Run all commands from the repository root. Setting `REPLICATE = True` at the top of a script makes it use the train/test pairs and seeds in `data/pair_table_100.csv`.

## Order

Each training script simulates the trajectories it needs with dysts, trains models on Train-ID trajectories, and evaluates them on Test-OOD trajectories from a different system.

1. `uv run icicl/train_models.py` trains one transformer per Train-ID/Test-OOD pair. It writes to `local_data/training_run/<train>_<test>/` and skips pairs whose folder exists:
2. `uv run python -m icicl.sweep_probabilities` (uses the trained models from step 1). This reads the run folders above and estimates transfer operators. It writes one array per pair and metric to `local_data/<metric>_<train>_<test>2.pkl`, with one entry per checkpoint.
3. `uv run icicl/train_scaling_laws.py` (independent of other steps) trains each across a range of context lengths. It writes to `local_data/scaling_law_vocab<V>/<time>_<train>_<test>/`.
4. `uv run icicl/scaling_law_data_sweep.py` (independent of other steps) trains each pair at context length 128 with logarithmically spaced numbers of training tokens. It writes to `local_data/scaling_law_data_vocab<V>/<time>_<train>_<test>/`, with the same files under `ntrain<N>/`, named `*_context128_ntrain<N>`.

`train_models.py` always simulates new trajectories. Both scaling-law scripts first reuse an existing `traj_<system>.pkl` found under the working directory if any are available, and otherwise simulate and save a new trajectory. `<V>` is the `--vocab-size` argument.

## Outputs and files

- `traj_<train>.pkl`, `traj_<test>.pkl` correspond to Train-ID and Test-OOD trajectories
- `ckpt_<step>.pt` correspond to saved checkpoints every 1000 steps
- `losses_<time>_<train>_<test>.npz` correspond to Train-ID, Test-ID and Test-OOD loss every 100 steps
- `tiny_lm_<time>_<train>_<test>.pt` represents the final model checkpoint

