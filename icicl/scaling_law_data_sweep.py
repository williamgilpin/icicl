"""
## Scaling-law sweep script

This script trains next-token language models on randomly sampled pairs of continuous
dynamical systems from `dysts`. For each train/test pair, it generates trajectories,
tokenizes the first state variable with `ChronosTokenizer`, trains models across
multiple amounts of pretraining data at fixed context length 128, and saves
checkpoints, losses, forecast metrics, trajectories, and run metadata.

By default, ./private_data/scaling_law_data_vocab{vocab_size} is used as the root
directory for all runs, and all available train/test pairs are sampled. Each pair is
trained with logarithmically spaced training-token counts up to `--n-train`.

### Basic usage

```bash
python scaling_law_data_sweep.py
```

Run with custom seed and training-token counts
```
python scaling_law_data_sweep.py --seed 42 --train-sizes 256 1024 4096 80000 --vocab-size 20
```

"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from dysts.systems import get_attractor_list
import dysts.flows

try:
    from icicl.models import (
        ChronosTokenizer,
        generate_autoregressive,
        save_checkpoint,
        train_next_token,
    )
except ModuleNotFoundError:
    from models import (
        ChronosTokenizer,
        generate_autoregressive,
        save_checkpoint,
        train_next_token,
    )


VOCAB_SIZE = 100
N_TRAIN = 2 * 40_000
N_TEST = 3_000
DEFAULT_CONTEXT_LENGTH = 128
DEFAULT_NUM_TRAIN_SIZES = 10
DEFAULT_STEPS = 60_000
DEFAULT_HORIZON = 60
DEFAULT_BATCH_SIZE = 64 * 2
DEFAULT_D_MODEL = 128 * 2
DEFAULT_D_K = 64
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1.0
DEFAULT_CADENCE = 1000
DEFAULT_SEED = 2
DEFAULT_TORCH_SEED = DEFAULT_SEED
DEFAULT_PTS_PER_PERIOD = 30

from pathlib import Path
def check_for_existing_traj_files(traj_name, min_len=None):
    """
    Given a trajectory name, check for existing trajectory files in the current 
    directory.
    """
    files = [str(item) for item in list(Path(".").rglob("traj_*.pkl"))]
    matching_files = [f for f in files if f"traj_{traj_name}.pkl" in f]
    if min_len is not None:
        valid_files = []
        for f in matching_files:
            try:
                traj = np.load(f, allow_pickle=True)
                if len(traj) >= min_len:
                    valid_files.append(f)
            except Exception as exc:
                print(
                    f"Short or invalid trajectory file {f}: {type(exc).__name__}: {exc}", 
                    flush=True
                )
        matching_files = valid_files
    return matching_files


def mse_loss(xtrue, xpred):
    xtrue = np.asarray(xtrue)
    xpred = np.asarray(xpred)
    diff = xpred - xtrue
    return float((diff ** 2).mean())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train scaling-law sweeps over randomly sampled dynamical-system pairs."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Root directory for all scaling-law runs.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=None,
        help="Number of random train/test pairs to attempt. Defaults to all systems.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Numpy RNG seed used for pair sampling.",
    )
    parser.add_argument(
        "--torch-seed",
        type=int,
        default=DEFAULT_TORCH_SEED,
        help="Torch RNG seed used before each model training run.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Fixed context length used for every training-data sweep point.",
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of training-token counts. Defaults to a logarithmic ramp up to --n-train.",
    )
    parser.add_argument(
        "--num-train-sizes",
        type=int,
        default=DEFAULT_NUM_TRAIN_SIZES,
        help="Number of logarithmically spaced training-token counts when --train-sizes is omitted.",
    )
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--d-k", type=int, default=DEFAULT_D_K)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--cadence", type=int, default=DEFAULT_CADENCE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--pts-per-period", type=int, default=DEFAULT_PTS_PER_PERIOD)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain data-size sweep points even if a final checkpoint already exists.",
    )
    args = parser.parse_args()
    if args.base_path is None:
        args.base_path = Path(f"./private_data/scaling_law_data_vocab{args.vocab_size}")
    return args


def sample_system_pairs(num_pairs, seed):
    all_attractors = np.asarray(get_attractor_list(sys_class="continuous_no_delay"))
    rng = np.random.default_rng(seed)
    training_systems = rng.permutation(all_attractors)
    test_systems = rng.permutation(all_attractors)
    if num_pairs is None:
        num_pairs = len(all_attractors)
    num_pairs = min(num_pairs, len(all_attractors))
    pair_list = list(zip(training_systems[:num_pairs], test_systems[:num_pairs]))
    # pre_pairs = [["Lorenz", "Rossler"], ["SanUmSrisuchinwong", "BelousovZhabotinsky"], ["SprottR", "SprottQ"]]
    # pair_list = pre_pairs + pair_list
    return pair_list


def should_skip_pair(train_name, test_name):
    if train_name == test_name:
        return f"Skipping {train_name} == {test_name}"
    if "Lorenz96" in train_name or "Lorenz96" in test_name:
        return f"Skipping {train_name}, {test_name} because Lorenz96 is excluded"
    eq = getattr(dysts.flows, train_name)()
    if len(eq.ic) > 3:
        return f"Skipping {train_name}, {test_name} because train system dimension > 3"
    return None


def build_tokenizer(vocab_size):
    return ChronosTokenizer(vocab_size, -3, 3)


def make_pair_datasets(train_name, test_name, args):
    tokenizer = build_tokenizer(args.vocab_size)

    fpaths = check_for_existing_traj_files(train_name, min_len=args.n_train)
    if len(fpaths) > 0:
        traj_train = np.load(fpaths[0], allow_pickle=True)
        print(f"Found existing trajectory file for {train_name}: {fpaths[0]}", flush=True)
    else:
        eq_train = getattr(dysts.flows, train_name)()
        traj_train = eq_train.make_trajectory(
            args.n_train,
            standardize=True,
            resample=True,
            pts_per_period=args.pts_per_period,
        )
        if traj_train is None:
            raise RuntimeError(f"Failed to generate training trajectory for {train_name}")
    
    eq_train_id = getattr(dysts.flows, train_name)()
    eq_train_id.ic = np.asarray(eq_train_id.ic) + 1.1
    traj_test_id = eq_train_id.make_trajectory(
        200 + args.n_test,
        standardize=True,
        resample=True,
        pts_per_period=args.pts_per_period,
    )
    if traj_test_id is None:
        raise RuntimeError(f"Failed to generate ID test trajectory for {train_name}")

    fpaths = check_for_existing_traj_files(test_name)
    if len(fpaths) > 0:
        traj_test_ood = np.load(fpaths[0], allow_pickle=True)
        print(f"Found existing trajectory file for {test_name}: {fpaths[0]}", flush=True)
    else:
        eq_test_ood = getattr(dysts.flows, test_name)()
        traj_test_ood = eq_test_ood.make_trajectory(
            args.n_train,
            standardize=True,
            resample=True,
            pts_per_period=args.pts_per_period,
        )
        if traj_test_ood is None:
            raise RuntimeError(f"Failed to generate OOD test trajectory for {test_name}")
    
    x_train = traj_train[:, 0]
    tok_train, _ = tokenizer.encode_series(x_train, 100, 10)
    tok_train = tok_train[:-1]
    
    x_test_id = traj_test_id[:, 0]
    tok_test_id, _ = tokenizer.encode_series(x_test_id, 100, 10)
    tok_test_id = tok_test_id[:-1]
    
    x_test_ood = traj_test_ood[:, 0]
    tok_test_ood, _ = tokenizer.encode_series(x_test_ood, 100, 10)
    tok_test_ood = tok_test_ood[:-1]

    return {
        "traj_train": traj_train,
        "traj_test_id": traj_test_id,
        "traj_test_ood": traj_test_ood,
        "tok_train": tok_train,
        "tok_test_id": tok_test_id,
        "tok_test_ood": tok_test_ood,
    }


def get_train_sizes(args, max_train_size):
    min_train_size = args.context_length + 2
    if max_train_size < min_train_size:
        return np.asarray([], dtype=int)

    if args.train_sizes is not None:
        train_sizes = np.asarray(args.train_sizes, dtype=int)
    else:
        train_sizes = np.geomspace(
            min_train_size,
            max_train_size,
            num=args.num_train_sizes,
        )
        train_sizes = np.rint(train_sizes).astype(int)

    train_sizes = np.unique(np.clip(train_sizes, min_train_size, max_train_size))
    if len(train_sizes) == 0 or train_sizes[-1] != max_train_size:
        train_sizes = np.unique(np.append(train_sizes, max_train_size))
    return train_sizes


def save_pair_metadata(pair_dir, train_name, test_name, args, train_sizes):
    metadata = {
        "train_name": train_name,
        "test_name": test_name,
        "vocab_size": args.vocab_size,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "d_k": args.d_k,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "cadence": args.cadence,
        "horizon": args.horizon,
        "pts_per_period": args.pts_per_period,
        "seed": args.seed,
        "torch_seed": args.torch_seed,
        "context_length": int(args.context_length),
        "train_sizes": [int(x) for x in train_sizes],
    }
    with open(pair_dir / "run_config.json", "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)


def save_pair_trajectories(pair_dir, train_name, test_name, datasets):
    datasets["traj_train"].dump(pair_dir / f"traj_train_{train_name}.pkl")
    datasets["traj_test_id"].dump(pair_dir / f"traj_test_id_{train_name}.pkl")
    datasets["traj_test_ood"].dump(pair_dir / f"traj_test_ood_{test_name}.pkl")


def make_unique_pair_dir(base_path, train_name, test_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pair_dir = base_path / f"{timestamp}_{train_name}_{test_name}"
    suffix = 1
    while pair_dir.exists():
        pair_dir = base_path / f"{timestamp}_{train_name}_{test_name}_{suffix:02d}"
        suffix += 1
    pair_dir.mkdir(parents=True, exist_ok=False)
    return pair_dir


def forecast_metrics(model, tok_test_id, tok_test_ood, context_length, horizon):
    if min(len(tok_test_id), len(tok_test_ood)) < context_length + horizon:
        return None

    ctx = torch.tensor(
        tok_test_id[-(context_length + horizon) : -horizon], dtype=torch.long
    )
    out = generate_autoregressive(
        model, ctx, max_new_tokens=horizon, greedy=True
    ).squeeze()

    ctx_ood = torch.tensor(
        tok_test_ood[-(context_length + horizon) : -horizon], dtype=torch.long
    )
    out_ood = generate_autoregressive(
        model, ctx_ood, max_new_tokens=horizon, greedy=True
    ).squeeze()

    preds_id = out.tolist()[context_length : context_length + horizon]
    preds_ood = out_ood.tolist()[context_length : context_length + horizon]
    true_id = tok_test_id[-horizon:]
    true_ood = tok_test_ood[-horizon:]

    return {
        "in_mse": mse_loss(true_id, preds_id),
        "out_mse": mse_loss(true_ood, preds_ood),
        "pred_id": preds_id,
        "pred_ood": preds_ood,
        "true_id": true_id.tolist(),
        "true_ood": true_ood.tolist(),
    }


def train_data_size(train_dir, train_size, datasets, args):
    context_length = args.context_length
    final_ckpt = train_dir / f"tiny_lm_context{context_length}_ntrain{train_size}.pt"
    losses_path = train_dir / f"losses_context{context_length}_ntrain{train_size}.npz"
    metrics_path = train_dir / f"forecast_metrics_context{context_length}_ntrain{train_size}.json"
    min_token_count = min(
        len(datasets["tok_train"]),
        len(datasets["tok_test_id"]),
        len(datasets["tok_test_ood"]),
    )

    if final_ckpt.exists() and not args.overwrite:
        print(f"Skipping n_train {train_size}; found {final_ckpt}", flush=True)
        return
    if context_length >= min_token_count:
        print(
            f"Skipping n_train {train_size}; shortest token stream has length {min_token_count}",
            flush=True,
        )
        return
    if train_size > len(datasets["tok_train"]):
        print(
            f"Skipping n_train {train_size}; training token stream has length {len(datasets['tok_train'])}",
            flush=True,
        )
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training with n_train {train_size} at context length {context_length}", flush=True)

    torch.manual_seed(args.torch_seed)
    tokens_train = torch.tensor(datasets["tok_train"][:train_size], dtype=torch.long)
    tokens_val = torch.tensor(datasets["tok_test_id"], dtype=torch.long)
    tokens_test_out = torch.tensor(datasets["tok_test_ood"], dtype=torch.long)

    model, losses, val_losses, val_losses_ood = train_next_token(
        tokens_train,
        tokens_val[: args.n_test],
        tokens_test_out[: args.n_test],
        vocab_size=1 + args.vocab_size,
        block_size=context_length,
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        d_model=args.d_model,
        d_k=args.d_k,
        weight_decay=args.weight_decay,
        save_path=str(train_dir) + "/",
        cadence=args.cadence,
    )

    np.array([losses, val_losses, val_losses_ood]).T.dump(losses_path)
    save_checkpoint(final_ckpt, model)

    metrics = forecast_metrics(
        model,
        datasets["tok_test_id"],
        datasets["tok_test_ood"],
        context_length,
        args.horizon,
    )
    if metrics is not None:
        with open(metrics_path, "w", encoding="ascii") as f:
            json.dump(metrics, f, indent=2)


def train_pair(train_name, test_name, args):
    pair_dir = make_unique_pair_dir(args.base_path, train_name, test_name)

    datasets = make_pair_datasets(train_name, test_name, args)
    train_sizes = get_train_sizes(args, len(datasets["tok_train"]))
    save_pair_metadata(pair_dir, train_name, test_name, args, train_sizes)
    save_pair_trajectories(pair_dir, train_name, test_name, datasets)

    for train_size in train_sizes:
        train_dir = pair_dir / f"ntrain{int(train_size)}"
        train_data_size(train_dir, int(train_size), datasets, args)


def main():
    args = parse_args()
    args.base_path.mkdir(parents=True, exist_ok=True)

    for train_name, test_name in sample_system_pairs(args.num_pairs, args.seed):
        skip_reason = should_skip_pair(train_name, test_name)
        if skip_reason is not None:
            print(skip_reason, flush=True)
            continue

        print(
            f"Running pretraining-data sweep for {train_name} -> {test_name} "
            f"at context length {args.context_length}",
            flush=True,
        )
        try:
            train_pair(train_name, test_name, args)
        except Exception as exc:
            print(
                f"Failed {train_name} -> {test_name}: {type(exc).__name__}: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
