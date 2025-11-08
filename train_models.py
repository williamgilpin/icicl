import numpy as np

VOCAB_SIZE = 100
N_TRAIN = 2*40_000
N_TEST = 1000
CONTEXT_LENGTH = 32*4
EPOCHS = 60_000

from models import ChronosTokenizer
tokenizer = ChronosTokenizer(VOCAB_SIZE, -3, 3)

# VOCAB_SIZE = 100
# N_TRAIN = 5_000
# N_TEST = 1000
# CONTEXT_LENGTH = 32*4
# EPOCHS = 1_000

from dysts.systems import get_attractor_list
import dysts.flows
import datetime

import torch
from models import TinyCausalLM, train_next_token
from models import save_checkpoint

all_attractors = get_attractor_list(sys_class="continuous_no_delay")

## Only train on models not seen before
import glob
all_model_paths = glob.glob("./private_data/tiny_lm_*.pt")
all_model_paths = [item for item in all_model_paths if len(item.split("_")) > 5]
seen_models = [item.split("_")[5] for item in all_model_paths]
all_attractors = [item for item in all_attractors if item not in seen_models]

np.random.seed(49)
num_systems = len(all_attractors)
training_systems = np.random.choice(all_attractors, size=num_systems, replace=False)
test_systems = np.random.choice(all_attractors, size=num_systems, replace=False)
print(training_systems)
print(test_systems)

for train_name, test_name in zip(training_systems, test_systems):
    if train_name == test_name:
        print(f"Skipping {train_name} == {test_name}", flush=True)
        continue
    if "Lorenz96" in train_name or "Lorenz96" in test_name:
        print(f"Skipping {train_name}, {test_name}", flush=True)
        continue
    print(train_name, test_name, flush=True)

    ## Make train tokens
    eq = getattr(dysts.flows, train_name)()
    traj_train = eq.make_trajectory(N_TRAIN, standardize=True, resample=True, pts_per_period=30)
    if traj_train is None:
        print(f"Skipping {train_name}, {test_name}", flush=True)
        continue
    traj_train.dump(f"private_cache/traj_{train_name}.pkl")
    x_train = traj_train[:, 0]

    ## Make in-distribution test set
    tok_train, aux = tokenizer.encode_series(x_train, 100, 10)
    ## Make In-distribution test set
    eq.ic += 1.1
    traj_test = eq.make_trajectory(200 + N_TEST, standardize=True, resample=True, pts_per_period=30)
    if traj_test is None:
        print(f"Skipping {train_name}, {test_name}", flush=True)
        continue
    traj_test.dump(f"private_cache/traj_{test_name}.pkl")
    x_test = traj_test[:, 0]
    tok_test, aux = tokenizer.encode_series(x_test, 100, 10)

    eq2 = getattr(dysts.flows, test_name)()
    traj_test_out = eq2.make_trajectory(N_TEST, standardize=True, resample=True, pts_per_period=30)
    if traj_test_out is None:
        print(f"Skipping {train_name}, {test_name}", flush=True)
        continue
    traj_test_out.dump(f"private_cache/traj_{test_name}.pkl")
    x_test_out = traj_test_out[:, 0]
    tok_test_out, aux = tokenizer.encode_series(x_test_out, 100, 10)


    torch.manual_seed(0)
    tokens_train = torch.tensor(tok_train, dtype=torch.long)
    tokens_val = torch.tensor(tok_test, dtype=torch.long)
    tokens_test_out = torch.tensor(tok_test_out, dtype=torch.long)
    model, losses, val_losses, val_losses_ood = train_next_token(tokens_train, tokens_val[:N_TEST], tokens_test_out[:N_TEST], vocab_size=(1 + VOCAB_SIZE), 
                                                                 block_size=CONTEXT_LENGTH, lr=1e-4, batch_size=64*2, steps=EPOCHS, d_model=128 * 2, 
                                                                 d_k=64, weight_decay=1e0)
    
    dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dt_str += f"_{train_name}_{test_name}"
    losses = np.array([losses, val_losses, val_losses_ood]).T
    losses.dump(f"./private_data/losses_{dt_str}.npz")
    ckpt_name = f"./private_data/tiny_lm_{dt_str}.pt"
    save_checkpoint(ckpt_name, model)