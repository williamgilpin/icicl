import glob
import os
import numpy as np
import torch
import torch.nn.functional as F

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, entropy
from transitions import transition_probs_mc, transition_probs2, transition_probs_mc_greedy_one_step
from operators import SymbolicMarkovChain, invariant_distribution, reduce_markov_chain
from models import load_model

order = 8

device = "mps"

VOCAB_SIZE = 100
# N_TRAIN = 2*40_000
# N_TEST = 1000
CONTEXT_LENGTH = 32*4

# all_system_folders = glob.glob("./private_data/training_run/*/")
all_model_paths = glob.glob("./private_data/training_run/*/tiny_lm_*.pt")
all_data_paths = glob.glob("./private_data/training_run/*/*.pkl")
all_data_paths_large = []
for data_path in all_data_paths:
    traj_test_out = np.load(data_path, allow_pickle=True)
    if traj_test_out.shape[0] > 1001:
        all_data_paths_large.append(data_path)
all_data_paths_large = np.array(all_data_paths_large)
all_model_paths = np.array(all_model_paths)

# np.random.seed(1)
np.random.seed(0)
for _ in range(50):

    model_path = np.random.choice(all_model_paths, size=1, replace=False)[0]
    data_path = np.random.choice(all_data_paths_large, size=1, replace=False)[0]
    loss_path = glob.glob(os.path.dirname(data_path) + "/losses_*.npz")[0]
    eq_name = model_path.split("/")[-1].split("_")[4]
    eq_name_downstream = model_path.split("/")[-1].split("_")[5].split(".")[0]
    eq_name_target = data_path.split("/")[-1].split(".")[0].split("_")[1]
    ## Skip the same train/test pair
    if eq_name_target == eq_name:
        continue
    overall_name = eq_name + "_" + eq_name_target
    print(overall_name, flush=True)

    model = load_model(model_path, device=device)

    ## Load the losses
    losses = np.load(loss_path, allow_pickle=True)
    losses, val_losses, val_losses_ood = losses.T 

    ## Make the test data and tokenize it
    traj_test_out = np.load(data_path, allow_pickle=True)
    x_test_out = traj_test_out[:, 0]
    from models import ChronosTokenizer
    tokenizer = ChronosTokenizer(VOCAB_SIZE, -3, 3)
    tok_test_out, aux = tokenizer.encode_series(x_test_out, 100, 10)
    ## Compute fully-observed transition matrix
    mk = SymbolicMarkovChain()
    label_order = mk.fit_predict(traj_test_out, 25, 1)
    P_symbolic = mk.P_.copy()
    centroids = mk.clusterer.cluster_centers_
    centroids_x = centroids[:, 0]
    centroid_closest_inds = order + np.argmin(cdist(centroids_x[:, None], x_test_out[order:][:, None]), axis=1)
    offsets = np.arange(order - 1, -1, -1)   # shape (order,)
    centroid_kgrams = tok_test_out[centroid_closest_inds[:, None] - offsets]


    test_tensor = torch.tensor(
            np.lib.stride_tricks.sliding_window_view(tok_test_out[:-1], CONTEXT_LENGTH),
            # np.lib.stride_tricks.sliding_window_view(tok_train[:-1], CONTEXT_LENGTH),
            dtype=torch.long,
            #  device="cpu"
    )

    device = test_tensor.device
    kmers = test_tensor[:, -order:]
    kmers_unique, kmers_indices, kmers_counts = torch.unique(
        kmers, dim=0, return_inverse=True, return_counts=True
    )

    pi_dist_symbolic = invariant_distribution(P_symbolic)

    ## Added
    eigvals_symbolic, eigvecs_symbolic = np.linalg.eig(P_symbolic.T)
    sort_order = np.argsort(np.abs(eigvals_symbolic))[::-1]
    eigvals_symbolic = eigvals_symbolic[sort_order[:10]]
    eigvecs_symbolic = eigvecs_symbolic[:, sort_order[:10].copy()]

    ## Map the kmers to the centroids of the full-state Markov chain
    kgram_map = np.argmin(cdist(kmers_unique, centroid_kgrams), axis=1)


    all_spearman, all_kldiv = [], []
    all_all_eig_spearman = []
    all_entropy_true_invariant, all_entropy_model_invariant, all_entropy_model_empirical = [], [], []
    model_paths = sorted(glob.glob("./private_data/training_run/ckpt_*.pt"))
    for model_path in model_paths:
        print(f"Evaluating {model_path}", flush=True)
        model = load_model(model_path, device="cpu")

        # set eval mode
        model.eval()
        torch.set_grad_enabled(False)


        ## Stochastic decoding
        K = len(centroid_kgrams)
        P_reduced_sum = torch.zeros((K, K))
        for ind in range(400 * 4):
            if ind % 100 == 0:
                print(f"Running {ind}", flush=True)
            P_reduced = transition_probs_mc(
                test_tensor, 
                torch.tensor(centroid_kgrams), 
                model,
                # ground_truth=True,
                shift=1,
                n_input_samples=2000 // 4, 
                n_samples_per_input=1,
                temperature=0.00001,
                use_compile=True,
                normalize=False
            )
            P_reduced_sum += P_reduced
        P_reduced = P_reduced_sum.clone()
        row_sum = P_reduced.sum(dim=-1, keepdim=True)
        row_sum  = row_sum.clamp_min(1e-30)
        P_reduced = P_reduced / row_sum


        ## Greedy one-step decoding
        # P_reduced = transition_probs_mc_greedy_one_step(
        #     test_tensor, torch.tensor(centroid_kgrams), model,
        #     use_compile=True,
        # )
        
        ## Exact decoding
        # P = transition_probs2(test_tensor, kmers_unique, model, use_compile=True)
        # P_reduced = reduce_markov_chain(P, kgram_map, weights=invariant_distribution(P))

        pi_dist_model = invariant_distribution(P_reduced)

        ## Additional eigvecs
        eigvals, eigvecs = np.linalg.eig(P_reduced.T)
        sort_order = np.argsort(np.abs(eigvals))[::-1]
        eigvals_model = eigvals[sort_order[:10]]
        eigvecs_model = eigvecs[:, sort_order[:10].copy()]
        all_eig_spearman = []
        for i in range(len(eigvals_model)):
            all_eig_spearman.append(spearmanr(eigvecs_model[:, i], eigvecs_symbolic[:, i])[0])
        all_all_eig_spearman.append(np.array(all_eig_spearman))

        all_spearman.append(spearmanr(pi_dist_symbolic, pi_dist_model)[0])
        all_kldiv.append(entropy(pi_dist_symbolic, pi_dist_model))

        ## compute entropy of the model distribution
        logits = model(test_tensor[:5000])[..., -1]
        probs = F.softmax(logits, dim=-1)
        entropy_model = entropy(probs, axis=-1)
        average_entropy_model_empirical = entropy_model.mean()
        average_entropy_model_dist = entropy(pi_dist_model)
        average_entropy_true_dist = entropy(pi_dist_symbolic)
        all_entropy_true_invariant.append(average_entropy_true_dist)
        all_entropy_model_invariant.append(average_entropy_model_dist)
        all_entropy_model_empirical.append(average_entropy_model_empirical)


        print(f"Spearman r={all_spearman[-1]:.2f}, KLD={all_kldiv[-1]:.2f}")


    all_all_eig_spearman = np.array(all_all_eig_spearman)
    np.array(all_all_eig_spearman).dump(f"private_data/eig_spearman_{overall_name}2.pkl")
    np.array(all_spearman).dump(f"private_data/invariant_spearman_{overall_name}2.pkl")
    np.array(all_kldiv).dump(f"private_data/invariant_kldiv_{overall_name}2.pkl")
    np.array(val_losses_ood).dump(f"private_data/val_losses_ood_{overall_name}2.pkl")
    np.array(all_entropy_true_invariant).dump(f"private_data/entropy_true_invariant_{overall_name}2.pkl")
    np.array(all_entropy_model_invariant).dump(f"private_data/entropy_model_invariant_{overall_name}2.pkl")
    np.array(all_entropy_model_empirical).dump(f"private_data/entropy_model_empirical_{overall_name}2.pkl")

