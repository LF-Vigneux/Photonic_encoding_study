import numpy as np
import merlin as ml
from math import comb
import torch.nn as nn
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_complexity.utils import (
    dataset_wasserstein,
    distributional_entropy,
    correlation_order,
    kolmogorov_complexity,
    topological_complexity,
    hilbert_space_support_dim,
    quantum_fisher_information_spread,
    entanglement_entropy,
    kernel_spectrum_flatness,
    locality_vs_expressibility,
    topological_invariants_of_embedding,
    encoded_states_classes_overlap,
    average_bipartite_entanglement_entropy,
    multipartite_total_correlation,
    effective_kernel_rank,
    nonclassicality,
    quantum_fisher_information,
    topological_quantum_complexity,
)
from nn_embedding.lib.merlin_based_model import (
    NeuralEmbeddingMerLinKernel,
)
from nn_embedding.utils.merlin_model_utils import assign_params  # noqa: E402


def classical_complexity(
    X: torch.Tensor,
    Y: torch.Tensor,
    hyper_parameters: list[float] = [1, 1, 1, 1, 1],
    max_order_correlation: int | None = None,
    max_dim_topology: int = 2,
    weights_topology: list[float] | None = None,
    max_samples_topology: int | None = 1000,
) -> dict:
    de = hyper_parameters[0] * distributional_entropy(X)
    co = hyper_parameters[1] * correlation_order(X, max_order=max_order_correlation)
    kc = hyper_parameters[2] * kolmogorov_complexity(X)
    tc = hyper_parameters[3] * topological_complexity(
        X,
        max_dim=max_dim_topology,
        weights=weights_topology,
        max_samples=max_samples_topology,
    )
    ws = hyper_parameters[4] * dataset_wasserstein(X, Y)
    N = int(X.size(0))
    max_order = 4 if max_order_correlation is None else max_order_correlation
    min_max = [
        [0, np.log2(N)],
        [
            -((2**max_order) - 2) * np.log(max_order),
            ((2**max_order) - 2) * np.log(max_order),
        ],
        [0, 1],
        [
            0,
            (N - 1) + np.sum([comb(N, k) for k in range(2, max_dim_topology + 2)]),
        ],
        [0, np.prod(X.shape[1:])],
    ]
    return {
        "distributional_entropy": de,
        "correlation_order": co,
        "kolmogorov_complexity": kc,
        "topological_complexity": tc,
        "wasserstein distance": ws,
        "total": de + co + kc + tc + ws,
        "min_max": min_max,
    }


def induced_quantum_complexity(
    X: torch.Tensor,
    Y: torch.Tensor,
    encoding: nn.Module,
    hyper_parameters: list[float] = [1, 1, 1, 1, 1, 1, 1],
    epsilon_hilbert_support_dim: float = 1e-8,
    n_samples_loc_vs_express: int = 1000,
    n_bins_loc_vs_express: int = 50,
    max_dim_topology: int = 2,
    weights_topology: list[float] | None = None,
    max_samples_topology: int | None = 1000,
    max_samples: int | None = 5000,
    distance: str = "Trace",
) -> dict:
    """
    Compute induced quantum complexity metrics for a dataset with quantum encoding.

    Computes 7 metrics characterizing the quantum properties of the encoded dataset:
    1. hilbert_space_support_dim: Dimension of Hilbert space occupied by encoding
    2. quantum_fisher_information_spread: Quantum Fisher information metric spread
    3. entanglement_entropy: Von Neumann entropy of entanglement
    4. kernel_spectrum_flatness: Flatness of kernel eigenvalue spectrum
    5. locality_vs_expressibility: Trade-off between locality and expressibility
    6. topological_invariants_of_embedding: Topological properties of embedded data
    7. encoded_states_classes_overlap: Overlap of encoded class states in Hilbert space

    Parameters
    ----------
    X : torch.Tensor
        Input feature data, shape (n_samples, *feature_shape)
    Y : torch.Tensor
        Class labels, shape (n_samples,)
    encoding : nn.Module
        Quantum encoding layer
    hyper_parameters : list[float], optional
        Weights for each of the 7 metrics, by default [1, 1, 1, 1, 1, 1, 1]
    epsilon_hilbert_support_dim : float, optional
        Threshold for Hilbert space dimension computation, by default 1e-8
    n_samples_loc_vs_express : int, optional
        Samples for locality vs expressibility computation, by default 1000
    n_bins_loc_vs_express : int, optional
        Bins for locality vs expressibility histogram, by default 50
    max_dim_topology : int, optional
        Maximum simplicial dimension for topology computation, by default 2
    weights_topology : list[float] | None, optional
        Weights for topological dimensions, by default None
    max_samples_topology : int | None, optional
        Max samples for topology computation, by default 1000
    max_samples : int | None, optional
        Max samples for induced complexity computation, by default 5000
    distance : str, optional
        Distance metric for state overlap, by default "Trace"

    Returns
    -------
    dict
        Dictionary with keys:
        - Individual metric names (7 total)
        - "total": Sum of weighted metrics
        - "min_max": List of [min, max] bounds for each metric
    """
    if max_samples is not None and X.size(0) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.size(0), size=max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]

    print("[induced_quantum_complexity] Computing hilbert_space_support_dim...")
    hsd = hyper_parameters[0] * hilbert_space_support_dim(
        X, encoding, eps=epsilon_hilbert_support_dim
    )
    print("[induced_quantum_complexity] Computing quantum_fisher_information_spread...")
    qfi = hyper_parameters[1] * quantum_fisher_information_spread(X, encoding)
    print("[induced_quantum_complexity] Computing entanglement_entropy...")
    ee = hyper_parameters[2] * entanglement_entropy(X, encoding)
    print("[induced_quantum_complexity] Computing kernel_spectrum_flatness...")
    ksf = hyper_parameters[3] * kernel_spectrum_flatness(X, encoding)
    print("[induced_quantum_complexity] Computing locality_vs_expressibility...")
    lve = hyper_parameters[4] * locality_vs_expressibility(
        X,
        encoding,
        n_samples=n_samples_loc_vs_express,
        n_bins=n_bins_loc_vs_express,
        ee=ee,
    )
    print(
        "[induced_quantum_complexity] Computing topological_invariants_of_embedding..."
    )
    tie = hyper_parameters[5] * topological_invariants_of_embedding(
        X,
        encoding,
        max_dim=max_dim_topology,
        weights=weights_topology,
        max_samples=max_samples_topology,
    )
    ove = hyper_parameters[6] * encoded_states_classes_overlap(
        X,
        Y,
        encoding,
        distance=distance,
    )
    print("[induced_quantum_complexity] All metrics computed.")
    if isinstance(encoding, NeuralEmbeddingMerLinKernel):

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical_model = encoding.classical_encoder

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    output = assign_params(encoding.quantum_embedding_layer, params)
                return output

        encoder = Encoder()
    else:
        encoder = encoding
    hilbert_dim = encoder(X[0]).squeeze().size(0)
    d = X[0].numel()
    eps = epsilon_hilbert_support_dim
    N = int(X.size(0))
    min_max = [
        [1, hilbert_dim],
        [
            0,
            (4 * d) / (eps**2),
        ],
        [0, 0.5 * np.log2(hilbert_dim)],
        [1, N],
        [0, 1],
        [
            0,
            (N - 1) + np.sum([comb(N, k) for k in range(2, max_dim_topology + 2)]),
        ],
        [0, 1],
    ]
    return {
        "hilbert_space_support_dim": hsd,
        "quantum_fisher_information_spread": qfi,
        "entanglement_entropy": ee,
        "kernel_spectrum_flatness": ksf,
        "locality_vs_expressibility": lve,
        "topological_invariants_of_embedding": tie,
        "encoded_states_classes_overlap": ove,
        "total": hsd + qfi + ee + ksf + lve + tie + ove,
        "min_max": min_max,
    }


def quantum_complexity(
    X: torch.Tensor,
    computation_space: ml.ComputationSpace,
    n_modes: int,
    state_keys: list[tuple[int]],
    n_photons: int | None = None,
    hyper_parameters: list[float] = [1, 1, 1, 1, 1, 1],
) -> float:
    print("[quantum_complexity] Computing average_bipartite_entanglement_entropy...")
    abee = hyper_parameters[0] * average_bipartite_entanglement_entropy(
        X,
        computation_space=computation_space,
        state_keys=state_keys,
        n_modes=n_modes,
        n_photons=n_photons,
    )
    print("[quantum_complexity] Computing multipartite_total_correlation...")
    mtc = hyper_parameters[1] * multipartite_total_correlation(
        X,
        num_subsystem=(
            n_modes // 2
            if computation_space == ml.ComputationSpace.DUAL_RAIL
            else n_modes
        ),
        dim_per_state=(
            2 if computation_space == ml.ComputationSpace.DUAL_RAIL else n_photons + 1
        ),
        fock_space=(
            False if computation_space == ml.ComputationSpace.DUAL_RAIL else True
        ),
        num_photons=n_photons,
        state_keys=state_keys,
    )
    print("[quantum_complexity] Computing effective_kernel_rank...")
    ekr = hyper_parameters[2] * effective_kernel_rank(X)
    print("[quantum_complexity] Computing nonclassicality...")
    nc = hyper_parameters[3] * nonclassicality(X)
    print("[quantum_complexity] Computing quantum_fisher_information...")
    qfi = hyper_parameters[4] * quantum_fisher_information(X)
    print("[quantum_complexity] Computing topological_quantum_complexity...")
    tqc = hyper_parameters[5] * topological_quantum_complexity(X)
    print("[quantum_complexity] All metrics computed.")
    return abee + mtc + ekr + nc + qfi + tqc
