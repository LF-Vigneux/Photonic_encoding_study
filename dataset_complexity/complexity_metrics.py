import numpy as np
import merlin as ml
import torch.nn as nn
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_complexity.utils import (
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
    average_bipartite_entanglement_entropy,
    multipartite_total_correlation,
    effective_kernel_rank,
    nonclassicality,
    quantum_fisher_information,
    topological_quantum_complexity,
)


def classical_complexity(
    X: torch.Tensor,
    hyper_parameters: list[float] = [1, 1, 1, 1],
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
    return {
        "distributional_entropy": de,
        "correlation_order": co,
        "kolmogorov_complexity": kc,
        "topological_complexity": tc,
        "total": de + co + kc + tc,
    }


def induced_quantum_complexity(
    X: torch.Tensor,
    encoding: nn.Module,
    hyper_parameters: list[float] = [1, 1, 1, 1, 1, 1],
    epsilon_hilbert_support_dim: float = 1e-8,
    n_samples_loc_vs_express: int = 1000,
    n_bins_loc_vs_express: int = 50,
    max_dim_topology: int = 2,
    weights_topology: list[float] | None = None,
    max_samples_topology: int | None = 1000,
    max_samples: int | None = 5000,
) -> float:
    if max_samples is not None and X.size(0) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.size(0), size=max_samples, replace=False)
        X = X[idx]

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
    print("[induced_quantum_complexity] All metrics computed.")
    return {
        "hilbert_space_support_dim": hsd,
        "quantum_fisher_information_spread": qfi,
        "entanglement_entropy": ee,
        "kernel_spectrum_flatness": ksf,
        "locality_vs_expressibility": lve,
        "topological_invariants_of_embedding": tie,
        "total": hsd + qfi + ee + ksf + lve + tie,
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
