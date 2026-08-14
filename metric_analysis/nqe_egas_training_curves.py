"""
Train NQE and EGAS on a chosen dataset (same procedure as run_dataset_complexity.py),
plot their training curves, and report train/test/total accuracy and loss for both.

python3 nqe_egas_training_curves.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import merlin as ml
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import data_load_and_process  # noqa: E402
from encodings_merlin.utils import find_mode_photon_config  # noqa: E402
from egas.photonic_circuits import build_token_pool  # noqa: E402
from egas.photonic_egas import (  # noqa: E402
    run_egas as run_photonic_egas,
    unique_sorted_candidates,
    refine_candidates,
    pairwise_energy,
)
from egas.photonic_kernel_svm import C_SVM  # noqa: E402
from egas.statevec import fidelity_matrix  # noqa: E402
from encodings_merlin.encoding_layers import EGASEncoder  # noqa: E402
from nn_embedding.lib.merlin_based_model import NeuralEmbeddingMerLinKernel  # noqa: E402
from nn_embedding.utils.merlin_model_utils import assign_params  # noqa: E402
from nn_embedding.utils.utils import create_balanced_pairs  # noqa: E402

# ── Dataset hyperparameters ─────────────────────────────────────────────────
DATASET_NAME = "wine"
FEATURE_REDUCTION = 8
CLASSES = (0, 1, 2)
NOISE_GENERATED = 0.0

# ── Training hyperparameters (independent per method) ───────────────────────
NQE_NUM_EPOCHS = 1000
EGAS_NUM_EPOCHS = 4000  # GPT search iterations, mirrors n_iters in run_dataset_complexity
EGAS_REFINE_EPOCHS = 400  # bias refinement epochs per candidate, mirrors run_dataset_complexity

COMPUTATION_SPACE = ml.ComputationSpace.UNBUNCHED
SEED = 0

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_PDF = OUTPUT_DIR / "nqe_egas_training_curves.pdf"
OUTPUT_SUMMARY_JSON = OUTPUT_DIR / "nqe_egas_training_summary.json"

# Font size hyperparameters
TITLE_FONTSIZE = 22
SUBPLOT_TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 11


def _peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB.
    return ru_maxrss / (1024**2) if sys.platform == "darwin" else ru_maxrss / 1024


class _StageTimer:
    """Measures wall-clock time and process peak RSS growth for a training stage."""

    def __init__(self, num_epochs: int | None = None):
        self.num_epochs = num_epochs

    def __enter__(self):
        self._start = time.perf_counter()
        self._rss_before_mb = _peak_rss_mb()
        return self

    def __exit__(self, *exc_info):
        self.time_s = time.perf_counter() - self._start
        self.peak_rss_mb = _peak_rss_mb()
        self.rss_growth_mb = self.peak_rss_mb - self._rss_before_mb
        self.time_per_epoch_s = (
            self.time_s / self.num_epochs if self.num_epochs else None
        )

    def as_dict(self, prefix: str) -> dict:
        return {
            f"{prefix}_time_s": self.time_s,
            f"{prefix}_time_per_epoch_s": self.time_per_epoch_s,
            f"{prefix}_peak_rss_mb": self.peak_rss_mb,
            f"{prefix}_rss_growth_mb": self.rss_growth_mb,
        }


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.size(0), -1) if tensor.ndim > 2 else tensor


def _states_from_embedder(x: torch.Tensor, embedder) -> torch.Tensor:
    """Return the quantum output states for either NQE or EGAS style embedders."""
    if isinstance(embedder, NeuralEmbeddingMerLinKernel):
        params = embedder.classical_encoder(x)
        params = params.reshape(params.size(0), -1)
        with torch.no_grad():
            return assign_params(embedder.quantum_embedding_layer, params)
    with torch.no_grad():
        return embedder(x)


def _qksvm_accuracies(embedder, x_train, y_train, x_test, y_test) -> dict:
    """Fit a QKSVM on the train fidelity kernel; report train/test/total accuracy."""
    y_train_np = y_train.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()
    y_total_np = np.concatenate([y_train_np, y_test_np])

    states_train = _states_from_embedder(x_train, embedder)
    states_test = _states_from_embedder(x_test, embedder)
    states_total = torch.cat([states_train, states_test], dim=0)

    K_train = fidelity_matrix(states_train, states_train).cpu().numpy()
    K_test = fidelity_matrix(states_test, states_train).cpu().numpy()
    K_total = fidelity_matrix(states_total, states_train).cpu().numpy()

    svc = SVC(kernel="precomputed", C=C_SVM)
    svc.fit(K_train, y_train_np)

    return {
        "train_accuracy": float((svc.predict(K_train) == y_train_np).mean()),
        "test_accuracy": float((svc.predict(K_test) == y_test_np).mean()),
        "total_accuracy": float((svc.predict(K_total) == y_total_np).mean()),
    }


def _train_nqe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    X: torch.Tensor,
    n_features: int,
    num_modes_encoder: int,
    num_photons_encoder: int,
    num_epochs: int,
) -> tuple[NeuralEmbeddingMerLinKernel, dict, dict]:
    """Build and train the NQE model exactly as in run_dataset_complexity.py."""
    batch_size = (
        len(CLASSES) * 100 if CLASSES is not None else len(torch.unique(y_train)) * 100
    )

    general_unitary = ml.CircuitBuilder(n_modes=num_modes_encoder)
    general_unitary.add_entangling_layer()
    general_unitary.add_entangling_layer()

    encoder = ml.QuantumLayer(
        input_size=0,
        builder=deepcopy(general_unitary),
        n_photons=num_photons_encoder,
        measurement_strategy=ml.MeasurementStrategy.amplitudes(
            computation_space=COMPUTATION_SPACE
        ),
    )

    if len(X.shape) == 2:
        classical_model = nn.Sequential(
            nn.Linear(n_features, n_features // 2 + 10),
            nn.ReLU(),
            nn.Linear(n_features // 2 + 10, n_features // 2 + 10),
            nn.ReLU(),
            nn.Linear(
                n_features // 2 + 10,
                sum(i.numel() for i in encoder.parameters()),
            ),
        )
    else:
        in_channels = X.shape[1] if X.ndim == 4 else 1
        classical_model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LazyLinear(sum(i.numel() for i in encoder.parameters())),
        )

    model = NeuralEmbeddingMerLinKernel(
        classical_model=classical_model,
        quantum_embedding_layer=encoder,
    )

    # Mirrors NeuralEmbeddingMerLinKernel.train_embedding's inner loop, but keeps
    # per-epoch train/test loss so a training curve can be plotted.
    optimizer = torch.optim.Adam(model.classical_encoder.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    train_loss_history = []
    test_loss_history = []

    with _StageTimer(num_epochs=num_epochs) as train_timer:
        for epoch in range(num_epochs):
            model.kernel_function.train()
            X1_batch, X2_batch, Y_batch = create_balanced_pairs(
                batch_size, x_train, y_train
            )
            pairs = torch.cat([X1_batch, X2_batch], dim=1)

            optimizer.zero_grad()
            outputs = model.kernel_function(pairs)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            model.kernel_function.eval()
            train_loss_history.append(float(loss.detach().cpu()))

            with torch.no_grad():
                X1_test, X2_test, Y_test_batch = create_balanced_pairs(
                    batch_size, x_test, y_test
                )
                test_pairs = torch.cat([X1_test, X2_test], dim=1)
                test_loss = criterion(model.kernel_function(test_pairs), Y_test_batch)
            test_loss_history.append(float(test_loss.cpu()))

            print(
                f"[NQE] Epoch {epoch + 1}/{num_epochs} "
                f"train_loss={train_loss_history[-1]:.6f} "
                f"test_loss={test_loss_history[-1]:.6f}"
            )

    print(
        f"[NQE] Training took {train_timer.time_s:.1f}s "
        f"({train_timer.time_per_epoch_s:.4f}s/epoch), "
        f"peak RSS {train_timer.peak_rss_mb:.1f} MB"
    )

    history = {"train_loss": train_loss_history, "test_loss": test_loss_history}

    Y_total = torch.cat([y_train, y_test], dim=0)
    X1_total, X2_total, Y_total_batch = create_balanced_pairs(batch_size, X, Y_total)
    with torch.no_grad():
        total_outputs = model.kernel_function(torch.cat([X1_total, X2_total], dim=1))
    loss_summary = {
        "train_loss": train_loss_history[-1],
        "test_loss": test_loss_history[-1],
        "total_loss": float(criterion(total_outputs, Y_total_batch)),
    }
    accuracy_summary = _qksvm_accuracies(model, x_train, y_train, x_test, y_test)

    return model, history, {**loss_summary, **accuracy_summary, **train_timer.as_dict("train")}


def _train_egas(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    num_modes_encoder: int,
    num_photons_encoder: int,
    num_epochs: int,
):
    """Build and train the EGAS model exactly as in run_dataset_complexity.py."""
    computation_space = ml.ComputationSpace.FOCK

    X_flat = _flatten(X)
    x_train_flat = _flatten(x_train)
    x_test_flat = _flatten(x_test)
    y_train_flat = y_train

    pool = build_token_pool(num_modes_encoder, X_flat.shape[-1])

    all_indices = np.arange(len(x_train_flat))
    search_idx, val_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train.cpu().numpy(),
    )

    x_search = x_train_flat[search_idx]
    y_search = y_train_flat[search_idx]
    x_val = x_train_flat[val_idx]
    y_val = y_train_flat[val_idx]

    rng = np.random.default_rng(SEED)
    search_samples = min(750, len(x_search))
    idx = rng.choice(len(x_search), search_samples, replace=False)
    Xe = x_search[idx]
    ye = y_search[idx]

    with _StageTimer(num_epochs=num_epochs) as search_timer:
        gpt, history_raw, buffer = run_photonic_egas(
            pool,
            Xe,
            ye,
            num_modes_encoder,
            seq_len=max(28, num_modes_encoder * 3 + 4),
            num_photons=num_photons_encoder,
            computation_space=computation_space,
            n_iters=num_epochs,
            n_candidates=12,
            select_k=6,
            lr=5e-5,
            n_layers=1,
            n_heads=2,
            n_embd=32,
            dropout=0.2,
            temp_max=10.0,
            temp_min=0.1,
            seed=SEED,
            device=X_flat.device,
            log_every=50,
        )
    print(
        f"[EGAS] Search took {search_timer.time_s:.1f}s "
        f"({search_timer.time_per_epoch_s:.4f}s/iter), "
        f"peak RSS {search_timer.peak_rss_mb:.1f} MB"
    )

    G_ids, _ = unique_sorted_candidates(buffer, top=5, bottom=0)
    with _StageTimer(num_epochs=EGAS_REFINE_EPOCHS * len(G_ids)) as refine_timer:
        refined = refine_candidates(
            G_ids,
            pool,
            Xe,
            ye,
            num_modes_encoder,
            num_photons=num_photons_encoder,
            computation_space=computation_space,
            device=X_flat.device,
            epochs=EGAS_REFINE_EPOCHS,
            batch_samples=64,
            lr=5e-4,
            seed=SEED,
        )
    print(
        f"[EGAS] Bias refinement of {len(G_ids)} candidates took {refine_timer.time_s:.1f}s "
        f"({refine_timer.time_per_epoch_s:.4f}s/epoch across all candidates), "
        f"peak RSS {refine_timer.peak_rss_mb:.1f} MB"
    )
    if not refined:
        raise RuntimeError("EGAS produced no refined candidates")

    X_val = _flatten(x_val).to(X_flat.device)
    Y_val = y_val.to(X_flat.device)

    for candidate in refined:
        candidate["encoder"].eval()
        with torch.no_grad():
            states = candidate["encoder"](X_val)
            candidate["validation_energy"] = pairwise_energy(states, Y_val).item()

    best = min(refined, key=lambda candidate: candidate["validation_energy"])

    model = EGASEncoder(best["encoder"])

    history = {
        "iter": history_raw["iter"],
        "loss": history_raw["loss"],
        "min_energy": history_raw["min_energy"],
        "mean_energy": history_raw["mean_energy"],
    }

    def _energy(x_flat, y):
        with torch.no_grad():
            states = model(x_flat)
        return float(pairwise_energy(states, y))

    Y_flat_total = torch.cat([y_train, y_test], dim=0)
    loss_summary = {
        "train_loss": _energy(x_train_flat, y_train),
        "test_loss": _energy(x_test_flat, y_test),
        "total_loss": _energy(_flatten(X), Y_flat_total),
    }
    accuracy_summary = _qksvm_accuracies(
        model, x_train_flat, y_train, x_test_flat, y_test
    )
    timing_summary = {
        **search_timer.as_dict("search"),
        **refine_timer.as_dict("refine"),
        "total_time_s": search_timer.time_s + refine_timer.time_s,
    }
    split_summary = {
        "n_search_pool": len(x_search),
        "n_search_samples": len(Xe),
        "n_val": len(x_val),
    }

    return model, history, {**loss_summary, **accuracy_summary, **timing_summary, **split_summary}


def plot_training_curves(
    nqe_history: dict, egas_history: dict, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(nqe_history["train_loss"], label="Train loss", color="tab:blue")
    axes[0].plot(nqe_history["test_loss"], label="Test loss", color="tab:orange")
    axes[0].set_title("NQE Training Curve", fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    axes[0].set_ylabel("MSE loss", fontsize=LABEL_FONTSIZE)
    axes[0].tick_params(labelsize=TICK_FONTSIZE)
    axes[0].legend(fontsize=LEGEND_FONTSIZE)
    axes[0].grid(alpha=0.3)

    axes[1].plot(egas_history["iter"], egas_history["loss"], label="GPT loss", color="tab:blue")
    axes[1].plot(
        egas_history["iter"],
        egas_history["min_energy"],
        label="Min energy",
        color="tab:green",
    )
    axes[1].plot(
        egas_history["iter"],
        egas_history["mean_energy"],
        label="Mean energy",
        color="tab:orange",
    )
    axes[1].set_title("EGAS Training Curve", fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight="bold")
    axes[1].set_xlabel("Search iteration", fontsize=LABEL_FONTSIZE)
    axes[1].set_ylabel("Loss / Energy", fontsize=LABEL_FONTSIZE)
    axes[1].tick_params(labelsize=TICK_FONTSIZE)
    axes[1].legend(fontsize=LEGEND_FONTSIZE)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"NQE vs EGAS training — {DATASET_NAME}-{FEATURE_REDUCTION}-{len(CLASSES) if CLASSES else '?'}c",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train, x_test, y_train, y_test = data_load_and_process(
        dataset=DATASET_NAME,
        classes=CLASSES,
        feature_reduction=FEATURE_REDUCTION,
        noise_generated=NOISE_GENERATED,
    )
    X = torch.cat((x_train, x_test), 0)
    Y = torch.cat((y_train, y_test), 0)
    n_features = int(np.prod(X.shape[1:]))

    num_classes_count = (
        len(CLASSES) if CLASSES is not None else len(torch.unique(y_train))
    )
    num_modes_encoder, num_photons_encoder = find_mode_photon_config(
        max(n_features // 2 + 1, num_classes_count)
    )

    print("Training NQE...")
    nqe_model, nqe_history, nqe_summary = _train_nqe(
        x_train,
        y_train,
        x_test,
        y_test,
        X,
        n_features,
        num_modes_encoder,
        num_photons_encoder,
        NQE_NUM_EPOCHS,
    )
    print(f"NQE summary: {nqe_summary}")

    print("Training EGAS...")
    egas_model, egas_history, egas_summary = _train_egas(
        x_train,
        y_train,
        x_test,
        y_test,
        X,
        Y,
        num_modes_encoder,
        num_photons_encoder,
        EGAS_NUM_EPOCHS,
    )
    print(f"EGAS summary: {egas_summary}")

    plot_training_curves(nqe_history, egas_history, OUTPUT_PDF)

    OUTPUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "config": {
                    "dataset_name": DATASET_NAME,
                    "feature_reduction": FEATURE_REDUCTION,
                    "classes": list(CLASSES) if CLASSES is not None else None,
                    "nqe_num_epochs": NQE_NUM_EPOCHS,
                    "egas_num_epochs": EGAS_NUM_EPOCHS,
                },
                "dataset_sizes": {
                    "n_train": len(x_train),
                    "n_test": len(x_test),
                    "n_total": len(X),
                },
                "nqe": nqe_summary,
                "egas": egas_summary,
            },
            indent=2,
        )
    )
    print(f"Saved training curves to {OUTPUT_PDF}")
    print(f"Saved summary to {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
