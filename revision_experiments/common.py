"""Shared helpers for the revision experiments (Revisions_1018.md, sections 2.5-2.9).

Nothing here modifies the repository code: it only imports the encoders, the
metric functions and the data loader that already exist.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import merlin as ml  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from data.loader import data_load_and_process, _minmax_normalize  # noqa: E402
from encodings_merlin.encoding_layers import (  # noqa: E402
    angle_encoding_layer,
    dense_angle_encoding_layer,
    fourier_basis_layer,
    AmplitudeEncoder,
    DenseAmplitudeEncoder,
)
from encodings_merlin.utils import find_mode_photon_config  # noqa: E402
from nn_embedding.lib.merlin_based_model import NeuralEmbeddingMerLinKernel  # noqa: E402
from nn_embedding.utils.merlin_model_utils import assign_params  # noqa: E402
from nn_embedding.utils.utils import TransparentModel  # noqa: E402

# Encodings that need no training. NQE and EGAS are excluded on purpose: they
# are trained per dataset and are handled by the existing pipeline.
STATIC_ENCODINGS = ["angle", "dense_angle", "fourier", "amplitude", "dense_amplitude"]


# ── Config / output helpers ───────────────────────────────────────────────────
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        p = Path(__file__).resolve().parent / "configs" / p.name
    cfg = json.loads(p.read_text())
    cfg["_config_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True).encode()
    ).hexdigest()
    cfg["_started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return cfg


def out_dir(cfg: dict, name: str) -> Path:
    d = PROJECT_ROOT / cfg.get("outdir", "results/revisions") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


def set_seed(seed: int) -> None:
    np.random.seed(seed)  # randomize_entangling draws from np.random
    torch.manual_seed(seed)


# ── Data ──────────────────────────────────────────────────────────────────────
def _openml(name: str, **kw):
    from sklearn.datasets import fetch_openml

    d = fetch_openml(name, as_frame=False, return_X_y=True, **kw)
    X = np.asarray(d[0], dtype=np.float32)
    y = np.unique(d[1], return_inverse=True)[1].astype(np.int64)
    return X, y


def _raw_heldout(name: str, seed: int = 0):
    """Datasets that the repository loader does not know."""
    if name == "digits":
        from sklearn.datasets import load_digits

        X, y = load_digits(return_X_y=True)
        X, y = X.astype(np.float32), y.astype(np.int64)
    elif name == "sonar":
        X, y = _openml("sonar", version=1)
    elif name == "ionosphere":
        X, y = _openml("ionosphere", version=1)
    elif name == "credit_default":
        X, y = _openml("default-of-credit-card-clients", version=1)
    else:
        raise ValueError(name)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def parity_dataset(n_samples: int, n_features: int, order: int, seed: int = 0):
    """Label = parity of `order` thresholded features; features are U[0,1]^d.

    All label information sits in an order-`order` interaction: every
    lower-order marginal (and any linear model) is at chance by construction.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features)).astype(np.float32)
    bits = (X[:, :order] > 0.5).astype(np.int64)
    y = bits.sum(axis=1) % 2
    return X, y.astype(np.int64)


def load_dataset(
    name: str,
    feature_reduction: int = 8,
    classes=None,
    max_per_split: int | None = None,
    seed: int = 0,
    parity_order: int = 3,
    n_samples_synth: int = 1000,
):
    """Return (x_train, x_test, y_train, y_test) as torch tensors, PCA(d) + min-max.

    Repository datasets go through `data_load_and_process` unchanged. Held-out
    datasets use the same preprocessing (PCA fitted on train, min-max fitted on
    train).
    """
    if name.startswith("parity"):
        X, y = parity_dataset(n_samples_synth, feature_reduction, parity_order, seed)
        xtr, xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        out = [torch.as_tensor(a) for a in (xtr, xte, ytr, yte)]
    elif name in {"digits", "sonar", "ionosphere", "credit_default"}:
        xtr, xte, ytr, yte = _raw_heldout(name)
        if classes is not None:
            keep = {c: i for i, c in enumerate(classes)}
            mtr = np.isin(ytr, list(keep))
            mte = np.isin(yte, list(keep))
            xtr, ytr = xtr[mtr], np.array([keep[c] for c in ytr[mtr]])
            xte, yte = xte[mte], np.array([keep[c] for c in yte[mte]])
        d = min(feature_reduction, xtr.shape[1])
        pca = PCA(d).fit(xtr.reshape(len(xtr), -1))
        xtr = pca.transform(xtr.reshape(len(xtr), -1))
        xte = pca.transform(xte.reshape(len(xte), -1))
        xtr, xte = _minmax_normalize(xtr, xte)
        out = [torch.as_tensor(xtr, dtype=torch.float32),
               torch.as_tensor(xte, dtype=torch.float32),
               torch.as_tensor(ytr), torch.as_tensor(yte)]
    else:
        out = list(
            data_load_and_process(
                dataset=name,
                classes=tuple(classes) if classes is not None else None,
                feature_reduction=feature_reduction,
            )
        )
    if max_per_split is not None:
        rng = np.random.default_rng(seed)
        for k, n in ((0, max_per_split), (1, max_per_split)):
            if len(out[k]) > n:
                idx = np.sort(rng.choice(len(out[k]), n, replace=False))
                out[k], out[k + 2] = out[k][idx], out[k + 2][idx]
    return tuple(out)


# ── Encoders ──────────────────────────────────────────────────────────────────
class StateEncoder(nn.Module):
    """Uniform wrapper: x (N,d) -> normalised complex states (N,D).

    Matches what the repo metric functions call `encoder(x)`, so it can be
    passed as `embedder` to `encoded_states_classes_overlap`,
    `kernel_spectrum_flatness`, ... for non-NQE encodings.
    """

    def __init__(self, name: str, module: nn.Module, meta: dict):
        super().__init__()
        self.name = name
        self.module = module
        self.meta = meta
        self.is_kernel_model = isinstance(module, NeuralEmbeddingMerLinKernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        was_1d = x.dim() == 1
        with torch.no_grad():
            if self.is_kernel_model:
                params = self.module.classical_encoder(x)
                s = assign_params(self.module.quantum_embedding_layer, params)
            else:
                s = self.module(x)
        s = s.reshape(1, -1) if s.dim() == 1 else s
        s = s.to(torch.complex128)
        s = s / torch.linalg.norm(s, dim=1, keepdim=True).clamp_min(1e-12)
        return s[0] if was_1d else s


def build_encoders(
    n_features: int,
    n_classes: int,
    names=STATIC_ENCODINGS,
    computation_space=ml.ComputationSpace.UNBUNCHED,
    randomize_entangling: bool = True,
    seed: int = 0,
    num_qubits_per_feature_fourier: int = 1,
) -> dict[str, StateEncoder]:
    """Build the static encoders exactly as run_dataset_complexity.py does."""
    set_seed(seed)
    out = {}
    n_ph_angle = n_features // 2
    for name in names:
        if name == "angle":
            layer = angle_encoding_layer(
                num_features=n_features, num_modes=None, num_photons=n_ph_angle,
                computation_space=computation_space,
                randomize_entangling=randomize_entangling,
            )
            mod = NeuralEmbeddingMerLinKernel(TransparentModel(), layer)
            meta = {"n_modes": n_features + 1, "n_photons": n_ph_angle}
        elif name == "dense_angle":
            layer = dense_angle_encoding_layer(num_features=n_features, num_modes=None)
            mod = NeuralEmbeddingMerLinKernel(TransparentModel(), layer)
            meta = {"n_modes": 2 * int(np.ceil(n_features / 2)),
                    "n_photons": int(np.ceil(n_features / 2))}
        elif name == "fourier":
            layer = fourier_basis_layer(
                num_features=n_features,
                num_qubits_per_feature=num_qubits_per_feature_fourier,
            )
            mod = NeuralEmbeddingMerLinKernel(TransparentModel(), layer)
            meta = {"n_modes": None, "n_photons": None}
        elif name == "amplitude":
            m, p = find_mode_photon_config(max(n_features, n_classes))
            mod = AmplitudeEncoder(m, p, computation_space)
            meta = {"n_modes": m, "n_photons": p}
        elif name == "dense_amplitude":
            m, p = find_mode_photon_config(max(n_features // 2 + 1, n_classes))
            mod = DenseAmplitudeEncoder(m, p, computation_space)
            meta = {"n_modes": m, "n_photons": p}
        else:
            raise ValueError(name)
        out[name] = StateEncoder(name, mod, meta)
    return out


def encode_all(enc: StateEncoder, X: torch.Tensor, batch: int = 256) -> torch.Tensor:
    """Encode in batches (the NQE-style wrapper loops per sample internally)."""
    return torch.cat([enc(X[i : i + batch]) for i in range(0, len(X), batch)], 0)


# ── State-only metrics (no circuit information needed) ────────────────────────
def kernel_from_states(S: torch.Tensor, T: torch.Tensor | None = None) -> np.ndarray:
    T = S if T is None else T
    return (S.conj() @ T.T).abs().pow(2).real.cpu().numpy()


def m1_support_dim(S: torch.Tensor, eps: float = 1e-8) -> float:
    rho = (S.T @ S.conj()) / S.shape[0]
    ev = torch.linalg.eigvalsh(rho).clamp_min(0).cpu().numpy()
    return float(np.sum(ev / (ev + eps)))


def m4_kernel_flatness(S: torch.Tensor) -> float:
    ev = np.clip(np.linalg.eigvalsh(kernel_from_states(S)), 0, None)
    return float(ev.sum() ** 2 / np.sum(ev**2))


def m7_class_overlap(S: torch.Tensor, y: torch.Tensor) -> float:
    """Mean over class pairs of 1 - trace distance of class-mean states
    (same definition as `encoded_states_classes_overlap`)."""
    classes = torch.unique(y).tolist()
    rhos = []
    for c in classes:
        Sc = S[y == c]
        rhos.append((Sc.T @ Sc.conj()) / Sc.shape[0])
    vals = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ev = torch.linalg.eigvalsh(rhos[i] - rhos[j])
            vals.append(1.0 - 0.5 * float(ev.abs().sum()))
    return float(np.mean(vals))
