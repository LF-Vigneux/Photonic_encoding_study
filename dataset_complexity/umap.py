import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
import builtins
import warnings
from typing import Any

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


from nn_embedding.lib.merlin_based_model import (
    NeuralEmbeddingMerLinKernel,
)
from nn_embedding.utils.merlin_model_utils import assign_params  # noqa: E402
from nn_embedding.utils.utils import (  # noqa: E402
    calculate_distance,
    state_vector_to_density_matrix,
)

import plotly.graph_objects as go


def _import_umap_without_tensorflow():
    """Import umap-learn while intentionally disabling TensorFlow-dependent extras.

    The default `umap` package import attempts to import ParametricUMAP, which in
    turn imports TensorFlow. On some macOS setups this can block for a long time
    during startup even though we only need non-parametric UMAP.
    """

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ImportError("TensorFlow disabled for non-parametric UMAP usage")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _guarded_import
    try:
        import umap as _umap
    finally:
        builtins.__import__ = original_import

    return _umap


umap = _import_umap_without_tensorflow()


def _transform_with_embedder(
    X: torch.Tensor, embedder: Any | None = None
) -> torch.Tensor:
    if embedder is None:
        return X

    if isinstance(embedder, NeuralEmbeddingMerLinKernel):

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.classical_model = embedder.classical_encoder

            def forward(self, x: torch.Tensor):
                params = self.classical_model(x)
                with torch.no_grad():
                    output = assign_params(embedder.quantum_embedding_layer, params)
                return output

        encoder = Encoder()
        return encoder(X)

    return embedder(X)


def _to_numpy_array(data: torch.Tensor | NDArray) -> NDArray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _to_umap_feature_matrix(data: NDArray) -> NDArray:
    """Return a 2D real-valued matrix suitable for UMAP.

    If the input is complex, features are expanded as
    [Re(x0), Im(x0), Re(x1), Im(x1), ...].
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if np.iscomplexobj(data):
        real = data.real
        imag = data.imag
        out = np.empty((data.shape[0], data.shape[1] * 2), dtype=np.float64)
        out[:, 0::2] = real
        out[:, 1::2] = imag
        return out

    return np.asarray(data, dtype=np.float64)


def _prepare_umap_inputs(X: torch.Tensor, Y: torch.Tensor) -> tuple[NDArray, NDArray]:
    X_np = _to_numpy_array(X)
    X_features = _to_umap_feature_matrix(X_np)
    return X_features, _to_numpy_array(Y).reshape(-1)


def _prepare_umap_distance_inputs(
    X: torch.Tensor, Y: torch.Tensor, embedder: Any
) -> tuple[NDArray, NDArray]:
    encoded_states = _transform_with_embedder(X, embedder)
    if not isinstance(encoded_states, torch.Tensor):
        encoded_states = torch.as_tensor(encoded_states)

    # Fast path: encoded pure states (N, D). For pure states,
    # trace distance is sqrt(1 - |<psi_i|psi_j>|^2), so we can compute all
    # pairwise distances in one vectorized pass.
    if encoded_states.ndim == 2:
        psis = encoded_states.to(torch.complex128)
        norms = torch.linalg.norm(psis, dim=1, keepdim=True)
        psis = psis / torch.clamp(norms, min=1e-12)
        gram = psis @ torch.conj(psis).T
        fidelities = torch.abs(gram) ** 2
        distances = torch.sqrt(torch.clamp(1.0 - fidelities.real, min=0.0))
        distance_matrix = distances.detach().cpu().numpy().astype(np.float64)
        np.fill_diagonal(distance_matrix, 0.0)
        return distance_matrix, _to_numpy_array(Y).reshape(-1)

    rhos = state_vector_to_density_matrix(encoded_states)
    if not isinstance(rhos, torch.Tensor):
        rhos = torch.as_tensor(rhos)

    rhos = rhos.to(torch.complex128)
    n_samples = int(rhos.shape[0])
    distance_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = calculate_distance(rhos[i], rhos[j], distance="Trace")
            dist_value = float(dist.item() if hasattr(dist, "item") else dist)
            distance_matrix[i, j] = dist_value
            distance_matrix[j, i] = dist_value

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix, _to_numpy_array(Y).reshape(-1)


def _class_color_metadata(labels: NDArray) -> tuple[NDArray, list[str], list[float]]:
    unique_labels = np.unique(labels)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    color_values = np.asarray([label_to_index[label] for label in labels], dtype=float)
    tick_text = [str(label) for label in unique_labels]
    tick_values = [float(index) for index in range(len(unique_labels))]
    return color_values, tick_text, tick_values


def _annotation_block(dataset_name: str, embedding_strategy: str) -> str:
    dataset_line = dataset_name if dataset_name else "____________"
    strategy_line = embedding_strategy if embedding_strategy else "____________"
    return f"Dataset: {dataset_line}<br>" f"Embedding: {strategy_line}"


def _slugify(value: str) -> str:
    return "_".join(str(value).strip().lower().split())


def umap_data(
    X: torch.Tensor,
    Y: torch.Tensor,
    embedder=None,
    umap_state: int = 42,
    n_neighbors: int = 15,
    n_epochs: int = 200,
) -> tuple[NDArray, NDArray]:
    if embedder is None:
        X_to_use, labels = _prepare_umap_inputs(X, Y)
        metric = "euclidean"
    else:
        X_to_use, labels = _prepare_umap_distance_inputs(X, Y, embedder)
        metric = "precomputed"

    reducer_2 = umap.UMAP(
        n_components=2,
        random_state=umap_state,
        n_neighbors=n_neighbors,
        n_epochs=n_epochs,
        metric=metric,
        n_jobs=1,
    )
    reducer_3 = umap.UMAP(
        n_components=3,
        random_state=umap_state,
        n_neighbors=n_neighbors,
        n_epochs=n_epochs,
        metric=metric,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="using precomputed metric; inverse_transform will be unavailable",
            category=UserWarning,
        )
        reduced_2d = reducer_2.fit_transform(X_to_use, y=labels)
        reduced_3d = reducer_3.fit_transform(X_to_use, y=labels)
    return reduced_2d, reduced_3d


def plot_umap_2d_interactive(
    reduced_2d: torch.Tensor | NDArray,
    Y: torch.Tensor | NDArray,
    *,
    dataset_name: str = "",
    embedding_strategy: str = "",
    output_path: str | Path | None = None,
    marker_size: int = 8,
) -> Any:
    points = _to_numpy_array(reduced_2d)
    labels = _to_numpy_array(Y).reshape(-1)
    color_values, tick_text, tick_values = _class_color_metadata(labels)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": color_values,
                    "colorscale": "Turbo",
                    "opacity": 0.82,
                    "showscale": True,
                    "colorbar": {
                        "title": "Class",
                        "tickvals": tick_values,
                        "ticktext": tick_text,
                        "len": 0.82,
                        "x": 1.1,
                    },
                },
                text=[f"Class: {label}" for label in labels],
                hovertemplate="UMAP 1: %{x:.3f}<br>UMAP 2: %{y:.3f}<br>%{text}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="UMAP projection (2D)",
        width=1100,
        height=700,
        template="plotly_white",
        margin={"l": 60, "r": 280, "t": 70, "b": 60},
        xaxis={"title": "UMAP 1", "domain": [0.0, 0.78]},
        yaxis={"title": "UMAP 2"},
        annotations=[
            {
                "x": 0.84,
                "y": 0.92,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "align": "left",
                "text": _annotation_block(dataset_name, embedding_strategy),
                "bordercolor": "#BFC5D2",
                "borderwidth": 1,
                "borderpad": 8,
                "bgcolor": "rgba(255,255,255,0.92)",
                "font": {"size": 13},
            }
        ],
    )

    if output_path is not None:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def plot_umap_3d_interactive(
    reduced_3d: torch.Tensor | NDArray,
    Y: torch.Tensor | NDArray,
    *,
    dataset_name: str = "",
    embedding_strategy: str = "",
    output_path: str | Path | None = None,
    marker_size: int = 5,
) -> Any:
    points = _to_numpy_array(reduced_3d)
    labels = _to_numpy_array(Y).reshape(-1)
    color_values, tick_text, tick_values = _class_color_metadata(labels)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": color_values,
                    "colorscale": "Turbo",
                    "opacity": 0.8,
                    "showscale": True,
                    "colorbar": {
                        "title": "Class",
                        "tickvals": tick_values,
                        "ticktext": tick_text,
                        "len": 0.82,
                        "x": 1.12,
                    },
                },
                text=[f"Class: {label}" for label in labels],
                hovertemplate=(
                    "UMAP 1: %{x:.3f}<br>UMAP 2: %{y:.3f}<br>"
                    "UMAP 3: %{z:.3f}<br>%{text}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="UMAP projection (3D)",
        width=1200,
        height=750,
        template="plotly_white",
        margin={"l": 20, "r": 310, "t": 70, "b": 20},
        scene={
            "domain": {"x": [0.0, 0.78], "y": [0.0, 1.0]},
            "xaxis": {"title": "UMAP 1"},
            "yaxis": {"title": "UMAP 2"},
            "zaxis": {"title": "UMAP 3"},
        },
        annotations=[
            {
                "x": 0.84,
                "y": 0.92,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "align": "left",
                "text": _annotation_block(dataset_name, embedding_strategy),
                "bordercolor": "#BFC5D2",
                "borderwidth": 1,
                "borderpad": 8,
                "bgcolor": "rgba(255,255,255,0.92)",
                "font": {"size": 13},
            }
        ],
    )

    if output_path is not None:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def save_umap_plots(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    embedder: Any | None = None,
    dataset_name: str = "",
    embedding_strategy: str = "",
    output_dir: str | Path | None = None,
    umap_state: int = 42,
    umap_n_neighbors: int = 15,
    umap_n_epochs: int = 200,
    save_2d: bool = True,
    save_3d: bool = True,
) -> dict[str, Path]:
    reduced_2d, reduced_3d = umap_data(
        X,
        Y,
        embedder=embedder,
        umap_state=umap_state,
        n_neighbors=umap_n_neighbors,
        n_epochs=umap_n_epochs,
    )

    if output_dir is None:
        output_root = Path.cwd()
    else:
        output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_slug = _slugify(dataset_name or "dataset")
    encoding_slug = _slugify(embedding_strategy or "embedding")
    base_name = f"u_map_{encoding_slug}_{dataset_slug}"

    saved_paths: dict[str, Path] = {}

    if save_2d:
        output_path_2d = output_root / f"{base_name}_2d.html"
        plot_umap_2d_interactive(
            reduced_2d,
            Y,
            dataset_name=dataset_name,
            embedding_strategy=embedding_strategy,
            output_path=output_path_2d,
        )
        saved_paths["2d"] = output_path_2d

    if save_3d:
        output_path_3d = output_root / f"{base_name}_3d.html"
        plot_umap_3d_interactive(
            reduced_3d,
            Y,
            dataset_name=dataset_name,
            embedding_strategy=embedding_strategy,
            output_path=output_path_3d,
        )
        saved_paths["3d"] = output_path_3d

    return saved_paths
