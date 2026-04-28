"""Data loading functions.

Adapted from the original repository:
https://github.com/takh04/neural-quantum-embedding
"""

import numpy as np
import torch
from sklearn.datasets import fetch_openml, make_moons, make_circles
from merlin.datasets import spiral
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import medmnist
import numpy as np
import torch
from medmnist import INFO
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor


def _load_datasets(
    dataset: str,
    n_features_generated: int = 2,
    noise_generated: float = 0.0,
    n_samples_generated: int = 1000,
    n_classes_spiral: int = 2,
):
    open_ml = False
    if dataset == "mnist":
        openml_name = "mnist_784"
        open_ml = True
    elif dataset == "fashion":
        openml_name = "Fashion-MNIST"
        open_ml = True
    elif dataset == "kmnist":
        openml_name = "Kuzushiji-MNIST"
        open_ml = True

    if open_ml:
        data, targets = fetch_openml(
            openml_name,
            version=1,
            return_X_y=True,
            as_frame=False,
        )

        data = data.astype(np.float32).reshape(-1, 28, 28)
        targets = targets.astype(np.int64)

        return train_test_split(
            data,
            targets,
            test_size=10000,
            random_state=42,
            stratify=targets,
        )
    else:
        transform_32 = Compose([Resize((32, 32)), ToTensor()])
        root = "datasets/data"
        # ---- CIFAR-10 ----
        if dataset in ["cifar10", "cifar-10"]:
            train_base = datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform_32
            )
            eval_base = datasets.CIFAR10(
                root=root, train=False, download=True, transform=transform_32
            )

            x_train = np.stack([img.numpy() for img, _ in train_base])
            y_train = np.array([label for _, label in train_base], dtype=np.int64)
            x_test = np.stack([img.numpy() for img, _ in eval_base])
            y_test = np.array([label for _, label in eval_base], dtype=np.int64)
            return x_train, x_test, y_train, y_test

        # ---- EuroSAT ----
        if dataset in ["eurosat", "euro_sat", "euro-sat"]:
            base = datasets.EuroSAT(root=root, download=True, transform=transform_32)

            x_all = np.stack([img.numpy() for img, _ in base])
            y_all = np.array([label for _, label in base], dtype=np.int64)
            return train_test_split(
                x_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )

        # ---- PathMNIST ----
        if dataset in ["pathmnist", "path_mnist", "path-mnist"]:
            info = INFO["pathmnist"]
            DataClass = getattr(medmnist, info["python_class"])

            train_base = DataClass(
                split="train", download=True, root=root, transform=transform_32
            )
            eval_base = DataClass(
                split="test", download=True, root=root, transform=transform_32
            )

            x_train = np.stack([img.numpy() for img, _ in train_base])
            y_train = np.array([label for _, label in train_base], dtype=np.int64)
            x_test = np.stack([img.numpy() for img, _ in eval_base])
            y_test = np.array([label for _, label in eval_base], dtype=np.int64)
            return x_train, x_test, y_train, y_test

        # ---- Spiral ----
        if dataset == "spiral":
            X, Y = spiral.get_data(
                num_instances=n_samples_generated,
                num_features=n_features_generated,
                num_classes=n_classes_spiral,
            )
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # ---- Moons ----
        if dataset == "moons":
            X, y = make_moons(n_samples=n_samples_generated, noise=noise_generated)
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # ---- Circles ----
        if dataset == "circles":
            X, y = make_circles(n_samples=n_samples_generated, noise=noise_generated)
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        raise ValueError(
            f"Unsupported dataset: '{dataset}'. "
            "Choose from: mnist, fashion, kmnist, cifar10, eurosat, pathmnist, spiral, moons, circles."
        )


def _minmax_normalize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature min-max normalization to [0, 1], fitted on x_train only."""
    original_shape_train = x_train.shape
    original_shape_test = x_test.shape
    x_train_2d = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test_2d = x_test.reshape(len(x_test), -1).astype(np.float32)

    feat_min = x_train_2d.min(axis=0)
    feat_max = x_train_2d.max(axis=0)
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0  # avoid division by zero for constant features

    x_train_norm = (x_train_2d - feat_min) / denom
    x_test_norm = np.clip((x_test_2d - feat_min) / denom, 0.0, 1.0)

    return (
        x_train_norm.reshape(original_shape_train),
        x_test_norm.reshape(original_shape_test),
    )


def _limit_samples_per_class(
    x: np.ndarray,
    y: np.ndarray,
    samples_per_class: int | None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_class is None:
        return x, y

    rng = np.random.default_rng(random_state)
    selected_indices = []

    for class_label in np.unique(y):
        class_indices = np.flatnonzero(y == class_label)
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Requested {samples_per_class} samples for class {class_label}, "
                f"but only {len(class_indices)} are available."
            )
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.append(chosen)

    selected_indices = np.sort(np.concatenate(selected_indices))
    return x[selected_indices], y[selected_indices]


def data_load_and_process(
    dataset,
    feature_reduction: int | None = None,
    classes=(0, 1),
    samples_per_class: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 42,
):
    """
    This part of the code was originally written to use Brain signal dataset.
    This implementation is currently out of interest; hence commented out.
    Will include this back later when needed.

    if dataset == 'signal':
        dataset_signal = pd.read_csv('/data/ROI_' +str(ROI)+ '_df_length256_zero_padding.csv')

        dataset_value = dataset_signal.iloc[:,:-1]
        dataset_label = dataset_signal.iloc[:,-1]

        x_train, x_test, y_train, y_test = train_test_split(dataset_value, dataset_label, test_size=0.2, shuffle=True,
                                                            stratify=dataset_label, random_state=10)

        x_train, x_test, y_train, y_test =\
            x_train.values.tolist(), x_test.values.tolist(), y_train.values.tolist(), y_test.values.tolist()
        y_train = [1 if y == 1 else -1 for y in y_train]
        y_test = [1 if y ==1 else -1 for y in y_test]
    """

    x_train, x_test, y_train, y_test = _load_datasets(dataset)

    if classes is not None:
        mask_train = np.zeros(len(y_train), dtype=bool)
        mask_test = np.zeros(len(y_test), dtype=bool)
        for c in classes:
            mask_train |= y_train == c
            mask_test |= y_test == c
        train_filter_tf = np.where(mask_train)
        test_filter_tf = np.where(mask_test)
        x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
        x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

        label_map = {c: i for i, c in enumerate(classes)}
        y_train = np.array([label_map[c] for c in y_train], dtype=np.int64)
        y_test = np.array([label_map[c] for c in y_test], dtype=np.int64)

    x_train, y_train = _limit_samples_per_class(x_train, y_train, samples_per_class)
    x_test, y_test = _limit_samples_per_class(
        x_test, y_test, samples_per_class, random_state=43
    )

    if not feature_reduction:
        x_train, x_test = _minmax_normalize(x_train, x_test)
        if shuffle:
            rng = np.random.default_rng(shuffle_seed)
            train_perm = rng.permutation(len(x_train))
            test_perm = rng.permutation(len(x_test))
            x_train, y_train = x_train[train_perm], y_train[train_perm]
            x_test, y_test = x_test[test_perm], y_test[test_perm]
        return (
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(x_test, dtype=torch.float32),
            torch.as_tensor(y_train),
            torch.as_tensor(y_test),
        )

    if isinstance(feature_reduction, int):
        x_train_flat = x_train.reshape(len(x_train), -1)
        x_test_flat = x_test.reshape(len(x_test), -1)

        pca = PCA(feature_reduction)
        X_train = pca.fit_transform(x_train_flat)
        X_test = pca.transform(x_test_flat)

        x_train, x_test = _minmax_normalize(X_train, X_test)
        if shuffle:
            rng = np.random.default_rng(shuffle_seed)
            train_perm = rng.permutation(len(x_train))
            test_perm = rng.permutation(len(x_test))
            x_train, y_train = x_train[train_perm], y_train[train_perm]
            x_test, y_test = x_test[test_perm], y_test[test_perm]
        return (
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(x_test, dtype=torch.float32),
            torch.as_tensor(y_train),
            torch.as_tensor(y_test),
        )
