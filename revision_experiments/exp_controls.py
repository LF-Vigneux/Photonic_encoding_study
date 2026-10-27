"""Revisions 2.7: closed-form validation, random-label control, parity family.

  closed_form   metrics vs analytic values on synthetic state ensembles
  random_label  M7, Wasserstein and kernel-SVM accuracy under permuted labels
  parity        synthetic family whose label lives only in order-k interactions

Usage:  python revision_experiments/exp_controls.py --config controls.json
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as C
import merlin as ml
from dataset_complexity.utils import (
    hilbert_space_support_dim,
    kernel_spectrum_flatness,
    encoded_states_classes_overlap,
    kl_div,
    entanglement_entropy,
    dataset_wasserstein,
)
from egas.photonic_kernel_svm import C_SVM


# ── Synthetic ensembles ───────────────────────────────────────────────────────
class TableEncoder(nn.Module):
    """x[:, 0] = (i + 0.5)/N selects row i of a fixed state table. Quacks like a
    non-NQE encoder for the repo metric functions."""

    def __init__(self, S: torch.Tensor, num_modes=None, num_photons=None,
                 computation_space=ml.ComputationSpace.DUAL_RAIL):
        super().__init__()
        self.S = S.to(torch.complex128)
        self.output_size = S.shape[1]
        self.num_modes, self.num_photons = num_modes, num_photons
        self.computation_space = computation_space

    def forward(self, x):
        idx = (x[..., 0] * self.S.shape[0]).floor().long().clamp(0, self.S.shape[0] - 1)
        return self.S[idx]


def ids(N, d=2):
    x = torch.zeros(N, d)
    x[:, 0] = (torch.arange(N) + 0.5) / N
    return x


def haar_states(N, D, rng):
    z = rng.normal(size=(N, D)) + 1j * rng.normal(size=(N, D))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return torch.as_tensor(z)


def product_states(thetas):  # (N,n) -> (N,2^n); cos t|0> + sin t|1> per qubit
    N, n = thetas.shape
    s = torch.ones(N, 1, dtype=torch.complex128)
    for q in range(n):
        q_state = torch.stack([torch.cos(thetas[:, q]), torch.sin(thetas[:, q])], 1).to(torch.complex128)
        s = (s[:, :, None] * q_state[:, None, :]).reshape(N, -1)
    return s


def ghz_states(thetas, n):  # cos t|0..0> + sin t|1..1>
    N = len(thetas)
    s = torch.zeros(N, 2**n, dtype=torch.complex128)
    s[:, 0], s[:, -1] = torch.cos(thetas), torch.sin(thetas)
    return s


def closed_form(cfg):
    rng = np.random.default_rng(cfg["seed"])
    N, tol = cfg["n_states"], cfg["tolerance"]
    rows = []

    def add(ens, metric, expected, observed, note=""):
        ok = abs(observed - expected) <= tol * max(1.0, abs(expected))
        rows.append(dict(ensemble=ens, metric=metric, expected=expected,
                         observed=float(observed), ok=bool(ok), note=note))

    eps = 1e-8
    # identical ensemble (two classes with the same state)
    D = 8
    s0 = haar_states(1, D, rng).repeat(N, 1)
    y = torch.arange(N) % 2
    enc = TableEncoder(s0); x = ids(N)
    add("identical", "M1 support dim", 1.0, hilbert_space_support_dim(x, enc, eps))
    add("identical", "M4 kernel flatness", 1.0, float(kernel_spectrum_flatness(x, enc)))
    add("identical", "M7 class overlap", 1.0, encoded_states_classes_overlap(x, y, enc))
    # orthogonal ensemble (class c -> basis state e_c)
    for nc in cfg["orthogonal_classes"]:
        S = torch.zeros(N, D, dtype=torch.complex128)
        y = torch.arange(N) % nc
        S[torch.arange(N), y] = 1
        enc = TableEncoder(S)
        add(f"orthogonal_{nc}cls", "M1 support dim", nc * (1 / nc) / (1 / nc + eps), hilbert_space_support_dim(x, enc, eps))
        add(f"orthogonal_{nc}cls", "M4 kernel flatness", float(nc), float(kernel_spectrum_flatness(x, enc)))
        add(f"orthogonal_{nc}cls", "M7 class overlap", 0.0, encoded_states_classes_overlap(x, y, enc))
    # Haar: E K = 1/D, E K^2 = 2/(D(D+1)) off-diagonal
    for D in cfg["haar_dims"]:
        S = haar_states(N, D, rng); enc = TableEncoder(S)
        offdiag = 2.0 / (D * (D + 1))
        pr = N**2 / (N + N * (N - 1) * offdiag)
        add(f"haar_D{D}", "M1 support dim (-> D)", float(D), hilbert_space_support_dim(x, enc, eps), "needs N >> D")
        add(f"haar_D{D}", "M4 kernel flatness", pr, float(kernel_spectrum_flatness(x, enc)))
        kl = kl_div(x, enc, n_samples=min(N, cfg["kl_samples"]), n_bins=cfg["kl_bins"])
        add(f"haar_D{D}", "KL expressibility (-> ~0)", 0.0, kl, "finite-sample bias; absolute tol")
    # entanglement: product vs GHZ (dual-rail qubits, n qubits = n photons, 2n modes)
    for n in cfg["entanglement_qubits"]:
        th = torch.rand(N, dtype=torch.float64) * (math.pi / 2)
        for kind, S in (("product", product_states(th[:, None].repeat(1, n))), ("ghz", ghz_states(th, n))):
            enc = TableEncoder(S, num_modes=2 * n, num_photons=n)
            # analytic average over bipartitions, in bits
            if kind == "product":
                exp_bits = 0.0
            else:
                p = torch.cos(th) ** 2
                h = -(p * torch.log2(p.clamp_min(1e-12)) + (1 - p) * torch.log2((1 - p).clamp_min(1e-12)))
                exp_bits = float(h.mean())
            try:
                obs = float(entanglement_entropy(ids(N), enc))
                add(f"{kind}_{n}q", "M3 entanglement entropy (bits, avg bipartition)", exp_bits, obs,
                    "ratio observed/expected = %.3f -> units/normalisation check" % (obs / exp_bits if exp_bits else float('nan')))
            except Exception as e:  # keep going, report the failure
                rows.append(dict(ensemble=f"{kind}_{n}q", metric="M3 entanglement entropy",
                                 expected=exp_bits, observed=float("nan"), ok=False, note=f"error: {e!r}"))
    return pd.DataFrame(rows)


# ── Random-label control ──────────────────────────────────────────────────────
def random_label(cfg, out):
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for ds in cfg["datasets"]:
        name, classes = ds["name"], ds.get("classes")
        label = ds.get("label", f"{name}-{len(classes) if classes else 'all'}c")
        xtr, xte, ytr, yte = C.load_dataset(name, ds.get("feature_reduction", cfg["feature_reduction"]),
                                            classes, cfg.get("max_per_split"), cfg["seed"])
        X = torch.cat([xtr, xte]); y = torch.cat([ytr, yte]); ntr = len(xtr)
        nc = int(len(torch.unique(y)))
        encs = C.build_encoders(X.shape[1], nc, cfg["encodings"], randomize_entangling=True, seed=cfg["seed"])
        wd_true = float(dataset_wasserstein(X.numpy(), y.numpy()))
        wd_null = [float(dataset_wasserstein(X.numpy(), rng.permutation(y.numpy()))) for _ in range(cfg["n_perm"])]
        rows.append(dict(dataset=label, encoding="(classical)", metric="WD", true=wd_true,
                         null_mean=np.mean(wd_null), null_sd=np.std(wd_null),
                         p_upper=float(np.mean(np.array(wd_null) >= wd_true))))
        for ename, enc in encs.items():
            S = C.encode_all(enc, X)
            m7_true = C.m7_class_overlap(S, y)
            Ktr, Kte = C.kernel_from_states(S[:ntr]), C.kernel_from_states(S[ntr:], S[:ntr])
            acc_true = float((SVC(kernel="precomputed", C=C_SVM).fit(Ktr, ytr.numpy()).predict(Kte) == yte.numpy()).mean())
            m7n, accn = [], []
            for _ in range(cfg["n_perm"]):
                yp = torch.as_tensor(rng.permutation(y.numpy()))
                m7n.append(C.m7_class_overlap(S, yp))
                accn.append(float((SVC(kernel="precomputed", C=C_SVM).fit(Ktr, yp[:ntr].numpy())
                                   .predict(Kte) == yp[ntr:].numpy()).mean()))
            rows.append(dict(dataset=label, encoding=ename, metric="M7", true=m7_true,
                             null_mean=np.mean(m7n), null_sd=np.std(m7n),
                             p_upper=float(np.mean(np.array(m7n) <= m7_true))))  # lower M7 = more separable
            rows.append(dict(dataset=label, encoding=ename, metric="kernel_svm_acc", true=acc_true,
                             null_mean=np.mean(accn), null_sd=np.std(accn),
                             p_upper=float(np.mean(np.array(accn) >= acc_true))))
            print(f"  {label} {ename}: M7 true={m7_true:.3f} null={np.mean(m7n):.3f}±{np.std(m7n):.3f}", flush=True)
    return pd.DataFrame(rows)


# ── Parity family ─────────────────────────────────────────────────────────────
def parity(cfg):
    rows = []
    for k in cfg["parity_orders"]:
        xtr, xte, ytr, yte = C.load_dataset("parity", cfg["feature_reduction"], None, None,
                                            cfg["seed"], parity_order=k,
                                            n_samples_synth=cfg["parity_samples"])
        X = torch.cat([xtr, xte]); y = torch.cat([ytr, yte]); ntr = len(xtr)
        base = {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
            "gboost": GradientBoostingClassifier(random_state=cfg["seed"]),
        }
        for bn, clf in base.items():
            rows.append(dict(order=k, model=bn, acc=float(clf.fit(xtr.numpy(), ytr.numpy()).score(xte.numpy(), yte.numpy()))))
        encs = C.build_encoders(X.shape[1], 2, cfg["encodings"], randomize_entangling=True, seed=cfg["seed"])
        wd = float(dataset_wasserstein(X.numpy(), y.numpy()))
        for ename, enc in encs.items():
            S = C.encode_all(enc, X)
            Ktr, Kte = C.kernel_from_states(S[:ntr]), C.kernel_from_states(S[ntr:], S[:ntr])
            acc = float((SVC(kernel="precomputed", C=cfg["parity_svm_C"]).fit(Ktr, ytr.numpy()).predict(Kte) == yte.numpy()).mean())
            rows.append(dict(order=k, model=f"kernel_svm[{ename}]", acc=acc, M7=C.m7_class_overlap(S, y), WD=wd))
            print(f"  parity k={k} {ename}: M7={rows[-1]['M7']:.3f} acc={acc:.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = C.load_config(ap.parse_args().config)
    out = C.out_dir(cfg, "controls")
    for part, fn in (("closed_form", lambda: closed_form(cfg)),
                     ("random_label", lambda: random_label(cfg, out)),
                     ("parity", lambda: parity(cfg))):
        if cfg["run"].get(part, False):
            print(f"=== {part}", flush=True)
            df = fn(); df.to_csv(out / f"{part}.csv", index=False)
            print(df.to_string(index=False))
    C.save_json(cfg, out / "config_used.json")


if __name__ == "__main__":
    main()
