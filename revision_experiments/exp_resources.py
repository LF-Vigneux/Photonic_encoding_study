"""Revisions 2.8: resource table per photonic encoding + data-access mode.

Per encoding: modes, photons, Hilbert dimension, beam splitters, phase shifters,
optical depth, post-selection success probability (unbunched outcomes), and the
survival probability under a per-component loss epsilon, adapted from v2's
fidelity penalty (1 - eps)^(D*W):

    P_survive(eps) = (1 - eps)^(n_photons * depth)

(each photon crosses ~`depth` components; all n photons must survive). The
overall success probability reported is P_postselect * P_survive(eps).

Counts come from the Perceval circuit of the layer when one exists
(angle, dense_angle, fourier). The amplitude encoders have no circuit in the
repository (the vector is written directly into the Fock amplitudes), so their
cost is reported as "oracle" together with the degrees-of-freedom argument:
an m-mode interferometer has m^2 real parameters, but an arbitrary state of
dimension D = C(m, n) needs 2(D - 1).

Data-access labels are read from the config (`data_access`); confirm them
against v2 Sec. 4.5 before using them in the paper.

Usage:  python revision_experiments/exp_resources.py --config resources.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from math import comb

import numpy as np
import pandas as pd
import torch

import common as C
import merlin as ml


def circuit_stats(circuit) -> dict:
    """Component counts and ASAP-scheduled depth of a Perceval circuit."""
    counts = Counter()
    level = {}
    for modes, comp in circuit:
        counts[type(comp).__name__] += 1
        lo, hi = min(modes), max(modes)
        span = range(lo, hi + 1)
        t = 1 + max(level.get(m, 0) for m in span)
        for m in span:
            level[m] = t
    return dict(counts=dict(counts), depth=max(level.values(), default=0))


def postselection_probability(name, n_features, X, seed, computation_space):
    """P(unbunched outcome) averaged over data, from the FOCK-space amplitudes
    of the same layer (same seed -> same random entangling layers)."""
    C.set_seed(seed)
    enc = C.build_encoders(n_features, 2, [name], computation_space=ml.ComputationSpace.FOCK,
                           randomize_entangling=True, seed=seed)[name]
    layer = enc.module.quantum_embedding_layer
    keys = list(layer.output_keys)
    unb = torch.tensor([max(k) <= 1 for k in keys])
    with torch.no_grad():
        params = enc.module.classical_encoder(X)
        from nn_embedding.utils.merlin_model_utils import assign_params
        amps = assign_params(layer, params)
    p = (amps.abs() ** 2)
    p_unb = (p[:, unb].sum(1) / p.sum(1)).numpy()
    return float(p_unb.mean()), float(p_unb.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = C.load_config(ap.parse_args().config)
    out = C.out_dir(cfg, "resources")
    d = cfg["n_features"]
    X = torch.rand(cfg["n_data_samples"], d)
    rows = []
    for name in cfg["encodings"]:
        enc = C.build_encoders(d, cfg["n_classes"], [name], randomize_entangling=True, seed=cfg["seed"])[name]
        row = dict(encoding=name, data_access=cfg["data_access"].get(name, "unlabelled"))
        mod = enc.module
        if enc.is_kernel_model:
            layer = mod.quantum_embedding_layer
            n_modes, n_ph = layer.circuit.m, layer.n_photons
            st = circuit_stats(layer.circuit)
            row.update(n_modes=n_modes, n_photons=n_ph, hilbert_dim=layer.output_size,
                       n_BS=st["counts"].get("BS", 0), n_PS=st["counts"].get("PS", 0),
                       n_other=sum(v for k, v in st["counts"].items() if k not in ("BS", "PS")),
                       depth=st["depth"], components=st["counts"])
            try:
                row["P_postselect"], row["P_postselect_sd"] = postselection_probability(
                    name, d, X, cfg["seed"], ml.ComputationSpace.UNBUNCHED)
            except Exception as e:
                row["P_postselect"], row["note"] = float("nan"), f"postselection failed: {e!r}"
        else:
            n_modes, n_ph = enc.meta["n_modes"], enc.meta["n_photons"]
            D = mod.output_size
            row.update(n_modes=n_modes, n_photons=n_ph, hilbert_dim=int(D),
                       n_BS=None, n_PS=None, depth=None, P_postselect=None,
                       dof_interferometer=n_modes**2, dof_state_needed=2 * (int(D) - 1),
                       note="no circuit in repo: amplitudes written directly (oracle / QRAM-like loading)")
        for eps in cfg["loss_per_component"]:
            if row.get("depth"):
                surv = (1 - eps) ** (row["n_photons"] * row["depth"])
                row[f"P_survive_eps{eps}"] = surv
                if row.get("P_postselect") == row.get("P_postselect"):
                    row[f"P_total_eps{eps}"] = surv * row["P_postselect"]
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ("components",) if c in df]).to_csv(out / "resource_table.csv", index=False)
    C.save_json(dict(rows=rows, config=cfg), out / "resource_table.json")
    with open(out / "resource_table.tex", "w") as f:
        keep = [c for c in df.columns if c not in ("components", "note", "P_postselect_sd")]
        f.write(df[keep].round(4).to_latex(index=False, na_rep="--"))


if __name__ == "__main__":
    main()
