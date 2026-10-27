"""Revisions 2.9: uncertainty and held-out evaluation.

For every dataset and encoding:
  * interferometer repeats: the encoder is rebuilt `n_seeds` times with a
    different random entangling layer (randomize_entangling=True) -> mean, sd;
  * subsample intervals: `n_boot` random subsamples (without replacement,
    fraction `subsample_frac`) of the already-encoded states -> 95% interval of
    M1, M4, M7 (state-only versions of the repo metrics; bootstrap with
    replacement is avoided because duplicated states make the kernel
    rank-deficient and bias M1/M4);
  * kernel-SVM test accuracy and the best classical baseline on the same features.

The held-out run is the same script with a different config: the config file is
hashed (sha256) and timestamped into the outputs, so the thresholds written in
`prereg` are frozen *before* the held-out data are looked at. Commit the config
before running.

Usage:  python revision_experiments/exp_statistics.py --config statistics_main.json
        python revision_experiments/exp_statistics.py --config statistics_heldout.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as C
from dataset_complexity.utils import dataset_wasserstein
from egas.photonic_kernel_svm import C_SVM


def ci(v):
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = C.load_config(ap.parse_args().config)
    out = C.out_dir(cfg, cfg.get("name", "statistics"))
    rng = np.random.default_rng(cfg["seed"])
    rows = []

    for ds in cfg["datasets"]:
        name, classes = ds["name"], ds.get("classes")
        label = ds.get("label", f"{name}-{len(classes) if classes else 'all'}c")
        # Checkpoint per dataset: a rerun after a crash (e.g. OOM) skips finished datasets.
        ckpt = out / f"ckpt_{re.sub(r'[^A-Za-z0-9_.-]', '_', label)}.json"
        if ckpt.exists():
            saved = json.loads(ckpt.read_text())
            if saved["cfg"] == json.loads(json.dumps(cfg)):
                print(f"=== {label}: resumed from checkpoint", flush=True)
                rows.extend(saved["rows"])
                continue
        print(f"=== {label}", flush=True)
        start = len(rows)
        xtr, xte, ytr, yte = C.load_dataset(
            name, ds.get("feature_reduction", cfg["feature_reduction"]), classes,
            cfg.get("max_per_split"), cfg["seed"], ds.get("parity_order", 3))
        X = torch.cat([xtr, xte]); y = torch.cat([ytr, yte]); ntr = len(xtr)
        nc = int(len(torch.unique(y)))
        wd = float(dataset_wasserstein(X.numpy(), y.numpy()))
        clf_acc = {}
        for bn, clf in {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "gboost": GradientBoostingClassifier(random_state=cfg["seed"]),
        }.items():
            clf_acc[bn] = float(clf.fit(xtr.numpy(), ytr.numpy()).score(xte.numpy(), yte.numpy()))

        per_enc = {e: [] for e in cfg["encodings"]}
        for s in range(cfg["n_seeds"]):
            encs = C.build_encoders(X.shape[1], nc, cfg["encodings"],
                                    randomize_entangling=True, seed=cfg["seed"] + s)
            for ename, enc in encs.items():
                S = C.encode_all(enc, X)
                K = C.kernel_from_states(S[:ntr]); Kt = C.kernel_from_states(S[ntr:], S[:ntr])
                acc = float((SVC(kernel="precomputed", C=C_SVM).fit(K, ytr.numpy()).predict(Kt) == yte.numpy()).mean())
                sub = {"M1": [], "M4": [], "M7": []}
                m = int(cfg["subsample_frac"] * len(S))
                for _ in range(cfg["n_boot"]):
                    idx = torch.as_tensor(np.sort(rng.choice(len(S), m, replace=False)))
                    if len(torch.unique(y[idx])) < nc:
                        continue
                    sub["M1"].append(C.m1_support_dim(S[idx]))
                    sub["M4"].append(C.m4_kernel_flatness(S[idx]))
                    sub["M7"].append(C.m7_class_overlap(S[idx], y[idx]))
                per_enc[ename].append(dict(
                    M1=C.m1_support_dim(S), M4=C.m4_kernel_flatness(S), M7=C.m7_class_overlap(S, y),
                    acc=acc, **{f"sub_{k}": v for k, v in sub.items()}))
                print(f"  seed {s} {ename}: M7={per_enc[ename][-1]['M7']:.3f} acc={acc:.3f}", flush=True)

        for ename, reps in per_enc.items():
            row = dict(dataset=label, encoding=ename, n_classes=nc, WD=wd,
                       acc_logreg=clf_acc["logreg"], acc_gboost=clf_acc["gboost"],
                       headroom=clf_acc["gboost"] - clf_acc["logreg"], n_seeds=len(reps))
            for k in ("M1", "M4", "M7", "acc"):
                v = np.array([r[k] for r in reps])
                row[f"{k}_mean"], row[f"{k}_sd_repeats"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            for k in ("M1", "M4", "M7"):
                pooled = np.concatenate([r[f"sub_{k}"] for r in reps])
                row[f"{k}_sub_ci95"] = ci(pooled) if len(pooled) else None
            rows.append(row)
        ckpt.write_text(json.dumps(dict(cfg=cfg, rows=rows[start:])))

    df = pd.DataFrame(rows)
    df.to_csv(out / "statistics.csv", index=False)
    C.save_json(dict(config=cfg, rows=rows), out / "statistics.json")
    print(df[["dataset", "encoding", "n_classes", "WD", "M7_mean", "M7_sd_repeats", "acc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
