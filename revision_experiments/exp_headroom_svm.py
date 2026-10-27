"""Revisions 2.5 (preprocessing headroom) and 2.6 (downstream sanity check).

For every dataset:
  * classical baseline sweep on the same PCA features (LR, linear SVM, RBF SVM,
    kNN, random forest, gradient boosting);
  * headroom = acc(gradient boosting) - acc(logistic regression);
  * for each static photonic encoding: accuracy of an SVM on the precomputed
    fidelity kernel (same C as egas/photonic_kernel_svm.py) and M7.
Then Spearman(M7, kernel-SVM accuracy), pooled and per dataset, with a
bootstrap CI.

Usage:  python revision_experiments/exp_headroom_svm.py --config headroom_svm.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as C
from egas.photonic_kernel_svm import C_SVM


def baselines(seed):
    sc = StandardScaler
    return {
        "logreg": make_pipeline(sc(), LogisticRegression(max_iter=2000)),
        "svm_linear": make_pipeline(sc(), SVC(kernel="linear", C=C_SVM)),
        "svm_rbf": make_pipeline(sc(), SVC(kernel="rbf", C=C_SVM, gamma=0.125)),
        "knn5": make_pipeline(sc(), KNeighborsClassifier(5)),
        "rf": RandomForestClassifier(200, random_state=seed),
        "gboost": GradientBoostingClassifier(random_state=seed),
    }


def boot_ci(stat, n, B, rng, alpha=0.05):
    vals = [stat(rng.integers(0, n, n)) for _ in range(B)]
    vals = [v for v in vals if np.isfinite(v)]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = C.load_config(ap.parse_args().config)
    out = C.out_dir(cfg, "headroom_svm")
    rng = np.random.default_rng(cfg["seed"])
    rows_clf, rows_enc = [], []

    for ds in cfg["datasets"]:
        name, classes = ds["name"], ds.get("classes")
        label = ds.get("label", f"{name}-{len(classes) if classes else 'all'}c")
        print(f"=== {label}", flush=True)
        xtr, xte, ytr, yte = C.load_dataset(
            name, ds.get("feature_reduction", cfg["feature_reduction"]), classes,
            cfg.get("max_per_split"),
            cfg["seed"], ds.get("parity_order", 3),
        )
        Xtr, Xte, ytr_n, yte_n = xtr.numpy(), xte.numpy(), ytr.numpy(), yte.numpy()
        n_classes = int(len(np.unique(ytr_n)))
        chance = float(np.bincount(yte_n).max() / len(yte_n))  # majority-class rate

        for bname, clf in baselines(cfg["seed"]).items():
            acc = float(clf.fit(Xtr, ytr_n).score(Xte, yte_n))
            rows_clf.append(dict(dataset=label, model=bname, acc=acc, majority=chance,
                                 n_train=len(Xtr), n_test=len(Xte), n_classes=n_classes))

        # photonic kernels
        encs = C.build_encoders(
            Xtr.shape[1], n_classes, cfg["encodings"],
            randomize_entangling=cfg["randomize_entangling"], seed=cfg["seed"],
        )
        y_all = torch.cat([ytr, yte])
        for ename, enc in encs.items():
            Str, Ste = C.encode_all(enc, xtr), C.encode_all(enc, xte)
            Ktr, Kte = C.kernel_from_states(Str), C.kernel_from_states(Ste, Str)
            row = dict(dataset=label, encoding=ename, n_classes=n_classes, majority=chance,
                       M1=C.m1_support_dim(torch.cat([Str, Ste])),
                       M4=C.m4_kernel_flatness(Str),
                       M7=C.m7_class_overlap(torch.cat([Str, Ste]), y_all))
            for c in cfg["svm_C"]:
                svc = SVC(kernel="precomputed", C=c).fit(Ktr, ytr_n)
                row[f"acc_C{c}"] = float((svc.predict(Kte) == yte_n).mean())
            rows_enc.append(row)
            print(f"  {ename:16s} M7={row['M7']:.3f} acc={row[f'acc_C{cfg['svm_C'][0]}']:.3f}", flush=True)

    clf = pd.DataFrame(rows_clf)
    enc = pd.DataFrame(rows_enc)
    clf.to_csv(out / "classical_sweep.csv", index=False)
    enc.to_csv(out / "encoding_kernel_svm.csv", index=False)

    # headroom table
    piv = clf.pivot(index="dataset", columns="model", values="acc")
    head = pd.DataFrame({
        "logreg": piv["logreg"], "gboost": piv["gboost"],
        "headroom_gb_minus_lr": piv["gboost"] - piv["logreg"],
        "best_classical": piv.max(axis=1),
        "majority": clf.groupby("dataset")["majority"].first(),
    })
    head.to_csv(out / "headroom.csv")
    with open(out / "headroom.tex", "w") as f:
        f.write(head.round(3).to_latex())

    # Spearman(M7, accuracy): pooled + per dataset, with bootstrap CI
    acc_col = f"acc_C{cfg['svm_C'][0]}"
    res = {}
    def rho(idx, df=enc):
        d = df.iloc[idx]
        return spearmanr(d["M7"], d[acc_col])[0]
    r, p = spearmanr(enc["M7"], enc[acc_col])
    res["pooled"] = dict(rho=float(r), p=float(p), n=len(enc),
                         ci95=boot_ci(rho, len(enc), cfg["n_boot"], rng))
    # pooled after removing the per-dataset mean accuracy (dataset difficulty)
    e2 = enc.copy(); e2["acc_c"] = e2[acc_col] - e2.groupby("dataset")[acc_col].transform("mean")
    e2["M7_c"] = e2["M7"] - e2.groupby("dataset")["M7"].transform("mean")
    res["pooled_within_dataset_centered"] = dict(rho=float(spearmanr(e2["M7_c"], e2["acc_c"])[0]))
    res["per_dataset"] = {
        d: dict(rho=float(spearmanr(g["M7"], g[acc_col])[0]), n=len(g))
        for d, g in enc.groupby("dataset") if len(g) > 2
    }
    res["per_encoding_across_datasets"] = {
        e: dict(rho=float(spearmanr(g["M7"], g[acc_col])[0]), n=len(g))
        for e, g in enc.groupby("encoding") if len(g) > 2
    }
    C.save_json(dict(spearman_M7_vs_accuracy=res, config=cfg), out / "spearman.json")
    print(C.json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
