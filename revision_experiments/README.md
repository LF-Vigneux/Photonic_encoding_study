# Revision experiments

Additional experiments for the photonic encoding study, designed around the
questions raised by the updated paper, Pere, *Does Data Complexity Predict
Quantum Advantage? A framework, a pre-specified test, and an attribution of the
measured edge* (arXiv:2509.16410v2). Scripts only add files under
`revision_experiments/`; nothing in the existing pipeline is modified. Outputs go
to `results/revisions/<experiment>/`.

## Why these experiments

v2 of the paper tests whether data complexity predicts a quantum edge and
reports that, in its tested regime, it does not (pre-specified H5 fails:
leave-one-dataset-out R² = -0.32 against a 0.4 threshold). It attributes the
apparent signal to the classical baseline and to the classical preprocessing, and
it flags that the quantum model sits at chance in most cells. It draws four
lessons: the baseline determines the measured advantage; reducing data to a few
qubits removes nonlinear headroom; metrics and scaling laws must be audited
against known cases; and a null is only informative if quantum variation exists
to be detected. It also asks for explicit data-access modes (S / Q / N) with a
resource and loss model, and a frozen pre-specification before held-out data is
touched.

This repository measures the *induced* (state-based) side of that framework on
photonic encodings: M1 Hilbert-space support dimension, M4 kernel-spectrum
flatness, M7 class overlap, plus the Wasserstein distance. The experiments below
re-ask the paper's questions at the scale of this study (8 PCA features, up to
400 samples per split, five static encodings), so the metrics are not read in
isolation:

| Paper question | Experiment | What we check |
|---|---|---|
| Does the reduction leave nonlinear headroom for the model? (§7.3) | **Headroom** (`exp_headroom_svm.py`) | Gradient boosting minus logistic regression on the same reduced features. If it is about zero, there is nothing for an encoding to exploit, whatever its metrics say. |
| Does the classical baseline determine the apparent edge? (§7.2) | **Classical sweep** (`exp_headroom_svm.py`) | Six classical models per dataset, so any claim is made against the best one and not against a linear baseline only. |
| Does a complexity measure predict performance? (H5, §7.1) | **Spearman M7 vs kernel-SVM accuracy** (`exp_headroom_svm.py`) | Small-scale analogue of the central test: does the induced overlap rank the encodings by accuracy, pooled and within dataset? |
| Are the estimators correct, and is the null readable? (§4.1, App. B, §7.3 controls, §7.5, §7.12) | **Controls** (`exp_controls.py`) | Closed-form ensembles with known answers (estimator audit); a random-label permutation test (does the metric carry class information, does accuracy beat chance); a parity family with a complexity knob, as in the paper's parity probes. |
| What does the encoding cost, and under which data-access assumption? (§4.5, Modes S/Q/N) | **Resources** (`exp_resources.py`) | Beam-splitter / phase-shifter counts, depth, post-selection probability and a per-component loss model; amplitude encodings flagged as needing coherent loading. |
| Are the results stable, and is the held-out protocol frozen? (§6.5, §6.7) | **Statistics** (`exp_statistics.py`) | Repeats over seeds and subsample intervals; held-out datasets with a committed pre-registration. |

## Results

Run on 10 dataset/class configurations (breast_cancer, wine, moons, manifold,
mnist 2 and 4 classes, fashion, kmnist, cifar10, eurosat). All numbers are read
from `results/revisions/`. They describe this small regime only; like the paper's
null, they do not extend to larger widths or other algorithm families.

**Headroom.** Headroom (gradient boosting minus logistic regression) lies
between -1.5 and +1.8 points on nine of ten datasets. Moons is the exception
(+10 points). Most tasks are close to saturated by a linear model (wine 100 %,
mnist 0/1 99.8 %); cifar10 is the hardest (about 78 %). This matches the paper's
observation that few-qubit reduction leaves little nonlinear structure, and it
bounds what any encoding can gain here. As the paper notes for parity, near-zero
headroom can also mean the structure exists but gradient boosting fails to find it.

**M7 vs accuracy.** Pooled over 50 dataset/encoding pairs, Spearman rho(M7,
kernel-SVM accuracy) = -0.51 (p = 2e-4, 95 % CI [-0.71, -0.23]); -0.56 after
centring within dataset. Negative means lower overlap goes with higher accuracy.
It is negative on 8 of the 9 datasets where it is defined (kmnist: +0.30; wine is
undefined because every encoding sits at majority-class accuracy). Per encoding
across datasets rho ranges from -0.35 (angle) to -0.52 (fourier). Each
per-dataset value rests on five points. Unlike the paper's H5, this is a
correlation with kernel-SVM accuracy, not a budget-matched edge over the best
classical model, so it does not contradict the paper's null.

**Controls.**
* *Closed form:* M1, M4, M7 are exact on identical and orthogonal ensembles, M1
  reaches D on Haar states, and the entanglement entropy matches on GHZ and
  product states. Two checks fail: M4 on Haar states is far below its theoretical
  value (30.4 vs 111.0 at D=16; 105.7 vs 465.8 at D=64) and the KL expressibility
  estimate is biased at finite sample size (0.57 and 1.95 vs about 0). These are
  estimator limitations to resolve or state, not results.
* *Random labels:* true-label M7 is below the permuted-label null on every
  dataset (p = 0.00), so M7 carries class information. Kernel-SVM accuracy beats
  its permuted-label null only on mnist; on breast_cancer and wine it stays at
  the majority level (p between 0.52 and 0.76). This is the same regime the paper
  describes, where the quantum model is mostly at chance.
* *Parity:* all kernel SVMs fall to chance from order 3 (0.46 to 0.57), while
  the classical RBF SVM keeps signal at orders 2 and 3 and gradient boosting at
  orders 2 and 3 (0.995, 0.625). M7 rises with order (about 0.7 at order 1 to
  0.97 at order 6), as expected for harder tasks. The encodings beat chance only
  at orders 1 and 2.

**Resources.** Dense angle (depth 2) and fourier (depth 1) keep over 92 %
survival at 1 % loss per component. Angle needs 648 beam splitters at depth 413,
its post-selection probability is 0.25, and survival is essentially zero at 0.5 %
loss. Amplitude encodings have no circuit in the repo; they are tagged as
needing coherent (QRAM-like) loading, so they would be Mode Q in the paper's
taxonomy and would need a quantum-inspired classical comparator under matched
access.

**Statistics.** Fixed-circuit encodings
(dense_angle, fourier, amplitude, dense_amplitude) show zero spread over seeds
for M1, M4, M7. Only the randomly entangled angle encoding varies (M7 SD 0.02 to
0.06, M1 SD about 2), so metric rankings are not seed artefacts. Angle's
kernel-SVM accuracy is much less stable (SD up to 0.17 on fashion).

## Conclusion

Taken together, the experiments support a cautious reading consistent with the
paper's null, at the scale of this study (8 features, up to 400 samples, five
static encodings, no NQE/EGAS):

1. **The induced metrics are sound where they can be checked, and carry real
   class information.** M1, M4 and M7 are exact on closed-form ensembles
   (M4 on Haar states and the expressibility estimate are exceptions), and M7
   separates true labels from permuted ones on every dataset. They are
   meaningful descriptors of an encoding.
2. **They track kernel-SVM accuracy, but only as a ranking of encodings within a
   dataset.** The pooled rho = -0.51 is moderate, rests on 5 points per dataset,
   and is driven by tasks where the kernel SVM is above chance at all.
3. **That is not evidence of a quantum edge.** Most tasks leave almost no
   headroom beyond a linear model, and the kernel SVMs sit at or near chance on
   several datasets (wine, breast_cancer, cifar10, and parity beyond order 2).
   Where accuracy is high, a classical model matches it. So the metrics describe
   how an encoding behaves, not whether it beats the best classical baseline.
4. **Hardware cost separates the encodings more sharply than the metrics do.**
   Dense angle and fourier are cheap and loss-tolerant; angle is expensive and
   fragile; amplitude encodings assume coherent loading and need a matched-access
   classical comparator.
5. **Metric rankings are stable.** Only the randomly entangled angle encoding
   varies across seeds.

What this does not show: that complexity can never predict an edge. The regime
has little quantum variation to rank (see the paper, §7.5), the held-out
protocol has not been run, and 3 of 10 configurations are missing from the
statistics run. The natural next steps are the missing runs, the frozen held-out
test, including NQE and EGAS, and a regime with a designed quantum-sensitive
structure (the paper's signature control).

## What each script does

* **Encodings covered:** angle, dense_angle, fourier, amplitude, dense_amplitude
  (static, built as in `run_dataset_complexity.py`). NQE and EGAS are not
  included: they are trained per dataset; add them via `common.build_encoders`
  if you want them in the SVM check.
* **Metrics:** M1, M4, M7 are recomputed from the states (`common.py`) with the
  repo's current definitions. M7 is cross-checked against the repo function in
  `exp_controls.py` (closed-form block). M2, M3, M5, M6 are only used in the
  closed-form block (M3) since they need circuit information or are expensive.
  If you change the normalisation or definitions (§1.x), the state-only copies in
  `common.py` must be updated too.
* **Held-out / pre-registration:** the held-out script produces the per-dataset
  tables and stores a hash of the config; it does not evaluate a decision rule.
  Following the paper's pre-specification practice (§6.7), fill `prereg` in
  `statistics_heldout.json` and commit it before the run.
* **Subsample intervals** use subsampling without replacement (80%), not the
  bootstrap with replacement, because duplicated states make the kernel
  rank-deficient. They understate sampling variance; say "subsample interval".
* **Resource table:** BS/PS counts and depth come from the Perceval circuits.
  The amplitude encoders have no circuit in the repo, so they get the
  degrees-of-freedom argument instead of counts. The loss model is an
  assumption: `(1−ε)^(n_photons · depth)`, ε per component.
* **Mode labels** in `resources.json` use "native circuit" and "Mode Q". The
  paper (v2 §4.5) defines Mode S = sparse/structured, Mode Q = QRAM-dependent,
  Mode N = quantum-native; classical feature vectors are Mode S in its suite, so
  "native circuit" should be reconciled with that taxonomy.
