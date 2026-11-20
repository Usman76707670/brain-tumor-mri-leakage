"""Reproduce the statistical analyses of the paper from the released per-image predictions.

Usage (from the repository root):
    python code/compute_statistics.py

Inputs:  predictions/test_predictions_all_seeds.csv.gz, splits/split_patient.json
Outputs: results/statistics/*.csv

Methods
- Primary model: equal-weight average of the ResNet101 and Swin-V2-B softmax outputs, original view only (no TTA).
- Unit of resampling: a patient cluster. Test images with a Figshare patient ID are grouped by that ID; each test image
  without an ID is its own cluster. This cannot remove dependence between unidentified images of the same patient.
- Per-seed paired difference: correctness of model A minus model B on the same images; 95% CI by bootstrapping
  patient clusters (5,000 resamples).
- Pooled difference: per-image correctness is first averaged over the four seeds, so each image enters once; clusters
  are then bootstrapped. A second interval also resamples the four seeds with replacement, so that seed-to-seed
  variation is included.
- Tumor-only control: the decision is restricted to the three tumor classes (the no-tumor score is ignored), which
  removes the influence of the no-tumor image source.
"""
import json, os, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "statistics"); os.makedirs(OUT, exist_ok=True)
NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]; NT = 2; SEEDS = [42, 43, 44, 45]; B = 5000
MODELS = {"resnet": "ResNet101", "swin": "Swin-V2-B", "effb0": "EfficientNet-B0", "convnext": "ConvNeXt-T", "resnet50": "ResNet50"}

sp = json.load(open(os.path.join(ROOT, "splits", "split_patient.json")))
tail = lambda p: "/".join(str(p).split("/")[-3:])
files = [tail(p) for p in sp["test_paths"]]; y = np.array(sp["test_y"]); pid = sp["test_pid"]; N = len(y)
pr = pd.read_csv(os.path.join(ROOT, "predictions", "test_predictions_all_seeds.csv.gz"))
pr = pr[pr.view == "orig"]
cols = [f"p_{n}" for n in NAMES]
P = {s: {} for s in SEEDS}
for (m, s), g in pr.groupby(["model", "seed"]):
    g = g.set_index("file").loc[files]
    assert (g["label"].values == y).all(), "labels do not match the split file"
    P[int(s)][MODELS[m]] = g[cols].values
for s in SEEDS: P[s]["Fusion"] = (P[s]["ResNet101"] + P[s]["Swin-V2-B"]) / 2

clusters = {}; cid = np.array([clusters.setdefault(p if p else f"image{i}", len(clusters)) for i, p in enumerate(pid)])
def boot(stat, mask=None, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(N) if mask is None else np.where(mask)[0]
    c = cid[idx]; U = np.unique(c); mem = [idx[c == u] for u in U]
    v = [stat(np.concatenate([mem[j] for j in rng.integers(0, len(U), len(U))])) for _ in range(B)]
    return np.percentile(v, [2.5, 97.5])
def correct(m, s, three_class=False):
    p = P[s][m].copy()
    if three_class: p[:, NT] = -1
    return (p.argmax(1) == y).astype(float)

# 1. accuracy per model with patient-cluster intervals
rows = []
for m in ["Fusion", "Swin-V2-B", "ResNet101", "EfficientNet-B0", "ConvNeXt-T", "ResNet50"]:
    acc = [100 * correct(m, s).mean() for s in SEEDS]; c = np.mean([correct(m, s) for s in SEEDS], 0); lo, hi = boot(lambda i: 100 * c[i].mean())
    rows.append({"Model": m, "Accuracy % mean": round(np.mean(acc), 2), "SD over seeds": round(np.std(acc, ddof=1), 2),
                 "Per seed": [round(float(a), 2) for a in acc], "95% CI (patient clusters)": f"{lo:.2f} to {hi:.2f}"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "accuracy_patient_cluster_ci.csv"), index=False)

# 2. paired differences: per seed, pooled, and pooled with seeds also resampled
U = np.unique(cid); mem = [np.where(cid == u)[0] for u in U]; rows = []
for other in ["Swin-V2-B", "ResNet101", "EfficientNet-B0", "ConvNeXt-T", "ResNet50"]:
    D = np.array([correct("Fusion", s) - correct(other, s) for s in SEEDS])
    for k, s in enumerate(SEEDS):
        lo, hi = boot(lambda i: 100 * D[k][i].mean())
        rows.append({"Comparison": f"Fusion minus {other}", "Seed": s, "Difference (points)": round(100 * D[k].mean(), 2), "95% CI (patient clusters)": f"{lo:.2f} to {hi:.2f}"})
    d = D.mean(0); lo, hi = boot(lambda i: 100 * d[i].mean())
    rng = np.random.default_rng(1); v = []
    for _ in range(B):
        ks = rng.integers(0, len(SEEDS), len(SEEDS)); ii = np.concatenate([mem[j] for j in rng.integers(0, len(U), len(U))]); v.append(100 * D[ks][:, ii].mean())
    l2, h2 = np.percentile(v, [2.5, 97.5])
    rows.append({"Comparison": f"Fusion minus {other}", "Seed": "pooled", "Difference (points)": round(100 * d.mean(), 2),
                 "95% CI (patient clusters)": f"{lo:.2f} to {hi:.2f}", "95% CI (clusters and seeds resampled)": f"{l2:.2f} to {h2:.2f}"})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "paired_differences.csv"), index=False)

# 3. source-matched controls
tum, has = y != NT, np.array([p is not None for p in pid]); rows = []
for label, mask, three in [("All four classes", np.ones(N, bool), False), ("Tumor images only, three-class decision", tum, True),
                           ("Tumor images with a verified patient ID (single source), three-class", tum & has, True),
                           ("Tumor images without a patient ID, three-class", tum & ~has, True)]:
    r = {"Subset": label, "Images": int(mask.sum())}
    for m in ["Fusion", "Swin-V2-B", "ResNet101", "EfficientNet-B0"]:
        c = np.mean([correct(m, s, three) for s in SEEDS], 0); lo, hi = boot(lambda i: 100 * c[i].mean(), mask)
        r[m] = f"{100 * c[mask].mean():.2f} ({lo:.1f} to {hi:.1f})"
    d = np.mean([correct("Fusion", s, three) - correct("Swin-V2-B", s, three) for s in SEEDS], 0); lo, hi = boot(lambda i: 100 * d[i].mean(), mask)
    r["Fusion minus Swin-V2-B"] = f"{100 * d[mask].mean():+.2f} ({lo:.2f} to {hi:.2f})"; rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "source_matched_controls.csv"), index=False)

# 4. sensitivity: remove Testing/glioma_tumor/image(48).jpg, a glioma label on an image that matches six meningioma images
keep = np.array([f != "Testing/glioma_tumor/image(48).jpg" for f in files]); rows = []
for m in ["Fusion", "Swin-V2-B", "ResNet101", "EfficientNet-B0"]:
    rows.append({"Model": m, "All test images %": round(np.mean([100 * correct(m, s).mean() for s in SEEDS]), 2),
                 "Without image(48) %": round(np.mean([100 * correct(m, s)[keep].mean() for s in SEEDS]), 2)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_image48.csv"), index=False)
for f in sorted(os.listdir(OUT)): print(f"\n== {f}\n" + pd.read_csv(os.path.join(OUT, f)).to_string(index=False))
