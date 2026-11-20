# train_seed.py : trains the paper's models (and baselines) for ONE seed on ONE GPU,
# selects epochs on the validation set only, and saves all predictions.
import os, sys, glob, copy, json, time, random, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42                      # same split for every run, so the test set never changes
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES, NUM_EPOCHS, BATCH = 4, 12, 16
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class BrainTumorDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths, self.labels, self.transform = list(file_paths), list(labels), transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, i):
        img = Image.open(self.file_paths[i]).convert("RGB")
        return (self.transform(img) if self.transform else img), self.labels[i]

class OptimizedResNet101(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1)); self.bn = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 128); self.relu = nn.ReLU(); self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64); self.drop2 = nn.Dropout(0.2); self.out = nn.Linear(64, NUM_CLASSES)
    def forward(self, x):
        x = torch.flatten(self.gap(self.features(x)), 1); x = self.bn(x)
        x = self.drop1(self.relu(self.fc1(x))); x = self.drop2(self.relu(self.fc2(x)))
        return self.out(x)

def make_swin(pretrained=True):
    m = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if pretrained else None)
    m.head = nn.Linear(m.head.in_features, NUM_CLASSES); return m

def make_convnext(pretrained=True):
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES); return m

def make_resnet50(pretrained=True):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES); return m

def make_effb0(pretrained=True):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES); return m

# same optimiser, schedule, epochs and selection rule for every model; only the backbone changes
BUILDERS = {"resnet": (lambda: OptimizedResNet101(), 1e-4), "swin": (lambda: make_swin(), 5e-5),
            "convnext": (lambda: make_convnext(), 1e-4), "resnet50": (lambda: make_resnet50(), 1e-4),
            "effb0": (lambda: make_effb0(), 1e-4)}

def find_data_dir():
    for d in ["/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri",
              "/kaggle/input/brain-tumor-classification-mri"]:
        if os.path.exists(d): return d
    raise FileNotFoundError("Attach the Sartaj 'Brain Tumor Classification (MRI)' dataset")

def get_splits(split_file=None):
    split_file = split_file or os.environ.get("SPLIT_FILE", "/kaggle/working/split_grouped.json")
    """Reads the duplicate-grouped split made once by the launcher (same for every seed)."""
    sp = json.load(open(split_file))
    return ((sp["train_paths"], sp["train_y"]), (sp["val_paths"], sp["val_y"]), (sp["test_paths"], sp["test_y"]))

def all_images():
    d = find_data_dir(); paths, labels = [], []
    for phase in ["Training", "Testing"]:
        for c, cls in enumerate(CLASSES):
            for f in sorted(glob.glob(os.path.join(d, phase, cls, "*.jpg"))):
                paths.append(f); labels.append(c)
    return paths, labels

def make_grouped_split(out_file, thresh=5, seed=SPLIT_SEED):
    """Groups images whose perceptual hashes differ by <= thresh bits, then splits by GROUP
    (stratified) so no duplicate or near-duplicate can sit on both sides of a split."""
    import imagehash
    from sklearn.model_selection import StratifiedGroupKFold
    paths, labels = all_images(); n = len(paths)
    h = np.array([int(str(imagehash.phash(Image.open(p).convert("L"))), 16) for p in paths], dtype=np.uint64)
    parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i0 in range(0, n, 256):
        blk = h[i0:i0 + 256]
        dist = np.unpackbits((blk[:, None] ^ h[None, :]).view(np.uint8).reshape(len(blk), n, 8), axis=2).sum(2)
        for a, b in zip(*np.where(dist <= thresh)):
            a += i0
            if b > a:
                ra, rb = find(a), find(b)
                if ra != rb: parent[rb] = ra
    groups = np.array([find(i) for i in range(n)]); y = np.array(labels)
    conflict = [g for g in np.unique(groups) if len(set(y[groups == g])) > 1]
    idx = np.arange(n)
    te_f = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y, groups))
    rest, test = te_f
    va_f = next(StratifiedGroupKFold(8, shuffle=True, random_state=seed).split(rest, y[rest], groups[rest]))
    train, val = rest[va_f[0]], rest[va_f[1]]
    # verify: no group appears in more than one split
    gs = [set(groups[train]), set(groups[val]), set(groups[test])]
    assert not (gs[0] & gs[1]) and not (gs[0] & gs[2]) and not (gs[1] & gs[2])
    sp = {k + "_paths": [paths[i] for i in v] for k, v in [("train", train), ("val", val), ("test", test)]}
    sp.update({k + "_y": [int(y[i]) for i in v] for k, v in [("train", train), ("val", val), ("test", test)]})
    multi = [g for g in np.unique(groups) if (groups == g).sum() > 1]
    sp["dup_groups"] = [[[paths[i], int(y[i])] for i in np.where(groups == g)[0]] for g in multi]
    sizes = np.bincount(groups)[np.unique(groups)]
    sp["info"] = {"images": n, "groups": int(len(sizes)), "images_in_duplicate_groups": int(sizes[sizes > 1].sum()),
                  "largest_group": int(sizes.max()), "groups_with_conflicting_labels": len(conflict),
                  "hash_threshold": thresh, "counts": {k: np.bincount(y[v], minlength=NUM_CLASSES).tolist()
                                                       for k, v in [("train", train), ("val", val), ("test", test)]}}
    json.dump(sp, open(out_file, "w"), indent=1)
    return sp["info"]

def make_patient_split(out_file, map_csv, thresh=5, seed=SPLIT_SEED):
    """Patient-aware split. Images are linked if their perceptual hashes differ by <= thresh bits OR they share a
    Figshare patient ID; linked images form one group, and whole groups are split (stratified 70/10/20).
    Images whose Sartaj label disagrees with the original Figshare label are left out and listed."""
    import imagehash, pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    tail = lambda p: "/".join(str(p).split("/")[-3:])
    mp = pd.read_csv(map_csv, dtype={"pid": str})
    mp = {tail(r.path): (None if pd.isna(r.pid) else r.pid, None if pd.isna(r.fig_label) else int(r.fig_label)) for r in mp.itertuples()}
    paths_all, labels_all = all_images()
    pid_all = [mp.get(tail(p), (None, None))[0] for p in paths_all]
    fl_all = [mp.get(tail(p), (None, None))[1] for p in paths_all]
    excluded = [{"path": p, "sartaj_label": int(y), "figshare_label": fl, "pid": pd_}
                for p, y, fl, pd_ in zip(paths_all, labels_all, fl_all, pid_all) if fl is not None and fl != y]
    keep = [i for i in range(len(paths_all)) if not (fl_all[i] is not None and fl_all[i] != labels_all[i])]
    paths = [paths_all[i] for i in keep]; y = np.array([labels_all[i] for i in keep]); pid = [pid_all[i] for i in keep]; n = len(paths)
    h = np.array([int(str(imagehash.phash(Image.open(p).convert("L"))), 16) for p in paths], dtype=np.uint64)
    parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i0 in range(0, n, 256):                                   # link near-duplicates
        blk = h[i0:i0 + 256]
        dist = np.unpackbits((blk[:, None] ^ h[None, :]).view(np.uint8).reshape(len(blk), n, 8), axis=2).sum(2)
        for a, b in zip(*np.where(dist <= thresh)):
            if b > a + i0: union(a + i0, b)
    first = {}                                                    # link images of the same patient
    for i, p_ in enumerate(pid):
        if p_ is None: continue
        if p_ in first: union(first[p_], i)
        else: first[p_] = i
    groups = np.array([find(i) for i in range(n)]); idx = np.arange(n)
    rest, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y, groups))
    va_f = next(StratifiedGroupKFold(8, shuffle=True, random_state=seed).split(rest, y[rest], groups[rest]))
    train, val = rest[va_f[0]], rest[va_f[1]]
    gs = [set(groups[train]), set(groups[val]), set(groups[test])]
    assert not (gs[0] & gs[1]) and not (gs[0] & gs[2]) and not (gs[1] & gs[2]), "a group crosses splits"
    ps = [{pid[i] for i in v if pid[i] is not None} for v in (train, val, test)]
    assert not (ps[0] & ps[2]) and not (ps[1] & ps[2]) and not (ps[0] & ps[1]), "a patient crosses splits"
    sp = {}
    for k, v in [("train", train), ("val", val), ("test", test)]:
        sp[k + "_paths"] = [paths[i] for i in v]; sp[k + "_y"] = [int(y[i]) for i in v]; sp[k + "_pid"] = [pid[i] for i in v]
    sizes = np.bincount(groups)[np.unique(groups)]
    sp["excluded_label_conflicts"] = excluded
    sp["info"] = {"images_total": len(paths_all), "excluded_label_conflicts": len(excluded), "images_used": n,
                  "groups": int(len(sizes)), "largest_group": int(sizes.max()), "hash_threshold": thresh,
                  "patients": {k: len(s_) for k, s_ in zip(["train", "val", "test"], ps)},
                  "patients_crossing_splits": 0,
                  "counts": {k: np.bincount(y[v], minlength=NUM_CLASSES).tolist() for k, v in [("train", train), ("val", val), ("test", test)]},
                  "tumor_images_with_patient_id": {k: int(sum(pid[i] is not None for i in v if y[i] != 2)) for k, v in [("train", train), ("val", val), ("test", test)]},
                  "tumor_images_without_patient_id": {k: int(sum(pid[i] is None for i in v if y[i] != 2)) for k, v in [("train", train), ("val", val), ("test", test)]}}
    json.dump(sp, open(out_file, "w"), indent=1)
    return sp["info"]

def set_seed(s):
    os.environ["PYTHONHASHSEED"] = str(s); random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def train(model, opt, sch, train_loader, val_loader, name, device, log):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_ep, best_wts = -1.0, 0, None
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(NUM_EPOCHS):
        model.train(); tl = tc = tn = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            out = model(x); loss = crit(out, y); loss.backward(); opt.step()
            tl += loss.item() * x.size(0); tc += (out.argmax(1) == y).sum().item(); tn += x.size(0)
        sch.step(); model.eval(); vl = vc = vn = 0
        with torch.no_grad():
            for x, y in val_loader:                       # VALIDATION set, never test
                x, y = x.to(device), y.to(device); out = model(x)
                vl += crit(out, y).item() * x.size(0); vc += (out.argmax(1) == y).sum().item(); vn += x.size(0)
        for k, v in zip(hist, [tl/tn, tc/tn, vl/vn, vc/vn]): hist[k].append(v)
        log(f"{name} epoch {ep+1}/{NUM_EPOCHS} train_acc {tc/tn:.4f} val_acc {vc/vn:.4f} val_loss {vl/vn:.4f}")
        if vc / vn > best_acc:
            best_acc, best_ep, best_wts = vc / vn, ep + 1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_wts)
    log(f"{name}: kept epoch {best_ep} (val_acc {best_acc:.4f})")
    return model, hist, best_ep, best_acc

@torch.no_grad()
def predict_views(model, loader, device):
    model.eval(); o, f = [], []
    for x, _ in loader:
        x = x.to(device)
        o.append(F.softmax(model(x), 1).cpu()); f.append(F.softmax(model(torch.flip(x, [3])), 1).cpu())
    return torch.cat(o).numpy(), torch.cat(f).numpy()

def run(seed, out_dir, model_names):
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, f"log_seed{seed}.txt"), "a")
    def log(m):
        line = f"[seed {seed} | {time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    todo = [n for n in model_names if not (os.path.exists(os.path.join(out_dir, f"probs_{n}_seed{seed}.npz"))
                                           and os.path.exists(os.path.join(out_dir, f"{n}_seed{seed}.pt")))]
    if not todo: log("all requested models already done, skipping"); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device} {torch.cuda.get_device_name(0) if device.type=='cuda' else ''}, torch {torch.__version__}, training {todo}")
    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = get_splits()
    log(f"split train {len(tr_p)} | val {len(va_p)} | test {len(te_p)}")
    def wi(w): np.random.seed(seed + w); random.seed(seed + w)
    set_seed(seed); g = torch.Generator().manual_seed(seed)          # shared generator: same order as the original run
    tl = DataLoader(BrainTumorDataset(tr_p, tr_y, train_transform), batch_size=BATCH, shuffle=True, num_workers=2, worker_init_fn=wi, generator=g)
    vl = DataLoader(BrainTumorDataset(va_p, va_y, test_transform), batch_size=BATCH, shuffle=False, num_workers=2)
    te = DataLoader(BrainTumorDataset(te_p, te_y, test_transform), batch_size=BATCH, shuffle=False, num_workers=2)
    meta_path = os.path.join(out_dir, f"meta_seed{seed}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name in [n for n in ["resnet", "swin", "convnext", "resnet50", "effb0"] if n in todo]:
        build, lr = BUILDERS[name]
        loader = tl
        if name not in ("resnet", "swin"):                             # own generator, independent of the paper's two branches
            loader = DataLoader(BrainTumorDataset(tr_p, tr_y, train_transform), batch_size=BATCH, shuffle=True, num_workers=2,
                                worker_init_fn=wi, generator=torch.Generator().manual_seed(seed))
        set_seed(seed); m = build().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); sch = CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
        t0 = time.time(); m, h, ep, acc = train(m, opt, sch, loader, vl, name, device, log)
        v_o, v_f = predict_views(m, vl, device); t_o, t_f = predict_views(m, te, device)
        np.savez(os.path.join(out_dir, f"probs_{name}_seed{seed}.npz"), val_y=np.array(va_y), test_y=np.array(te_y),
                 val_paths=np.array(va_p), test_paths=np.array(te_p), train_paths=np.array(tr_p),
                 val_orig=v_o, val_flip=v_f, test_orig=t_o, test_flip=t_f)
        torch.save(m.state_dict(), os.path.join(out_dir, f"{name}_seed{seed}.pt"))
        meta[name] = {"hist": h, "best_epoch": ep, "best_val_acc": acc, "train_minutes": (time.time() - t0) / 60}
        json.dump(meta, open(meta_path, "w"), indent=1)
        del m; torch.cuda.empty_cache()
    combine(seed, out_dir); log("DONE")

def combine(seed, out_dir):
    """write the combined ResNet+Swin file used by the analysis, if both branches are available"""
    f = os.path.join(out_dir, f"preds_seed{seed}.npz")
    pr, ps = [os.path.join(out_dir, f"probs_{n}_seed{seed}.npz") for n in ["resnet", "swin"]]
    if os.path.exists(f) or not (os.path.exists(pr) and os.path.exists(ps)): return
    a, b = np.load(pr, allow_pickle=True), np.load(ps, allow_pickle=True)
    np.savez(f, val_y=a["val_y"], test_y=a["test_y"], val_paths=a["val_paths"], test_paths=a["test_paths"], train_paths=a["train_paths"],
             **{f"resnet_{k}": a[k] for k in ["val_orig", "val_flip", "test_orig", "test_flip"]},
             **{f"swin_{k}": b[k] for k in ["val_orig", "val_flip", "test_orig", "test_flip"]})

if __name__ == "__main__":
    import traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", help="JSON list of {split, out, models, seed}")
    ap.add_argument("--seeds", type=int, nargs="+"); ap.add_argument("--models", nargs="+", default=["resnet", "swin"])
    ap.add_argument("--out", default="/kaggle/working/runs_grouped"); a = ap.parse_args()
    jobs = json.load(open(a.jobs)) if a.jobs else [{"split": None, "out": a.out, "models": a.models, "seed": s} for s in a.seeds]
    for j in jobs:
        if j.get("split"): os.environ["SPLIT_FILE"] = j["split"]
        try:
            run(j["seed"], j["out"], j["models"])
        except Exception as e:
            print(f"[seed {j['seed']} {j['models']} {j['out']}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(); torch.cuda.empty_cache()
