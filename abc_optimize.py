"""Artificial Bee Colony search for the global calibration vector X* (paper Sec 3.6, eq. 2-5).

Fitness F(X) = 0.5*mAP@50 + 0.5*mAP@50-95 of the trained YOLO11m on the ENHANCED VALIDATION set.
(The paper does not define F exactly -> this is a documented assumption; edit `fitness()` to change.)
The TEST set is never touched here.
"""
import argparse, json, random, shutil, tempfile, yaml
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO
from enhance import BOUNDS, KEYS, vec_to_params, enhance_bgr

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
lo = np.array([BOUNDS[k][0] for k in KEYS]); hi = np.array([BOUNDS[k][1] for k in KEYS])


def build_enhanced_split(data_cfg, split, params, out_root, limit=None):
    """Write enhanced copy of `split` images (+ labels) and return a temp data.yaml path."""
    root = Path(data_cfg["path"])
    src_img = root / data_cfg[split]
    src_lab = root / str(data_cfg[split]).replace("images", "labels", 1)
    out_root = Path(out_root)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src_img.glob("*") if f.suffix.lower() in IMG)
    if limit: files = files[:limit]
    for f in files:
        im = cv2.imread(str(f))
        if im is None: continue
        cv2.imwrite(str(out_root / "images" / "val" / f.name), enhance_bgr(im, params))
        lab = src_lab / (f.stem + ".txt")
        if lab.exists(): shutil.copy(lab, out_root / "labels" / "val" / lab.name)
    cfg = {"path": str(out_root.resolve()), "train": "images/val", "val": "images/val",
           "names": data_cfg["names"]}
    p = out_root / "tmp.yaml"; p.write_text(yaml.safe_dump(cfg)); return str(p)


def fitness(x, model, data_cfg, limit, device):
    params = vec_to_params(x)
    with tempfile.TemporaryDirectory() as td:
        yml = build_enhanced_split(data_cfg, "val", params, td, limit)
        m = model.val(data=yml, imgsz=640, batch=16, device=device, verbose=False, plots=False)
    return 0.5 * float(m.box.map50) + 0.5 * float(m.box.map)


def abc(model, data_cfg, SN=6, iters=8, limit_trials=3, limit_imgs=None, device=None, seed=0):
    rng = np.random.default_rng(seed); D = len(KEYS)
    X = rng.uniform(lo, hi, (SN, D))
    X[0] = [0.0, 1.0, 1.0, 1.0, 1.0]   # 'no enhancement' start = baseline
    fit = np.array([fitness(x, model, data_cfg, limit_imgs, device) for x in X])
    trials = np.zeros(SN, int); history = []

    def try_neighbor(i):
        k = rng.choice([j for j in range(SN) if j != i]); j = rng.integers(D)
        v = X[i].copy(); phi = rng.uniform(-1, 1)
        v[j] = np.clip(X[i][j] + phi * (X[i][j] - X[k][j]), lo[j], hi[j])   # eq. (4)
        fv = fitness(v, model, data_cfg, limit_imgs, device)
        if fv > fit[i]: X[i], fit[i], trials[i] = v, fv, 0
        else: trials[i] += 1

    for it in range(1, iters + 1):
        for i in range(SN): try_neighbor(i)                                   # employed bees
        prob = (fit - fit.min() + 1e-9); prob /= prob.sum()                   # eq. (5)
        for _ in range(SN): try_neighbor(rng.choice(SN, p=prob))              # onlooker bees
        worst = int(np.argmax(trials))                                         # scout bee
        if trials[worst] > limit_trials:
            X[worst] = rng.uniform(lo, hi); fit[worst] = fitness(X[worst], model, data_cfg, limit_imgs, device); trials[worst] = 0
        history.append({"iter": it, "best": float(fit.max()), "mean": float(fit.mean())})
        print(f"iter {it}/{iters}  best={fit.max():.5f}  mean={fit.mean():.5f}", flush=True)
    b = int(np.argmax(fit)); return vec_to_params(X[b]), float(fit[b]), history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--colony", type=int, default=6); ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--limit-imgs", type=int, default=None, help="use only N val images per evaluation (faster)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.data)); cfg.setdefault("path", str(Path(a.data).parent))
    model = YOLO(a.weights)
    params, f, hist = abc(model, cfg, a.colony, a.iters, limit_imgs=a.limit_imgs, device=a.device)
    json.dump({"params": params, "fitness": f, "history": hist}, open("best_params.json", "w"), indent=2)
    print("Saved best_params.json:", params, "fitness", f)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        it = [h["iter"] for h in hist]
        plt.plot(it, [h["best"] for h in hist], "o-", label="Best Fitness")
        plt.plot(it, [h["mean"] for h in hist], "s-", label="Mean Fitness")
        plt.xlabel("ABC Iteration"); plt.ylabel("Fitness"); plt.legend(); plt.grid(alpha=.3)
        plt.title("Artificial Bee Colony Convergence"); plt.savefig("abc_convergence.png", dpi=150)
    except Exception as e: print("plot skipped:", e)
