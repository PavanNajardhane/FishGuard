"""Turn ANY unzipped YOLO-style dataset folder into the layout FishGuard needs.

usage:   python setup_dataset.py --src "C:/path/to/unzipped_folder"
result:  dataset/images/{train,val,test}, dataset/labels/{train,val,test}, dataset/data.yaml
It finds the images and .txt labels itself, keeps your existing train/val/test split if it
can see one, otherwise makes a random split, and reads class names from a yaml / names file.
"""
import argparse, random, shutil, sys, yaml
from pathlib import Path

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_WORDS = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}

def split_of(p, root):
    for part in reversed(p.relative_to(root).parts[:-1]):
        if part.lower() in SPLIT_WORDS:
            return SPLIT_WORDS[part.lower()]
    return None

def find_names(root):
    for y in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        try:
            d = yaml.safe_load(open(y, encoding="utf-8", errors="ignore"))
            n = d.get("names") if isinstance(d, dict) else None
            if isinstance(n, dict): return [n[k] for k in sorted(n, key=int)]
            if isinstance(n, list): return n
        except Exception:
            pass
    for f in list(root.rglob("classes.txt")) + list(root.rglob("*.names")) + list(root.rglob("labels.txt")):
        lines = [l.strip() for l in open(f, encoding="utf-8", errors="ignore") if l.strip()]
        if lines: return lines
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", default="dataset")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if not src.exists(): sys.exit(f"Folder not found: {src}")

    imgs = [f for f in src.rglob("*") if f.suffix.lower() in IMG]
    labels = {}
    for f in src.rglob("*.txt"):
        if f.stem.lower() in ("classes", "labels", "readme", "obj"): continue
        labels.setdefault(f.stem, f)
    pairs = [(i, labels[i.stem]) for i in imgs if i.stem in labels]
    print(f"Found {len(imgs)} images and {len(labels)} label files -> {len(pairs)} matched pairs")
    if not pairs:
        sys.exit("\nNo image+label pairs found. This dataset has no bounding-box labels (.txt), so a "
                 "detector cannot be trained on it. Send me a screenshot of the folder contents.")

    buckets = {"train": [], "val": [], "test": []}; unknown = []
    for i, l in pairs:
        s = split_of(i, src); (buckets[s] if s else unknown).append((i, l))
    if unknown:
        random.Random(a.seed).shuffle(unknown)
        if any(buckets.values()):
            buckets["train"] += unknown          # partly split already -> rest goes to train
        else:
            n = len(unknown); k = 700 if n >= 4000 else max(1, int(n * .1))
            buckets["val"], buckets["test"], buckets["train"] = unknown[:k], unknown[k:2*k], unknown[2*k:]
            print(f"No split folders found -> random split (val {k}, test {k}, train {n-2*k})")
    if not buckets["test"] and buckets["val"]:
        random.Random(a.seed).shuffle(buckets["val"]); h = len(buckets["val"]) // 2
        buckets["test"], buckets["val"] = buckets["val"][h:], buckets["val"][:h]
        print("No test folder found -> half of validation used as test")

    if out.exists(): shutil.rmtree(out)
    maxid = 0
    for s, items in buckets.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True); (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for i, l in items:
            shutil.copy(i, out / "images" / s / i.name); shutil.copy(l, out / "labels" / s / (i.stem + ".txt"))
            for line in open(l, encoding="utf-8", errors="ignore"):
                p = line.split()
                if p and p[0].isdigit(): maxid = max(maxid, int(p[0]))
        print(f"{s:5s}: {len(items)} images")

    names = find_names(src)
    if not names or len(names) <= maxid:
        print(f"Class names not found (labels use ids 0..{maxid}) -> using class0, class1, ... Edit data.yaml to rename.")
        names = [f"class{k}" for k in range(maxid + 1)]
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {k: n for k, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"\nDone. {len(names)} classes. Created {out}/data.yaml\nNext:  python train.py --data {out}/data.yaml --device cpu")

if __name__ == "__main__":
    main()
