"""Check / (optionally) re-split a YOLO-format dataset into train/val/test.

Expected YOLO layout (what the paper used: 13 classes, 6842/700/700):
  DATA/images/{train,val,test}/*.jpg   DATA/labels/{train,val,test}/*.txt   (+ data.yaml)
If you only have ONE pool of images+labels (DATA/images/*.jpg, DATA/labels/*.txt) use --resplit.

usage:
  python prepare_dataset.py --data /path/to/dataset --check
  python prepare_dataset.py --data /path/to/pool --resplit --out dataset --val 700 --test 700 --names AngelFish BlueTang ...
"""
import argparse, random, shutil, yaml
from pathlib import Path

IMG = {".jpg", ".jpeg", ".png", ".bmp"}

def check(data):
    data = Path(data)
    for s in ("train", "val", "test"):
        d = data / "images" / s
        n = len([f for f in d.glob("*") if f.suffix.lower() in IMG]) if d.exists() else 0
        l = len(list((data / "labels" / s).glob("*.txt"))) if (data / "labels" / s).exists() else 0
        print(f"{s:5s}: {n} images, {l} label files")

def resplit(pool, out, nval, ntest, names, seed=42):
    pool, out = Path(pool), Path(out)
    imgs = [f for f in (pool / "images").rglob("*") if f.suffix.lower() in IMG]
    random.Random(seed).shuffle(imgs)
    parts = {"val": imgs[:nval], "test": imgs[nval:nval + ntest], "train": imgs[nval + ntest:]}
    for s, files in parts.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for f in files:
            lab = pool / "labels" / (f.stem + ".txt")
            if not lab.exists():
                continue
            shutil.copy(f, out / "images" / s / f.name)
            shutil.copy(lab, out / "labels" / s / lab.name)
    cfg = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": {i: n for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    check(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--check", action="store_true")
    ap.add_argument("--resplit", action="store_true"); ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=int, default=700); ap.add_argument("--test", type=int, default=700)
    ap.add_argument("--names", nargs="+", default=[])
    a = ap.parse_args()
    if a.resplit: resplit(a.data, a.out, a.val, a.test, a.names)
    else: check(a.data)
