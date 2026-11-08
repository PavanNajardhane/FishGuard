"""Compare YOLO11m baseline vs YOLO11m+ABC on the TEST set (paper Table 6)."""
import argparse, json, tempfile, yaml
from pathlib import Path
from ultralytics import YOLO
from abc_optimize import build_enhanced_split
from enhance import PAPER_PARAMS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--weights", default="weights/best.pt")
ap.add_argument("--params", default="best_params.json"); ap.add_argument("--device", default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.data)); cfg.setdefault("path", str(Path(a.data).parent))
params = json.load(open(a.params))["params"] if Path(a.params).exists() else PAPER_PARAMS
model = YOLO(a.weights)

def run(yml):
    m = model.val(data=yml, imgsz=640, batch=16, device=a.device, split="val", verbose=False, plots=False)
    p, r = float(m.box.mp), float(m.box.mr)
    return {"Precision": 100*p, "Recall": 100*r, "F1": 100*2*p*r/max(p+r, 1e-9),
            "mAP50": 100*float(m.box.map50), "mAP50-95": 100*float(m.box.map)}

with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
    base = run(build_enhanced_split(cfg, "test", {"brightness": 0, "contrast": 1, "gamma": 1, "saturation": 1, "sharpness": 1}, t1))
    enh = run(build_enhanced_split(cfg, "test", params, t2))
print(f"{'Metric':10s}{'Baseline':>10s}{'+ABC':>10s}{'Change':>10s}")
for k in base: print(f"{k:10s}{base[k]:10.2f}{enh[k]:10.2f}{enh[k]-base[k]:+10.2f}")
json.dump({"baseline": base, "abc": enh, "params": params}, open("test_results.json", "w"), indent=2)
