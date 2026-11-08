"""Train YOLO11m with the paper's configuration (Table 3)."""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to data.yaml")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
a = ap.parse_args()

model = YOLO("yolo11m.pt")   # downloads pretrained weights on first run (needs internet)
model.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, optimizer="Adam",
            lr0=0.001, device=a.device, project="runs", name="yolo11m_fish", exist_ok=True)
print("Best weights: runs/yolo11m_fish/weights/best.pt  -> copy to weights/best.pt")
