import json
import random
from pathlib import Path

DATA_DIR = Path("fish4knowedge_dataset")
OUTPUT_FILE = Path("weights/class_split.json")
SEED = 42

random.seed(SEED)

split = {
    "train": [],
    "val": [],
    "test": []
}

for class_dir in sorted(DATA_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    random.shuffle(images)
    total = len(images)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    for image in images[:train_end]:
        split["train"].append(str(image))

    for image in images[train_end:val_end]:
        split["val"].append(str(image))

    for image in images[val_end:]:
        split["test"].append(str(image))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(split, f, indent=2)

print("Fixed dataset split created!")
print("Training images:", len(split["train"]))
print("Validation images:", len(split["val"]))
print("Testing images:", len(split["test"]))
print("Saved to:", OUTPUT_FILE)