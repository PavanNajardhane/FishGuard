
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = "fish4knowedge_dataset"
MODEL_PATH = "weights/fish_classifier.pth"
OUTPUT_DIR = "static"

IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", device)

# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

classes = dataset.classes
num_classes = len(classes)

print("Total images:", len(dataset))
print("Total classes:", num_classes)

# ==========================================
# CREATE TEST SPLIT
# ==========================================

targets = np.array(dataset.targets)
test_indices = []

for class_id in range(num_classes):

    indices = np.where(targets == class_id)[0].tolist()

    random.Random(SEED + class_id).shuffle(indices)

    if len(indices) >= 2:

        test_count = max(
            1,
            int(len(indices) * 0.20)
        )

        test_indices.extend(indices[:test_count])

if not test_indices:
    raise RuntimeError("Unable to create test dataset.")

test_data = Subset(dataset, test_indices)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_data))

# ==========================================
# LOAD TRAINED RESNET18 MODEL
# ==========================================

print("\nLoading trained ResNet18 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

saved_classes = checkpoint.get("classes")

if saved_classes is not None:
    if list(saved_classes) != classes:
        raise RuntimeError(
            "Dataset classes do not match saved model classes."
        )

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint
)

model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("Model loaded successfully!")

# ==========================================
# RUN PREDICTIONS
# ==========================================

print("\nEvaluating test images...")

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        print(
            f"Processed batch {batch_index + 1}/{len(test_loader)}"
        )

# ==========================================
# CALCULATE METRICS
# ==========================================

print("\nCalculating metrics...")

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

metrics = {
    "accuracy": round(float(accuracy) * 100, 2),

    "test_images": len(all_labels),

    "number_of_classes": num_classes,

    "precision_macro": round(
        report["macro avg"]["precision"] * 100, 2
    ),

    "recall_macro": round(
        report["macro avg"]["recall"] * 100, 2
    ),

    "f1_macro": round(
        report["macro avg"]["f1-score"] * 100, 2
    ),

    "classification_report": report
}

# ==========================================
# CREATE CONFUSION MATRIX
# ==========================================

print("\nGenerating confusion matrix...")

fig, ax = plt.subplots(figsize=(18, 15))

image = ax.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

fig.colorbar(image, ax=ax)

ax.set_title(
    "FishGuard - ResNet18 Confusion Matrix",
    fontsize=16
)

ax.set_xlabel("Predicted Species")
ax.set_ylabel("Actual Species")

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))

ax.set_xticklabels(
    classes,
    rotation=90,
    fontsize=8
)

ax.set_yticklabels(
    classes,
    fontsize=8
)

threshold = cm.max() / 2 if cm.size else 0

for i in range(num_classes):

    for j in range(num_classes):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=6,
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

fig.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

fig.savefig(
    cm_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

# ==========================================
# SAVE CLASSIFICATION METRICS
# ==========================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "classification_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(num_classes)),
            target_names=classes,
            zero_division=0
        )
    )

# ==========================================
# FINAL RESULTS
# ==========================================

print("\n===================================")
print("FISHGUARD CLASSIFIER EVALUATION")
print("===================================")

print("Accuracy:", metrics["accuracy"], "%")
print("Precision:", metrics["precision_macro"], "%")
print("Recall:", metrics["recall_macro"], "%")
print("F1 Score:", metrics["f1_macro"], "%")

print("\nGenerated files:")

print("1.", cm_path)
print("2.", metrics_path)
print("3.", report_path)

print("\nEvaluation completed successfully!")
