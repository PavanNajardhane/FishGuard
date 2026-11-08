"""
FishGuard - Fish Classification, YOLO Detection and ABC Enhancement
"""

import os
import json
import base64
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

from enhance import PAPER_PARAMS, enhance_bgr


# =========================================
# CONFIGURATION
# =========================================

WEIGHTS = os.environ.get(
    "FISHGUARD_WEIGHTS",
    "weights/fish_detector.pt"
)

CLASSIFIER_WEIGHTS = "weights/fish_classifier.pth"
DATASET_DIR = Path("fish4knowedge_dataset")
STATIC_DIR = Path("static")

app = Flask(__name__)

_model = None
_classifier = None
_classifier_classes = None


# =========================================
# CLASSIFICATION TRANSFORM
# =========================================

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def display_species(species):
    return species.split("_", 1)[-1].replace("_", " ")


# =========================================
# LOAD YOLO MODEL
# =========================================

def model():
    global _model

    if _model is None:
        if not Path(WEIGHTS).exists():
            raise FileNotFoundError(
                f"YOLO model not found: {WEIGHTS}"
            )

        _model = YOLO(WEIGHTS)

    return _model


# =========================================
# LOAD RESNET18 CLASSIFIER
# =========================================

def classifier_model():
    global _classifier, _classifier_classes

    if _classifier is None:

        if not Path(CLASSIFIER_WEIGHTS).exists():
            raise FileNotFoundError(
                f"Classifier model not found: {CLASSIFIER_WEIGHTS}"
            )

        checkpoint = torch.load(
            CLASSIFIER_WEIGHTS,
            map_location="cpu"
        )

        _classifier_classes = checkpoint["classes"]

        net = models.resnet18(weights=None)

        net.fc = nn.Linear(
            net.fc.in_features,
            len(_classifier_classes)
        )

        net.load_state_dict(
            checkpoint["model_state_dict"]
        )

        net.eval()
        _classifier = net

    return _classifier


# =========================================
# ABC PARAMETERS
# =========================================

def default_params():

    file_path = Path("best_params.json")

    if file_path.exists():

        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)["params"]

        except (ValueError, KeyError, OSError):
            app.logger.exception(
                "Could not load ABC parameters"
            )

    return PAPER_PARAMS.copy()


# =========================================
# IMAGE TO BASE64
# =========================================

def b64(img):

    success, buffer = cv2.imencode(".jpg", img)

    if not success:
        raise ValueError("Image encoding failed")

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(buffer).decode("utf-8")
    )


# =========================================
# HOMEPAGE
# =========================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        params=default_params()
    )


# =========================================
# FISH CLASSIFICATION
# =========================================

@app.route("/classify", methods=["POST"])
def classify():

    file = request.files.get("image")

    if not file or not file.filename:
        return jsonify(
            error="Please upload a fish image"
        ), 400

    try:

        image = Image.open(
            BytesIO(file.read())
        ).convert("RGB")

        net = classifier_model()

        image_tensor = classifier_transform(
            image
        ).unsqueeze(0)

        with torch.inference_mode():

            outputs = net(image_tensor)

            probabilities = torch.softmax(
                outputs, dim=1
            )

            confidence, predicted = torch.max(
                probabilities, 1
            )

        species = _classifier_classes[
            predicted.item()
        ]

        return jsonify({
            "species": display_species(species),
            "class_name": species,
            "confidence": round(
                confidence.item() * 100, 2
            ),
            "message": "Classification successful"
        })

    except Exception as error:

        app.logger.exception("Classification failed")

        return jsonify(
            error="Classification failed: " + str(error)
        ), 500


# =========================================
# YOLO DETECTION FUNCTION
# =========================================

def run_detection(yolo_model, image, confidence):

    result = yolo_model.predict(
        image,
        conf=confidence,
        imgsz=640,
        verbose=False
    )[0]

    detections = []

    height, width = image.shape[:2]

    for box in result.boxes:

        class_id = int(box.cls.item())

        detector_class = yolo_model.names[class_id]

        detector_confidence = round(
            float(box.conf.item()) * 100, 2
        )

        x1, y1, x2, y2 = [
            int(v) for v in box.xyxy[0].tolist()
        ]

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        detections.append({
            "class": detector_class,
            "detector_class": detector_class,
            "conf": round(
                detector_confidence / 100, 3
            ),
            "detector_confidence": detector_confidence,
            "box": [x1, y1, x2, y2]
        })

    return {
        "image": b64(result.plot()),
        "detections": detections
    }


# =========================================
# YOLO DETECTION + ABC ENHANCEMENT
# =========================================

@app.route("/detect", methods=["POST"])
def detect():

    file = request.files.get("image")

    if not file or not file.filename:
        return jsonify(
            error="No image uploaded"
        ), 400

    try:

        image = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify(
                error="Could not read image"
            ), 400

        try:
            confidence = float(
                request.form.get("conf", 0.25)
            )

        except ValueError:
            return jsonify(
                error="Invalid confidence value"
            ), 400

        if not 0 <= confidence <= 1:
            return jsonify(
                error="Confidence must be between 0 and 1"
            ), 400

        yolo_model = model()

        params = default_params()

        for key in params:

            value = request.form.get(key)

            if value is not None and value != "":
                params[key] = float(value)

        enhanced_image = enhance_bgr(
            image,
            params
        )

        baseline_result = run_detection(
            yolo_model,
            image,
            confidence
        )

        enhanced_result = run_detection(
            yolo_model,
            enhanced_image,
            confidence
        )

        return jsonify({
            "baseline": baseline_result,
            "enhanced": enhanced_result,
            "params": params,
            "enhanced_input": b64(enhanced_image)
        })

    except FileNotFoundError as error:

        return jsonify(error=str(error)), 503

    except Exception as error:

        app.logger.exception("Detection failed")

        return jsonify(
            error="Detection failed: " + str(error)
        ), 500


# =========================================
# CLASSIFICATION EVALUATION RESULTS
# =========================================

@app.route("/results")
def results():

    file_path = STATIC_DIR / "classification_metrics.json"

    if not file_path.exists():
        return jsonify({})

    try:

        with open(file_path, encoding="utf-8") as file:
            return jsonify(json.load(file))

    except Exception as error:

        app.logger.exception(
            "Could not read classification metrics"
        )

        return jsonify(
            error=str(error)
        ), 500


# =========================================
# MODEL AND DATA ANALYTICS
# =========================================

@app.route("/analytics")
def analytics():
    """Return only analytics that can be verified from files on disk."""
    metrics_path = STATIC_DIR / "classification_metrics.json"
    confusion_path = STATIC_DIR / "confusion_matrix.png"

    try:
        metrics = {}
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as file:
                metrics = json.load(file)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        class_names, class_counts = [], {}
        dataset_image_count = 0

        if DATASET_DIR.exists():
            class_dirs = sorted(
                folder for folder in DATASET_DIR.iterdir()
                if folder.is_dir()
            )
            class_names = [folder.name for folder in class_dirs]
            for folder in class_dirs:
                count = sum(
                    1 for path in folder.rglob("*")
                    if path.is_file() and path.suffix.lower() in image_extensions
                )
                class_counts[folder.name] = count
                dataset_image_count += count

        history_candidates = [
            STATIC_DIR / "training_history.json",
            Path("training_history.json"),
            Path("history.json"),
        ]
        history_file = next(
            (path for path in history_candidates if path.is_file()), None
        )
        training_history = None
        if history_file:
            with open(history_file, encoding="utf-8") as file:
                training_history = json.load(file)

        yolo_csv_files = (
            sorted(Path("runs").glob("**/results.csv"))
            if Path("runs").exists() else []
        )

        abc_candidates = [
            Path("abc_history.json"),
            Path("abc_convergence.json"),
            STATIC_DIR / "abc_history.json",
        ]
        abc_file = next(
            (path for path in abc_candidates if path.is_file()), None
        )
        abc_history = None
        if abc_file:
            with open(abc_file, encoding="utf-8") as file:
                abc_history = json.load(file)

        best_params_path = Path("best_params.json")
        best_params = None
        if best_params_path.is_file():
            try:
                with open(best_params_path, encoding="utf-8") as file:
                    best_params = json.load(file)
            except (OSError, ValueError):
                app.logger.exception("Could not read best_params.json")

        notes = {
            "classifier": "ResNet18",
            "classifier_weights": CLASSIFIER_WEIGHTS,
            "classifier_weights_available": Path(CLASSIFIER_WEIGHTS).is_file(),
            "detector": "YOLO",
            "detector_weights": WEIGHTS,
            "detector_weights_available": Path(WEIGHTS).is_file(),
            "dataset": str(DATASET_DIR),
            "number_of_classes": len(class_names),
            "dataset_images": dataset_image_count,
            "class_names": class_names,
            "class_counts": class_counts,
            "training_history_available": training_history is not None,
            "yolo_validation_available": bool(yolo_csv_files),
            "yolo_validation_files": [str(path) for path in yolo_csv_files],
            "abc_history_available": abc_history is not None,
            "best_params_available": best_params_path.is_file(),
            "confusion_matrix_available": confusion_path.is_file(),
            "evaluation_note": (
                "Previously saved classification evaluation metrics. "
                "Do not call them an independent test result unless the "
                "evaluation split has been verified."
            ),
        }

        return jsonify({
            "metrics": metrics,
            "notes": notes,
            "training_history": training_history,
            "abc_history": abc_history,
            "best_params": best_params,
        })

    except Exception as error:
        app.logger.exception("Analytics loading failed")
        return jsonify(error=str(error)), 500

# =========================================
# RUN APPLICATION
# =========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

