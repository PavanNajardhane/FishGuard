# FishGuard 🐟

**AI fish species classification and ABC-enhanced YOLO11m fish detection**, with a Flask web dashboard.

FishGuard combines three parts:

1. **ABC image calibration**: an Artificial Bee Colony optimizer tunes brightness, contrast, gamma, saturation and sharpness so underwater images are easier for the detector.
2. **YOLO11m detector**: finds fish and draws bounding boxes (baseline vs ABC-enhanced image, side by side).
3. **ResNet18 classifier**: identifies the species (23 classes).

## Web interface

The app has two pages, switched from the top navbar:

- **Dashboard**: workflow video, system architecture, classification metrics, confusion matrix, YOLO validation curves, precision–recall curve and ABC fitness convergence.
- **Model**: upload one image, then run classification and/or detection. ABC parameters and confidence are adjustable with sliders.

> Add your screenshots here, for example `static/screenshots/dashboard.png` and `static/screenshots/model.png`.

## Project structure

```
FishGuard/
├── app.py                    # Flask app
├── enhance.py                # ABC image enhancement
├── abc_optimize.py           # ABC parameter search
├── train.py                  # YOLO11m training
├── train_classifier.py       # ResNet18 training
├── evaluate.py               # detector: baseline vs ABC
├── evaluate_classifier.py    # classifier evaluation
├── prepare_dataset.py        # dataset check / split
├── setup_dataset.py
├── best_params.json          # created by abc_optimize.py
├── requirements.txt
├── templates/index.html
├── static/                   # images, videos, metrics json
└── weights/
    ├── fish_detector.pt      # YOLO11m weights
    ├── fish_classifier.pth   # ResNet18 weights
    └── class_split.json
```

## 1. Install

Python 3.10+ is required. A GPU is strongly recommended for training.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

If `torch` fails to install, get the right command for your system from [pytorch.org](https://pytorch.org).

## 2. Dataset

The dataset is not stored in this repository because of its size. Download the Fish4Knowledge-based dataset and place it in the project root as `fish4knowedge_dataset/`.

For the detector, use YOLO format:

```
dataset/images/{train,val,test}/*.jpg
dataset/labels/{train,val,test}/*.txt
dataset/data.yaml
```

Check it:

```bash
python prepare_dataset.py --data dataset --check
```

If you only have one pool of images and labels:

```bash
python prepare_dataset.py --data pool --resplit --out dataset --names Class0 Class1 ...
```

## 3. Train the YOLO11m detector

Settings: Adam, lr 0.001, 30 epochs, image size 640, batch 16.

```bash
python train.py --data dataset/data.yaml --device 0
```

Copy the best weights to the path the app expects:

```bash
copy runs\yolo11m_fish\weights\best.pt weights\fish_detector.pt
```

## 4. Train the classifier

```bash
python train_classifier.py
python evaluate_classifier.py
```

This produces `weights/fish_classifier.pth`, `static/classification_metrics.json` and `static/confusion_matrix.png`.

## 5. Run ABC on the validation set only

8 iterations:

```bash
python abc_optimize.py --data dataset/data.yaml --colony 6 --iters 8 --limit-imgs 300 --device 0
```

Outputs `best_params.json` and `abc_convergence.png`. Each fitness call is one YOLO validation pass; drop `--limit-imgs` to use the full validation set.

## 6. Evaluate on the test set (baseline vs ABC)

```bash
python evaluate.py --data dataset/data.yaml --device 0
```

Prints the baseline vs ABC comparison and writes `test_results.json`.

## 7. Run the web app

```bash
python app.py
```

Open <http://127.0.0.1:5000>.

Requirements for the app to start:

- `weights/fish_detector.pt` and `weights/fish_classifier.pth` exist
- `enhance.py` and the `static/` folder are present
- the command is run from the project root

If `best_params.json` is missing, the app falls back to the paper's default ABC parameters.

## Results

| Item | Value |
|---|---|
| Detector | YOLO11m, 30 epochs |
| Detector mAP@0.5 (precision–recall curve, all classes) | 0.846 |
| Classifier | ResNet18, 23 species |
| Classifier evaluation set | 5,465 images |

Training graphs, the confusion matrix, the precision–recall curve and ABC convergence are shown on the Dashboard page.

## No GPU?

Run the training steps (3–6) on Kaggle or Colab with a free GPU, download `fish_detector.pt`, `fish_classifier.pth` and `best_params.json`, then run the web app locally.

## Notes

- Metrics shown on the Dashboard are read from saved files in `static/`. They are only as reliable as the evaluation split that produced them.
- Dataset and virtual environment folders are excluded from Git (see `.gitignore`).
