# Leaffliction

Leaf-disease classification: distribution analysis, augmentation,
PlantCV-based transforms, and a trained classifier with predict/inference.
Team plan and role split: `leaffliction_team_plan.md`. Working rules for this
seat (Person A — Infra & Model): `CLAUDE.md`.

## Setup

The stack needs **Python 3.12**. Newer versions cannot build scipy from
source, which plantcv depends on, and TensorFlow does not yet ship wheels
for Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the environment before running anything. Without it the shebang
picks the system interpreter, which has no plantcv, and every program stops
at `ModuleNotFoundError`.

## Layout

```
leaffliction/
├── Distribution.py          # Part 1 — class distribution (pie + bar)
├── Augmentation.py          # Part 2 — Person B, six offline augmentations
├── Transformation.py        # Part 3 — Person C, PlantCV transforms
├── train.py                 # Part 4 — model + training loop
├── predict.py                # Part 4 — inference
├── leaffliction/
│   ├── io.py                # image load/save/walk — the only cv2 access point
│   ├── augment.py           # Person B — the 6 augmentation functions
│   ├── transform.py         # Person C — the PlantCV transforms
│   ├── split.py              # stratified train/val split
│   ├── viz.py                 # shared matplotlib helpers
│   ├── package.py            # builds the submission zip + signature.txt
│   └── metrics.py            # confusion matrix / per-class accuracy (used by train.py)
├── scripts/
│   └── make_toy_dataset.py  # generates a small synthetic dataset for dev
├── tests/                   # unit tests for io.py / split.py / package.py / metrics.py
├── requirements.txt
├── setup.cfg                # flake8 config
└── signature.txt
```

## Data set

Class directories hold the images, one directory per variety and disease:

```
images/
├── Apple_healthy/
├── Apple_scab/
├── Grape_Black_rot/
└── ...
```

The data set is not in this repository, as the subject requires.

## Usage

```bash
./Distribution.py ./Apple

./train.py ./Apple/
# splits ./Apple into train/val, builds a balanced augmented_directory from
# the train half only (leaffliction.augment's six offline augmentations,
# oversampling minority classes), then trains on that. Writes
# model/model.keras, model/labels.json, model/config.json,
# model/train_files.txt, model/val_files.txt, and (by default) packages
# leaffliction.zip + signature.txt. Add --no-package while iterating, or
# --no-augment to train on the raw split directly (fast toy-dataset runs).

./predict.py "./Apple/Apple_healthy/image (1).JPG"
```

### Transformation.py

Extract leaf features from one photo or a whole directory.

```bash
./Transformation.py "images/Apple_scab/image (1).JPG"
./Transformation.py -src images/Apple_scab -dst out
./Transformation.py -src images -dst out -mask -roi
./Transformation.py -h
```

Given an image it opens the transformations on screen, the histogram in its
own window. Given `-src` and `-dst` it walks the directory and writes the
results, mirroring the class directories into `-dst` and naming files
`{original}_{Transformation}{ext}`.

Quote paths: the data set uses spaces and parentheses in file names.

| Flag | Transformation |
| --- | --- |
| `-blur` | blurred LAB a channel, the input to the threshold |
| `-mask` | leaf with the background removed |
| `-roi` | leaf in green inside the region of interest |
| `-analyze` | contour, convex hull, axes and centroid |
| `-pseudolandmarks` | 3 x 20 homology points |
| `-histogram` | nine color channels measured inside the mask |

Flags combine. With none of them every transformation runs.

Exit codes: `0` success, `1` nothing to process, `2` bad arguments.

### Augmentation.py

Apply six offline augmentations to one image, or balance a whole data set
from the training split.

```bash
./Augmentation.py "images/Apple_scab/image (1).JPG"
./Augmentation.py images --train-list train_files.txt --output augmented_directory
./Augmentation.py -h
```

## The rule that matters most

**Split first, augment second.** `train.py` always computes the stratified
split from the raw directory and writes `model/{train,val}_files.txt` before
a single augmented pixel exists, then (by default) builds
`augmented_directory` from the train half only and trains on that.
`--train-dir` lets you point training at a directory already built the same
way instead of rebuilding it; `--no-augment` trains directly on the raw
split. Validation always reads the untouched raw images named in
`val_files.txt`, never anything under `augmented_directory`. See
`leaffliction_team_plan.md` section 0 for why getting this backwards zeroes
the grade.

## Development dataset

Don't develop against the full dataset. Build a small one:

```bash
python3 scripts/make_toy_dataset.py --out toy_dataset --per-class 10
./Distribution.py toy_dataset/Apple
./train.py toy_dataset/Apple --head-epochs 1 --fine-tune-epochs 1 --no-package
```

## Development

```bash
flake8 .
python3 -m pytest tests/
```

`setup.cfg` holds the flake8 configuration; `max-line-length` is 100.
