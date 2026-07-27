# Leaffliction

Leaf-disease classification: distribution analysis, augmentation,
PlantCV-based transforms, and a trained classifier with predict/inference.
Team plan and role split: `leaffliction_team_plan.md`. Working rules for this
seat (Person A — Infra & Model): `CLAUDE.md`.

## Setup

TensorFlow does not yet ship wheels for Python 3.14 — use Python 3.10–3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
leaffliction/
├── Distribution.py          # Part 1 — class distribution (pie + bar)
├── Augmentation.py          # Part 2 — Person B, stub for now
├── Transformation.py        # Part 3 — Person C, stub for now
├── train.py                 # Part 4 — model + training loop
├── predict.py                # Part 4 — inference
├── leaffliction/
│   ├── io.py                # image load/save/walk — the only cv2 access point
│   ├── augment.py           # Person B, stub for now
│   ├── transform.py         # Person C, stub for now
│   ├── split.py              # stratified train/val split
│   ├── viz.py                 # shared matplotlib helpers
│   └── package.py            # builds the submission zip + signature.txt
├── scripts/
│   └── make_toy_dataset.py  # generates a small synthetic dataset for dev
├── tests/                   # unit tests for io.py / split.py / package.py
├── requirements.txt
├── setup.cfg                # flake8 config
└── signature.txt
```

## Usage

```bash
./Distribution.py ./Apple

./train.py ./Apple/
# writes model/model.keras, model/labels.json, model/config.json,
# model/train_files.txt, model/val_files.txt, and (by default) packages
# leaffliction.zip + signature.txt. Add --no-package while iterating.

./predict.py "./Apple/Apple_healthy/image (1).JPG"
```

## The rule that matters most

**Split first, augment second.** `train.py` always computes the stratified
split from the raw directory and writes `model/{train,val}_files.txt` before
touching any augmented directory. `--train-dir` only changes where *training*
pixels are read from; validation always reads the untouched raw images named
in `val_files.txt`. See `leaffliction_team_plan.md` section 0 for why getting
this backwards zeroes the grade.

## Development dataset

Don't develop against the full dataset. Build a small one:

```bash
python3 scripts/make_toy_dataset.py --out toy_dataset --per-class 10
./Distribution.py toy_dataset/Apple
./train.py toy_dataset/Apple --head-epochs 1 --fine-tune-epochs 1 --no-package
```

## Testing

```bash
flake8 .
python3 -m pytest tests/
```
