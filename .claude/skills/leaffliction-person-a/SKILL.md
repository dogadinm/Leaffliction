---
name: leaffliction-person-a
description: "INVOKE THIS SKILL when building or reviewing A - Elena's parts of the Leaffliction project — repo skeleton, io.py, Distribution.py, split.py, the model + training loop in train.py, predict.py, or the zip/signature.txt packaging step. Covers the exact contracts B and C's code depends on, the split-before-augment rule, transfer-learning setup, and the freeze-the-zip-last discipline."
---

<oneliner>
A - Elena owns infra + model: repo skeleton, `Distribution.py`, `leaffliction/io.py`,
`leaffliction/split.py`, `train.py` (model half), `predict.py`, and packaging
(zip + `signature.txt`). Everything here is scoped to that seat only — B owns
augmentation, C owns PlantCV transforms. Full context: `leaffliction_team_plan.md`
and `CLAUDE.md` at the repo root.
</oneliner>

## Non-negotiable order of operations

Do the steps in this order regardless of what looks fastest. The two failure
modes that zero the grade both come from doing things out of order:

1. `io.py` before anything else touches an image — everyone else's imports depend on it.
2. Stratified split of the **raw** directory before any augmented directory exists
   or is trained on. If B's `augmented_directory` already exists when you write
   `split.py`, that's a sign the order was violated upstream — flag it, don't
   route around it by training on the augmented set.
3. Train, then freeze `model/`, then build the zip, then compute the sha1
   **last**. Never touch the zip again after `signature.txt` is written.

## 1. Repo skeleton

```
leaffliction/
├── Distribution.py
├── train.py
├── predict.py
├── leaffliction/
│   ├── __init__.py
│   ├── io.py
│   ├── split.py
│   └── viz.py
├── requirements.txt
├── setup.cfg                # flake8 config
├── .gitignore                # MUST exclude images/, model/, *.zip, toy_dataset/
├── signature.txt
└── README.md
```

Stub `Augmentation.py`, `Transformation.py`, `leaffliction/augment.py`,
`leaffliction/transform.py` as empty placeholders so imports resolve, but do
not implement them — that's B and C's work.

`setup.cfg` flake8 example:

```ini
[flake8]
max-line-length = 100
exclude = .git,__pycache__,model,images,toy_dataset
```

## 2. `leaffliction/io.py` — write this first

```python
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_images(root: Path) -> Iterator[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for p in sorted(root.rglob("*")):
        if p.suffix in exts:
            yield p


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label_of(path: Path) -> str:
    return path.parent.name


def class_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in iter_images(root):
        counts[label_of(p)] = counts.get(label_of(p), 0) + 1
    return counts


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
```

This is the *only* place `cv2.imread`/`cv2.imwrite` should appear in the whole
repo. If a grep for `cv2.imread` outside this file turns up a hit in B's or
C's code, that's worth flagging in the daily sync, not silently fixing in
their files.

Wrap `load_image` callers to skip corrupt files without a traceback — the
checklist requires this ("handles a corrupt or non-image file without a
traceback").

## 3. `Distribution.py`

- Takes a plant-type directory from argv (e.g. `./Distribution.py ./Apple`).
- Uses `class_counts()` from `io.py`, nothing else touches the filesystem.
- Renders a pie chart and a bar chart, labels taken from the subdirectory
  names — not hardcoded class names.
- Must run on every plant directory in the dataset, not just the one used to
  develop it.

```python
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from leaffliction.io import class_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    counts = class_counts(args.directory)
    if not counts:
        print(f"no images found under {args.directory}", file=sys.stderr)
        return 1

    labels, values = zip(*sorted(counts.items()))
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))
    ax_pie.pie(values, labels=labels, autopct="%1.1f%%")
    ax_bar.bar(labels, values)
    ax_bar.tick_params(axis="x", rotation=45)
    fig.suptitle(args.directory.name)
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 4. `leaffliction/split.py` — stratified, and provable

The split must happen on the raw directory, per class, before B's
augmentation pipeline runs. Write the file lists to disk — "prove the count
in code, not a claim" is a rehearsed defense question.

```python
from pathlib import Path

import numpy as np

from leaffliction.io import iter_images, label_of


def stratified_split(
    root: Path, val_split: float, seed: int
) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Path]] = {}
    for p in iter_images(root):
        by_class.setdefault(label_of(p), []).append(p)

    train, val = [], []
    for label, paths in sorted(by_class.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_split))
        val.extend(paths[:n_val])
        train.extend(paths[n_val:])
    return train, val


def write_split(train: list[Path], val: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_files.txt").write_text("\n".join(map(str, train)))
    (out_dir / "val_files.txt").write_text("\n".join(map(str, val)))
```

Sanity check to run before moving on: `len(val) >= 100` on the real dataset —
the subject requires it, and `wc -l val_files.txt` is the exact proof an
evaluator will ask for.

## 5. `train.py` — model half (B owns the input-pipeline half)

- Transfer learning first, not a from-scratch CNN — it's the fastest route
  past 90% and the plan explicitly budgets fine-tuning for later, not a
  custom architecture from day one.
- Sequence: frozen base (MobileNetV2/EfficientNetB0) → train head → unfreeze
  top block, fine-tune at a 10x lower learning rate.
- Reads `train_files.txt`/`val_files.txt` from `split.py`. Once B's
  `augmented_directory` exists, point training at that for the *train* split
  only — `val_files.txt` still points at untouched raw images.
- On completion, writes `model/model.keras`, `model/config.json`
  (`img_size`, `seed`, `val_split`, plus whatever else predict.py needs).
  `model/labels.json` is B's output from the input pipeline — read it, don't
  regenerate it.
- Still be ready to describe the from-scratch CNN you'd write instead
  (conv → BN → ReLU → pool, repeated, then dense → softmax) — it's a
  rehearsed defense question even though transfer learning is what ships.

Checkpointing / early stopping: save the best val-accuracy epoch, not the
last one. Keep the loss curves (train vs val) available — "your accuracy is
96%, how do I know you're not overfitting, show the curves" is on the
rehearsed question list.

## 6. `predict.py`

- Loads only `model/model.keras`, `model/labels.json`, `model/config.json`.
  Never imports `train.py` — that's the point of the model contract.
- Shows: original image, the transformed version (call into C's
  `leaffliction/transform.py` for display only, don't reimplement it), and
  the predicted class.
- Must survive: delete `model/`, retrain from the repo, does `predict.py`
  still work against the fresh artifacts? That's a checklist item.

## 7. Packaging — zip + `signature.txt`

This seat is the only one who touches the zip. Order:

```bash
# 1. Freeze model/ — no more training runs after this point.
# 2. Build the zip with exactly what the subject requires (code + model/,
#    never the dataset).
zip -r leaffliction.zip . -x '*.git*' -x 'images/*' -x 'toy_dataset/*'

# 3. Compute the hash LAST, once, and never rebuild the zip afterward.
shasum leaffliction.zip > signature.txt   # or: sha1sum on Linux
```

If a bug is found after `signature.txt` exists, rebuild the zip *and*
regenerate `signature.txt` in the same commit — never let them drift apart.

## Rehearsed answers this seat must own at defense

From the full question list in `leaffliction_team_plan.md` §6, the ones that
land on this seat specifically:

- "Show me the split. Did you augment before or after? Prove it." →
  point at `split.py` writing `train_files.txt`/`val_files.txt` before any
  augmented directory existed.
- "How many images in the validation set? Show the count in code." →
  `wc -l model/val_files.txt` or equivalent, not a verbal claim.
- "Draw the CNN architecture. What does each layer do?" and "Why freeze the
  base model, then unfreeze it?" → both transfer-learning questions.
- "Your accuracy is 96%. How do I know you're not overfitting? Show the
  curves." → have the train/val loss plot ready, not just the final number.
- "Delete `model/`, retrain from the repo. Does it work?" → this is a literal
  checklist item, run it before the defense, not during it.
