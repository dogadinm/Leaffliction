# Leaffliction

Recognise plant disease from a photograph of a leaf.

The data set is PlantVillage: 256x256 photographs of a single leaf on a plain
background, one directory per variety and disease. The programs here walk that
data set in four steps — measure it, balance it, extract features from it, and
train a classifier that names the disease on an unseen photo with over 90%
accuracy on a held-out validation set.

| Part | Program | What it answers |
| --- | --- | --- |
| 1 | `Distribution.py` | how unbalanced is the data set? |
| 2 | `Augmentation.py` | how do we even it out? |
| 3 | `Transformation.py` | what can be measured on a leaf? |
| 4 | `train.py`, `predict.py` | which disease is this? |

## Setup

The stack needs **Python 3.12**. Newer versions cannot build scipy from
source, which plantcv depends on, and TensorFlow does not yet ship wheels for
Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the environment before running anything. Without it the shebang picks
the system interpreter, which has no plantcv, and every program stops at
`ModuleNotFoundError`.

Running the test suite additionally needs `pip install pytest`; it is a
development tool and deliberately not a runtime dependency.

## Data set

One directory per class, the directory name **is** the label:

```
images/
├── Apple_healthy/
├── Apple_scab/
├── Grape_Black_rot/
└── ...
```

The data set is never committed, as the subject requires. Nested layouts such
as `Apple/apple_healthy/` work too — the directory walk is recursive.

Quote every path you pass in: the file names contain spaces and parentheses,
as in `"images/Apple_scab/image (1).JPG"`.

## Layout

```
.
├── Distribution.py           # Part 1 — class distribution (pie + bar)
├── Augmentation.py           # Part 2 — six offline augmentations
├── Transformation.py         # Part 3 — PlantCV feature extraction
├── train.py                  # Part 4 — split, augment, train, package
├── predict.py                # Part 4 — inference on one image
├── leaffliction/
│   ├── io.py                 # load/save/walk — the only cv2 access point
│   ├── viz.py                # shared matplotlib helpers
│   ├── augment.py            # the 6 augmentation functions
│   ├── transform.py          # the PlantCV transforms
│   ├── split.py              # stratified train/val split
│   ├── metrics.py            # confusion matrix, per-class accuracy
│   └── package.py            # submission zip + signature.txt
├── tests/
├── requirements.txt
├── setup.cfg                 # flake8 config
└── signature.txt
```

Every module loads images through `leaffliction/io.py`, the only file allowed
to call `cv2.imread`/`cv2.imwrite`. That is what keeps the BGR/RGB byte order
decided in exactly one place: images are RGB everywhere inside the project.

---

## Distribution.py

Part 1. Plots a pie chart (class share) and a bar chart (image count),
labelled from the subdirectory names, so it works on any plant directory.

```bash
./Distribution.py images
./Distribution.py -h
```

This is how the imbalance becomes visible — roughly six times more healthy
apple leaves than rusted ones. A classifier trained on that learns to answer
"the most common class" rather than to recognise disease, which is what Part 2
exists to fix.

## Augmentation.py

Part 2. Six offline augmentations: Flip, Rotate, Skew, Shear, Crop,
Distortion.

```bash
# one image: displays the six and writes them next to the original
./Augmentation.py "images/Apple_scab/image (1).JPG"

# balance a data set, reading only the training half of the split
./Augmentation.py images \
    --train-list model/train_files.txt \
    --output augmented_directory \
    --no-display

./Augmentation.py -h
```

Files are named `{original}_{Kind}{ext}`, for example
`image (1)_Flip.JPG` — the evaluator checks this with `ls`.

## Transformation.py

Part 3. Extracts leaf features from one photo or a whole directory.

```bash
./Transformation.py "images/Apple_scab/image (1).JPG"
./Transformation.py "images/Apple_scab/image (1).JPG" -histogram
./Transformation.py -src images/Apple_scab -dst out
./Transformation.py -src images -dst out -mask -roi
./Transformation.py -h
```

Given an image it opens the transformations on screen, the histogram in its
own window. Given `-src` and `-dst` it walks the directory and writes the
results, mirroring the class directories into `-dst` and naming files
`{original}_{Transformation}{ext}`.

| Flag | Transformation |
| --- | --- |
| `-blur` | blurred LAB a channel, the input to the threshold |
| `-mask` | leaf with the background removed |
| `-roi` | leaf in green inside the region of interest |
| `-analyze` | contour, convex hull, axes and centroid |
| `-pseudolandmarks` | 3 x 20 homology points |
| `-histogram` | nine colour channels measured inside the mask |

Flags combine. With none of them every transformation runs.

Everything rests on one binary mask, computed once per photo: the LAB `a` and
`b` channels are thresholded with Otsu and OR-ed together, holes enclosed by
the leaf are filled so that dark lesions stay inside the mask, and only the
largest blob survives. The colour histogram is measured inside that mask and
normalised to percent, so leaves of different sizes stay comparable.

Exit codes: `0` success, `1` nothing to process, `2` bad arguments.

## train.py

Part 4, model half. Transfer learning on MobileNetV2: train a frozen-base head
to convergence, then unfreeze the top block and fine-tune at a 10x lower
learning rate.

```bash
# full run
./train.py images

# fast iteration
./train.py images --no-augment --no-package \
    --head-epochs 3 --fine-tune-epochs 2 --batch-size 16

./train.py -h
```

It splits the raw directory into train/val, writes
`model/train_files.txt` and `model/val_files.txt`, builds a balanced
`augmented_directory` from the train half only, trains on that, prints a
confusion matrix with per-class accuracy, and by default packages
`leaffliction.zip` + `signature.txt`.

Use `--no-package` while iterating (see the warning below), `--no-augment` to
train on the raw split directly, and `--model-dir` to write somewhere other
than `model/`.

Outputs: `model/model.keras`, `model/labels.json`, `model/config.json`.

## predict.py

Part 4, inference. Displays the original image, a transformed version, and the
predicted class.

```bash
./predict.py "images/Apple_scab/image (1).JPG"
./predict.py "images/Apple_scab/image (1).JPG" --transform Mask
./predict.py "images/Apple_scab/image (1).JPG" --no-display
```

`--transform` accepts any name from `leaffliction.transform.TRANSFORMS`.
`--model-dir` defaults to `model/`. This program loads only `model.keras`,
`labels.json` and `config.json`, and never imports `train.py`.

---

## The rule that matters most

**Split first, augment second.** `train.py` always computes the stratified
split from the raw directory and writes `model/{train,val}_files.txt` before a
single augmented pixel exists, then builds `augmented_directory` from the
train half only and trains on that. Validation always reads the untouched raw
images named in `val_files.txt`, never anything under `augmented_directory`.

Getting this backwards puts a rotated copy of a training image into the
validation set. Accuracy looks excellent and the grade is zero.

Proving it takes two commands:

```bash
wc -l model/val_files.txt      # the subject requires >= 100 validation images
head -3 model/train_files.txt  # the split, recorded before augmentation ran
```

## Submission

```bash
python3 -m leaffliction.package
shasum leaffliction.zip        # must equal the contents of signature.txt
cat signature.txt
```

**Run this exactly once, last.** The module has no `--help`: any invocation
rebuilds the archive and overwrites `signature.txt`. If the zip is rebuilt
after the hash is recorded, the two no longer match and the grade is zero.
`train.py` also packages by default, which is why `--no-package` exists.

Before handing in, check that nothing large or generated slipped into git:

```bash
git ls-files | grep -iE "^(images|model|augmented_directory|toy_dataset|\.venv)/|\.zip$"
```

The output must be empty. Committing the data set is an automatic zero.

## Development

```bash
flake8 .                       # must print nothing
python3 -m pytest -q           # needs pip install pytest
```

`setup.cfg` holds the flake8 configuration; `max-line-length` is 100.

Do not develop against the full data set — 7000 images turn every check into
a coffee break. Copy a subset instead:

```bash
for c in Apple_healthy Apple_scab Grape_healthy Grape_Black_rot; do
    mkdir -p "toy_dataset/$c"
    ls "images/$c" | head -80 | while read f; do
        cp "images/$c/$f" "toy_dataset/$c/"
    done
done

./Distribution.py toy_dataset
./Transformation.py -src toy_dataset -dst out
./train.py toy_dataset --no-augment --no-package \
    --head-epochs 3 --fine-tune-epochs 2 --batch-size 16
```

Real leaves, not synthetic images: the segmentation in `Transformation.py`
needs an actual leaf on an actual background to mean anything. `toy_dataset/`
is gitignored.

Dry run on a clean clone, the day before the defense:

```bash
git clone . /tmp/clone && cd /tmp/clone
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./Transformation.py -h
```
