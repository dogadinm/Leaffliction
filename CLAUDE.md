# Leaffliction — A - Elena (Infra & Model)

This checkout belongs to **A - Elena** on a 3-person team building the 42 School
"Leaffliction" leaf-disease classifier. Full team plan: `leaffliction_team_plan.md`.
Read it once for the schedule and study guide — this file only encodes the rules
that must never be silently violated while coding here.

## Scope — what this seat owns

Owned files (write freely):
`Distribution.py`, `train.py` (model half), `predict.py`,
`leaffliction/io.py`, `leaffliction/split.py`, `leaffliction/viz.py`,
`setup.cfg`, `.gitignore`, `signature.txt`, top-level `README.md`, repo skeleton.

**Do not edit** `Augmentation.py`, `Transformation.py`, `leaffliction/augment.py`,
`leaffliction/transform.py` — those belong to B - Tanya and C - Misha. If a change
there is needed, it's a message to them, not a diff from this seat. `train.py`'s
pipeline half (`leaffliction/pipeline.py` if the file gets split) is B's — only
touch the model/training-loop half.

## The two rules that decide the grade

1. **Split first, augment second.** Stratified split into train/val happens on
   the *raw* directory. Only the training half ever gets augmented. Never call
   any training code on a directory that was balanced/augmented before the
   split existed. If you can't point at the line of code that split before B's
   augmented directory was built, the validation set is leaked and the whole
   project is a 0 at defense.
2. **Part 4 (training+packaging) is the biggest part.** Don't let it balloon
   silently — training, accuracy-chasing, zip, signature are all this seat's
   job and are explicitly budgeted as the bulk of the ~19h estimate.

## Frozen interfaces — do not change without a team sync

These were agreed on day 1 across all three people. B and C's code imports
`leaffliction/io.py`; changing its signatures breaks their branches.

```python
# leaffliction/io.py — this seat writes it first, everyone else imports it.
# Nobody calls cv2.imread directly anywhere in the repo — that's the classic
# BGR/RGB bug where a mask works in one file and silently breaks in another.
def iter_images(root: Path) -> Iterator[Path]: ...
def load_image(path: Path) -> np.ndarray:   # RGB uint8, HxWx3 — not BGR
def label_of(path: Path) -> str:            # parent dir name, e.g. "Apple_healthy"
def class_counts(root: Path) -> dict[str, int]: ...
def save_image(img: np.ndarray, path: Path) -> None: ...
```

```python
# leaffliction/split.py — this seat's contract with B
def stratified_split(root: Path, val_split: float, seed: int) -> tuple[list[Path], list[Path]]: ...
# Must write train_files.txt / val_files.txt so the split is provable in code,
# not just claimed at defense.
```

Model contract — `predict.py` loads **only** these three files, never imports
`train.py`:

```
model/
├── model.keras
├── labels.json     # {"0": "Apple_healthy", "1": "Apple_scab", ...}  — B writes this
└── config.json      # {"img_size": 128, "seed": 42, "val_split": 0.2, ...}
```

Constants hardcoded identically everywhere (do not change without the whole
team agreeing again): `IMG_SIZE = 128`, `SEED = 42`, `VAL_SPLIT = 0.2`.

## Coding standards

- `flake8 .` must be clean before every commit — the subject grades this.
- Every CLI entrypoint (`Distribution.py`, `train.py`, `predict.py`) takes a
  directory or image path from `argv` and supports `-h`.
- Handle a corrupt or non-image file without a traceback — catch it in
  `io.py`, not ad hoc in every caller.
- No dataset, model weights, or zip ever committed. `.gitignore` must exclude
  `images/`, `model/`, `*.zip`, `toy_dataset/`.
- One branch per task (`feat/distribution`, `feat/train`, `feat/predict`), no
  direct pushes to `main` alone.

## Testing

Build and develop against `toy_dataset/` (10 images per class) — the subject's
evaluation explicitly checks behavior on small datasets, not the full
20k-image set. Don't wait on a full-dataset run to sanity-check a change.

## Packaging discipline (this seat owns the zip)

- Compute the sha1 of the zip **last**, after everything is frozen.
- Never rebuild the zip after `signature.txt` is written — the hash changes
  and the grade goes to 0. If a late fix is needed, rebuild the zip *and*
  regenerate `signature.txt` together, in the same commit.
- Nobody else touches the zip.

## Delivery checklist (this seat's items, from the full checklist in the plan)

- [ ] `./Distribution.py ./Apple` prints pie + bar, labels from directory names
- [ ] Works on every plant directory in the dataset, not just Apple
- [ ] `./train.py ./Apple/` trains and produces the zip
- [ ] `./predict.py "<image>"` shows original + transformed + predicted class
- [ ] Validation accuracy > 90%, validation set ≥ 100 images, both provable in code
- [ ] `signature.txt` contains the sha1 of the final, frozen zip
- [ ] `flake8 .` returns nothing
- [ ] Delete `model/`, retrain from the repo — does it still work?
