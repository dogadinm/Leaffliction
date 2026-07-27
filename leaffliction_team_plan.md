# Leaffliction — Plan for a Team of 3

## 0. The two traps that decide this project

Read these before you split anything.

**Trap 1: Part 4 is bigger than Parts 1, 2 and 3 combined.** If you assign "one part per person", one person carries the model, the accuracy target, the zip, the signature and the defense, while another writes two pie charts. Part 4 must be sliced across all three.

**Trap 2: augment first, split later = leaked validation set.** If you balance the whole dataset and *then* split into train/val, a rotated copy of a training image lands in validation. You get 99% accuracy and a 0 at defense when the evaluator asks how you split. **Split first. Augment the training half only.** Write this rule on the wall.

Everything below assumes both.

---

## 1. Roles

Three roles, roughly 17–19 hours each. Names are placeholders.

| | Owner | Owns | Est. |
|---|---|---|---|
| **A** | Infra & Model | repo skeleton, `Distribution.py`, train/val split, `train.py` model + training loop, `predict.py`, zip + `signature.txt` | ~19h |
| **B** | Data | `Augmentation.py`, balancing pipeline, `augmented_directory`, tf.data/DataLoader input pipeline, metrics + confusion matrix, the ≥100-image validation proof | ~17h |
| **C** | Vision | `Transformation.py` in full (6+ transforms, color histogram, `-src`/`-dst`/`-h` CLI, batch mode) | ~17h |

Why this shape:

- Part 1 is 4 hours of work, so A pays for it with the model and the packaging.
- Part 3 is the single hardest file (PlantCV fights back), so C gets it alone and nothing else.
- B sits between them: augmentation feeds training, so B owns the pipeline from disk to batched tensors.

Everyone still has to explain everything at the defense. Section 6 covers that.

---

## 2. Freeze the interfaces on day 1 (2-hour session, all three)

You cannot work independently without contracts. Agree on these, commit them as stubs, then split.

### Repo layout

```
leaffliction/
├── Distribution.py          # A
├── Augmentation.py          # B
├── Transformation.py        # C
├── train.py                 # A (model) + B (pipeline)
├── predict.py               # A
├── leaffliction/
│   ├── __init__.py
│   ├── io.py                # A  — image loading, walking, labels
│   ├── augment.py           # B  — the 6 augmentation functions
│   ├── transform.py         # C  — the PlantCV transforms
│   ├── split.py             # A  — stratified train/val split
│   └── viz.py               # A  — matplotlib grid helpers, shared
├── requirements.txt
├── setup.cfg                # flake8 config
├── .gitignore               # MUST exclude images/, model/, *.zip
├── signature.txt
└── README.md
```

### `leaffliction/io.py` (A writes this first, day 1)

```python
def iter_images(root: Path) -> Iterator[Path]: ...
def load_image(path: Path) -> np.ndarray:   # RGB uint8, HxWx3
def label_of(path: Path) -> str:            # parent dir name, e.g. "Apple_healthy"
def class_counts(root: Path) -> dict[str, int]: ...
def save_image(img: np.ndarray, path: Path) -> None: ...
```

Everyone imports from here. Nobody calls `cv2.imread` directly. This kills the classic BGR/RGB bug where the mask works in one file and not another.

### `leaffliction/augment.py` (B)

```python
AUGMENTATIONS = ["Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"]

def apply(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray: ...
def augmented_name(src: Path, kind: str) -> Path:   # "image (1)_Flip.JPG"
```

The subject names the files `{original stem}_{Kind}{ext}`. Match it exactly, the evaluator will `ls`.

### `leaffliction/transform.py` (C)

```python
TRANSFORMS = ["GaussianBlur", "Mask", "RoiObjects", "AnalyzeObject",
              "Pseudolandmarks", "ColorHistogram"]

def apply(img: np.ndarray, kind: str) -> np.ndarray | Figure: ...
```

### Model contract (A + B agree, both code against it)

```
model/
├── model.keras
├── labels.json     # {"0": "Apple_healthy", "1": "Apple_scab", ...}
└── config.json     # {"img_size": 128, "seed": 42, "val_split": 0.2, ...}
```

`predict.py` loads only these three files. B's pipeline writes `labels.json`. A's `predict.py` never imports `train.py`.

### Constants everyone hardcodes identically

`IMG_SIZE = 128` (or 256, decide once), `SEED = 42`, `VAL_SPLIT = 0.2`.

---

## 3. Schedule

Two versions. Pick one at the kickoff.

### Standard track — 10 working days

| Day | A | B | C |
|---|---|---|---|
| 1 | Kickoff, freeze interfaces, `io.py`, repo + flake8 + gitignore | Kickoff, read PlantVillage structure | Kickoff, install PlantCV, run one image end to end |
| 2 | `Distribution.py` (pie + bar, per-plant-type, dir name from argv) | `augment.py`: Flip, Rotate, Crop | Blur + Mask (this is the hard one) |
| 3 | `split.py` stratified, writes `train_files.txt` / `val_files.txt` | Skew, Shear, Distortion + the 7-panel display | ROI objects + Analyze object |
| 4 | Model v1: transfer learning, MobileNetV2 frozen | `Augmentation.py` CLI: single image *and* directory-balance mode | Pseudolandmarks + color histogram (9 channels) |
| 5 | Training loop, checkpointing, saves `model/` | Builds `augmented_directory` from the **train split only** | `-src` / `-dst` / `-h` CLI, batch mode writes to dst |
| 6 | `predict.py`: loads model, shows original + transformed + class | Input pipeline: batching, resize, normalize, `labels.json` | Flake8 clean, docstrings, edge cases (grayscale, corrupt file) |
| 7 | **Integration day, all three.** Merge to `main`, run every program on a 40-image toy dataset | | |
| 8 | Fine-tune: unfreeze top layers, chase >90% | Confusion matrix, per-class accuracy, prove val set ≥ 100 images | Cross-review A and B's code, write the defense brief |
| 9 | Build the zip, compute sha1, write `signature.txt`, freeze it | Final README, requirements pinned | Full dry run on a clean clone |
| 10 | **Cross-teaching session** (section 6). Mock defense. | | |

### Fast track — 5 days

Same order, compressed: day 1 = kickoff + interfaces + start, days 2–3 = build in parallel, day 4 = integrate + train, day 5 = accuracy + zip + cross-teach. Cuts: use transfer learning from the start (no custom CNN), 6 transforms not 8, skip the confusion-matrix visualization and print a table instead.

---

## 4. Independence rules

- One branch per person: `feat/distribution`, `feat/augmentation`, `feat/transformation`. Nobody pushes to `main` alone.
- Nobody edits a file they do not own. If you need a change in someone else's module, open an issue or message them. This is the whole reason for section 2.
- Stub what you do not have. C needs no model. B needs no transforms. A trains on the raw split before B's augmented directory exists, then swaps the path.
- Daily 10-minute sync. Three things each: done, doing, blocked.
- Run `flake8 .` before every commit. The subject makes the norm a grading item.
- Test on 40 images, not 20,000. The subject explicitly says the evaluation checks small datasets. Keep `toy_dataset/` with 10 images per class in `.gitignore` but share the recipe to build it.

---

## 5. Study guide

Split as: **everyone** studies the shared core, then each person goes deep on their block, then you teach each other.

### 5.1 Shared core (all three, before day 1)

**Digital images**
- Raster representation, pixel, `HxWxC`, uint8 vs float32, why you divide by 255
- Color spaces: RGB, HSV (hue/saturation/value), LAB (lightness / green-magenta a / blue-yellow b). Know *why* the LAB b-channel separates a green leaf from a grey background better than raw RGB. This is the single most-asked question on the mask.
- BGR vs RGB, the OpenCV trap
- Histograms: what the x and y axes mean in the subject's Figure IV.7

**The dataset**
- PlantVillage: 256×256 JPEGs, leaf on uniform background, class = subdirectory name
- Class imbalance: why 1640 healthy vs 1000 scab breaks a classifier, what the model learns instead (the majority prior)

**Git & delivery**
- `.gitignore` discipline, sha1 hashing, why the subject bans the dataset in the repo
- flake8 / PEP 8, argparse

### 5.2 Block A — statistics, CNNs, evaluation

**Data analysis (Part 1)**
- Pie chart vs bar chart, when each lies
- matplotlib: `subplots`, `pie`, `bar`, labelling from directory names

**Splitting**
- Stratified split, why random split breaks on imbalanced data
- Train / validation / test, what each is for
- Data leakage: near-duplicates across the split boundary. Be ready to explain the augment-then-split failure in one sentence.

**Neural networks**
- Perceptron, layer, weights, bias, activation (ReLU, softmax)
- Loss: categorical cross-entropy, what the number means
- Backpropagation, gradient descent, SGD vs Adam, learning rate
- Epoch, batch, iteration, batch size effects

**CNNs**
- Convolution as a sliding kernel, feature maps
- Kernel size, stride, padding, receptive field
- Pooling (max, average) and why it gives translation tolerance
- Batch normalization, dropout
- The classic stack: conv → BN → ReLU → pool, repeated, then dense → softmax
- What early layers learn (edges) vs late layers (parts, textures)

**Transfer learning** (your fastest route past 90%)
- ImageNet pretraining, feature extraction vs fine-tuning
- Freezing layers, then unfreezing the top block at a 10× lower learning rate
- MobileNetV2 / EfficientNetB0 / ResNet50, the tradeoff on 42's hardware
- Know both: the evaluator may ask you to justify not writing a CNN from scratch. "Faster, less data, better features" is the answer. Also be able to describe the from-scratch architecture you *would* write.

**Evaluation**
- Accuracy, precision, recall, F1, and when accuracy misleads
- Confusion matrix, reading it per class
- Overfitting vs underfitting, the two loss curves and their shapes
- Early stopping, model checkpointing
- Why the subject demands ≥100 validation images (statistical noise on small sets)

### 5.3 Block B — augmentation

**Theory**
- What augmentation buys: invariance to nuisance factors, regularization, an artificially larger training set
- Which invariances are real for leaves (rotation, flip, scale, illumination) and which are not (extreme color shifts destroy disease signal, since brown lesions *are* the label)
- Offline augmentation (write files to disk, what this subject wants) vs online augmentation (transform per batch)
- Balancing strategies: oversample minority, undersample majority, class weights. Know all three, explain why the subject picked oversampling by augmentation.

**Geometry**
- Affine transform: the 2×3 matrix, what translation/rotation/scale/shear each look like
- **Shear vs skew**: shear slides one axis and keeps parallel lines parallel (affine); skew as this subject uses it is a projective/perspective warp where parallel lines converge. Have the distinction ready, evaluators ask.
- Projective transform / homography, the 3×3 matrix
- Barrel and pincushion distortion
- Interpolation: nearest, bilinear, bicubic. Border handling: constant, reflect, replicate.

**Photometric**
- Brightness, contrast, gamma
- Gaussian noise, blur as augmentation

**Tools**: `albumentations` (fastest), or `imgaug`, or plain OpenCV `warpAffine` / `warpPerspective`. Know how to do at least one by hand with a matrix, because "the library did it" scores badly.

### 5.4 Block C — classical computer vision & PlantCV

**Filtering**
- Convolution kernel, separable filters
- Gaussian blur: sigma, kernel size, why you blur before thresholding (noise suppression)
- Median filter vs Gaussian
- Edge detection: Sobel, Canny (know the concept even if unused)

**Segmentation**
- Grayscale conversion, and channel selection (`pcv.rgb2gray_lab` with `channel='b'`)
- Thresholding: global, Otsu (the between-class variance idea), triangle, adaptive
- Binary masks, `object_type='light'` vs `'dark'`

**Morphology**
- Erosion, dilation, opening, closing, structuring element
- Hole filling, small-object removal (`pcv.fill`)

**Objects & shape**
- Contours, connected components
- ROI (region of interest), the `roi_type` options: `partial`, `cutto`, `largest`
- Shape descriptors: area, perimeter, convex hull, solidity, ellipse fit, centroid, major/minor axis. Know what `pcv.analyze.size` draws and why.

**Pseudolandmarks**
- Homology points in plant phenotyping, why they exist (comparing shape across specimens without manual annotation)
- `x_axis_pseudolandmarks` returns three groups: top, bottom, center-vertical. Explain the colored dot groups in Figure IV.6.

**Color analysis**
- `pcv.analyze.color`, the 9 channels in Figure IV.7: blue, blue-yellow, green, green-magenta, hue, lightness, red, saturation, value
- Why the histogram is computed inside the mask, not on the whole frame
- Why proportion-of-pixels (%) rather than raw counts

**PlantCV mechanics**
- `pcv.params.debug`, `pcv.outputs`
- **Version pain**: v3 and v4 renamed and moved half the API (`pcv.analyze_object` → `pcv.analyze.size`, `pcv.threshold.binary` dropped `max_value`). Pin the version in `requirements.txt` on day 1 and all three install the same one.
- Fallback: everything here is doable in pure OpenCV if PlantCV misbehaves. Have the plan B.

---

## 6. Cross-teaching and defense prep

The peer evaluator picks who answers. Assume they pick the person who did not write that file.

### Deliverable per person: a 1-page brief

Each owner writes `docs/brief_<part>.md` by day 8, containing:

1. What the program does, in three sentences
2. Every algorithm used, named, with a one-line explanation
3. The three design decisions taken and the alternatives rejected
4. Three questions an evaluator would ask, with the answers
5. Known limitations

### Cross-teaching session, day 10 (2 hours)

- 20 minutes per part: the owner presents, the other two ask questions. No slides, run the code live.
- Then rotate: **A explains Part 3, B explains Part 4, C explains Part 1 and 2.** If someone stalls, the owner reteaches that piece on the spot.
- Finish with a full mock defense: one of you plays the evaluator, works through the subject page by page.

### The question list to rehearse

Every person answers all of these out loud:

- Why is the dataset unbalanced and what does that do to a classifier?
- Show me the split. Did you augment before or after? Prove it.
- How many images in the validation set? Show the count in code, not a claim.
- Why LAB and not RGB for the mask?
- What is the difference between shear and skew?
- Draw the CNN architecture on the whiteboard. What does each layer do?
- What is a convolution, in one sentence, without the word "convolution"?
- Why did you freeze the base model, then unfreeze it?
- Your accuracy is 96%. How do I know you are not overfitting? Show the curves.
- What are the pseudolandmarks for?
- Run it on this image I picked. Now run it on this directory.
- Delete `model/`, retrain from the repo. Does it work?

---

## 7. Delivery checklist

Run this on a clean clone the day before the defense.

- [ ] `git clone` fresh, `pip install -r requirements.txt`, everything runs
- [ ] `flake8 .` returns nothing
- [ ] `./Distribution.py ./Apple` prints pie + bar, colonnes named from the directory
- [ ] Works on **every** plant directory in the dataset, not just Apple
- [ ] `./Augmentation.py "./Apple/Apple_healthy/image (1).JPG"` displays 6 augmentations and writes 6 files with the exact naming pattern (test with the spaces and parentheses in the filename)
- [ ] `./Transformation.py "<image>"` displays the set
- [ ] `./Transformation.py -src X -dst Y -mask` writes to Y
- [ ] `./Transformation.py -h` prints usable help
- [ ] `./train.py ./Apple/` trains and produces the zip
- [ ] `./predict.py "<image>"` shows original + transformed + predicted class
- [ ] Validation accuracy > 90%, validation set ≥ 100 images, both provable in code
- [ ] No dataset, no model weights, no zip in the git repo
- [ ] `signature.txt` contains the sha1 of the **final, frozen** zip. Compute it last. Do not rebuild the zip afterwards, the hash changes and the grade goes to 0.
- [ ] Every program handles a corrupt or non-image file without a traceback
- [ ] All three can run and explain all four parts

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| PlantCV version mismatch breaks C's code on A's machine | Pin the version day 1, share `requirements.txt`, all install the same |
| Accuracy stuck below 90% | Transfer learning from day 4, not day 8. Leaves are easy for ImageNet features. |
| Zip rebuilt after hashing | One person (A) owns the zip. Nobody else touches it. Hash last. |
| Merge conflicts in `train.py` | A owns the model half, B owns the pipeline half, split into `train.py` + `leaffliction/pipeline.py` if it gets contentious |
| Someone finishes early | They start their brief and cross-review, not new features |
| 20,000 images make every test slow | Build `toy_dataset/` on day 1 and develop against it |
