# Leaffliction

Image classification by disease recognition on leaves.

## Install

The stack needs **Python 3.12**. Newer versions cannot build scipy from
source, which plantcv depends on.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Activate the environment before running anything:

```bash
source .venv/bin/activate
```

Without it the shebang picks the system interpreter, which has no
plantcv, and every program stops at `ModuleNotFoundError`.

## Data set

Class directories hold the images, one directory per variety and
disease:

```
images/
├── Apple_healthy/
├── Apple_scab/
├── Grape_Black_rot/
└── ...
```

The data set is not in this repository, as the subject requires.

## Transformation.py

Extract leaf features from one photo or a whole directory.

```bash
./Transformation.py "images/Apple_scab/image (1).JPG"
./Transformation.py -src images/Apple_scab -dst out
./Transformation.py -src images -dst out -mask -roi
./Transformation.py -h
```

Given an image it opens the transformations on screen, the histogram in
its own window. Given `-src` and `-dst` it walks the directory and
writes the results, mirroring the class directories into `-dst` and
naming files `{original}_{Transformation}{ext}`.

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

## Augmentation.py

Apply six offline augmentations to one image, or balance a whole data
set from the training split.

```bash
./Augmentation.py "images/Apple_scab/image (1).JPG"
./Augmentation.py images --train-list train_files.txt --output augmented_directory
./Augmentation.py -h
```

## Development

```bash
flake8 .
```

`setup.cfg` holds the configuration; `max-line-length` is 88.
