"""Shared library code for the Leaffliction project.

Package ownership (see ../leaffliction_team_plan.md section 2):
  io.py        - A - Elena - image loading/saving/walking, the only file
                 allowed to call cv2.imread/cv2.imwrite directly.
  split.py     - A - Elena - stratified train/val split.
  viz.py       - A - Elena - shared matplotlib helpers.
  package.py   - A - Elena - build the submission zip + signature.txt.
  metrics.py   - A - Elena - confusion matrix / per-class accuracy, used by
                 train.py after training to report the validation results.
  augment.py   - B - Tanya - the 6 augmentation functions.
  transform.py - C - Misha - the PlantCV transforms.
"""

IMG_SIZE = 128
SEED = 42
VAL_SPLIT = 0.2
