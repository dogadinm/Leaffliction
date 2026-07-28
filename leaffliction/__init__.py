"""Shared library code for the Leaffliction project.

Package ownership (see ../leaffliction_team_plan.md section 2):
  io.py        - Person A - image loading/saving/walking, the only file
                 allowed to call cv2.imread/cv2.imwrite directly.
  split.py     - Person A - stratified train/val split.
  viz.py       - Person A - shared matplotlib helpers.
  package.py   - Person A - build the submission zip + signature.txt.
  augment.py   - Person B - the 6 augmentation functions.
  transform.py - Person C - the PlantCV transforms.
"""

IMG_SIZE = 128
SEED = 42
VAL_SPLIT = 0.2
