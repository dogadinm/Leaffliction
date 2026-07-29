"""Shared library code for the Leaffliction project.

io.py is the only module allowed to call cv2.imread/cv2.imwrite, which
is what keeps the BGR/RGB byte order decided in one place.
"""

IMG_SIZE = 128
SEED = 42
VAL_SPLIT = 0.2
