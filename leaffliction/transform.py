"""PlantCV leaf transformations.

PlantCV takes its images in BGR order even though its parameters are
named rgb_img: rgb2gray_lab calls cv2.cvtColor(img, COLOR_BGR2LAB)
internally. The shared contract in io.py is RGB, so BGR stays private
to this module and every public function takes and returns RGB.
"""

from __future__ import annotations

import cv2
import numpy as np
from plantcv import plantcv as pcv

TRANSFORMS = [
    "GaussianBlur",
    "Mask",
    "RoiObjects",
    "AnalyzeObject",
    "Pseudolandmarks",
    "ColorHistogram",
]

BLUR_KSIZE = (5, 5)
FILL_SIZE = 200
MIN_BLOB_AREA = 200


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert the shared RGB contract into what PlantCV expects."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a PlantCV result back to the shared RGB contract."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _channel_binary(bgr: np.ndarray, channel: str, object_type: str):
    """Blur one LAB channel and split it with Otsu.

    The blur runs before the threshold so that JPEG noise and leaf
    texture do not each become their own speck in the binary image.
    Otsu picks the cut from the image itself, so exposure can vary
    between photos without breaking the segmentation.
    """
    gray = pcv.rgb2gray_lab(rgb_img=bgr, channel=channel)
    blurred = pcv.gaussian_blur(img=gray, ksize=BLUR_KSIZE)
    return pcv.threshold.otsu(gray_img=blurred, object_type=object_type)


def gaussian_blur_view(rgb: np.ndarray) -> np.ndarray:
    """Return the blurred LAB a channel, the input to the threshold."""
    gray = pcv.rgb2gray_lab(rgb_img=_to_bgr(rgb), channel="a")
    return pcv.gaussian_blur(img=gray, ksize=BLUR_KSIZE)


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected component.

    A photo holds one leaf, so anything detached from the main blob is
    a shadow edge or a speck of background that survived thresholding.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    biggest = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return ((labels == biggest) * 255).astype(np.uint8)


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a binary leaf mask (uint8, 0 or 255).

    Two LAB channels are thresholded and OR-ed together because neither
    covers the whole data set on its own:

    - a (green-magenta) separates chlorophyll from any neutral or brown
      background, but weakens once a leaf is mostly lesion.
    - b (blue-yellow) is strong on yellowish leaves and near-useless on
      dark blue-green ones, where leaf and background differ by about
      two levels and Otsu has nothing to cut.

    Then fill drops small specks of background wrongly marked as leaf,
    and fill_holes closes gaps enclosed by the leaf. fill_holes is what
    makes diseased leaves work: a black rot lesion is dark brown, lands
    on the background side of the threshold and gets punched out of the
    mask, leaving 22 holes on Grape_Black_rot. The lesion is part of the
    leaf and must stay in, otherwise the color histogram measures the
    leaf without its disease.

    Measured over 320 images, 40 per class: a alone 305, b alone 295,
    a|b 312, a|b with the largest blob kept 320.
    """
    bgr = _to_bgr(rgb)
    green = _channel_binary(bgr, "a", "dark")
    yellow = _channel_binary(bgr, "b", "light")
    joined = pcv.logical_or(bin_img1=green, bin_img2=yellow)
    filled = pcv.fill(bin_img=joined, size=FILL_SIZE)
    return _largest_blob(pcv.fill_holes(bin_img=filled))


def masked_leaf(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    background: str = "white",
) -> np.ndarray:
    """Return the leaf with its background replaced."""
    if mask is None:
        mask = leaf_mask(rgb)
    out = pcv.apply_mask(img=_to_bgr(rgb), mask=mask, mask_color=background)
    return _to_rgb(out)
