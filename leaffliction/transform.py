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

# Figure IV.4 draws the region of interest as a border around the frame.
ROI_MARGIN = 3
ROI_COLOR = (0, 0, 255)
LEAF_COLOR = (0, 255, 0)

# Figure IV.6 shows the three pseudolandmark groups in three colors.
LANDMARK_COLORS = {
    "top": (0, 0, 255),
    "bottom": (255, 0, 255),
    "center": (255, 90, 0),
}
LANDMARK_RADIUS = 4

# Figure IV.7 legend, in the subject's own order and colors.
HISTOGRAM_COLORS = {
    "blue": "#0000ff",
    "blue-yellow": "#f5e642",
    "green": "#007f0e",
    "green-magenta": "#ff00ff",
    "hue": "#7f3fbf",
    "lightness": "#555555",
    "red": "#ff0000",
    "saturation": "#3fd6d6",
    "value": "#ff9900",
}


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


def roi_objects(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Draw the region of interest and the leaf parts kept inside it.

    The ROI is the whole frame here because the data set is one
    centered leaf per photo. It still earns its place: roi_type
    "partial" keeps any blob that merely touches the region, so a leaf
    running off the edge of the frame survives instead of being dropped
    for not being fully contained.
    """
    if mask is None:
        mask = leaf_mask(rgb)
    height, width = mask.shape
    roi = pcv.roi.rectangle(img=_to_bgr(rgb), x=ROI_MARGIN, y=ROI_MARGIN,
                            h=height - 2 * ROI_MARGIN,
                            w=width - 2 * ROI_MARGIN)
    kept = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")

    out = rgb.copy()
    out[kept > 0] = LEAF_COLOR
    cv2.rectangle(out, (ROI_MARGIN, ROI_MARGIN),
                  (width - ROI_MARGIN - 1, height - ROI_MARGIN - 1),
                  ROI_COLOR, 3)
    return out


def analyze_object(rgb: np.ndarray,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Draw the shape analysis: contour, convex hull, axes and centroid.

    The measurements themselves (area, perimeter, solidity, width,
    height) land in pcv.outputs and are read back by shape_measures.
    """
    if mask is None:
        mask = leaf_mask(rgb)
    # analyze.size stamps the sample label across the centroid; the
    # figure in the subject has no such text, and text_size drives the
    # cv2.putText font scale, so zero removes it.
    previous = pcv.params.text_size
    pcv.params.text_size = 0
    try:
        drawn = pcv.analyze.size(img=_to_bgr(rgb), labeled_mask=mask,
                                 n_labels=1)
    finally:
        pcv.params.text_size = previous
    return _to_rgb(drawn)


def shape_measures(rgb: np.ndarray,
                   mask: np.ndarray | None = None) -> dict[str, float]:
    """Return the numbers behind analyze_object."""
    if mask is None:
        mask = leaf_mask(rgb)
    pcv.outputs.clear()
    pcv.analyze.size(img=_to_bgr(rgb), labeled_mask=mask, n_labels=1)
    observations = pcv.outputs.observations.get("default_1", {})
    wanted = ("area", "perimeter", "solidity", "width", "height",
              "convex_hull_area", "longest_path")
    return {key: observations[key]["value"]
            for key in wanted if key in observations}


def pseudolandmarks(rgb: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Mark 20 homology points along the top, bottom and centre lines.

    PlantCV walks the contour and cuts it into 20 equal steps along the
    x axis, then takes the topmost, bottommost and central point of the
    leaf at each step. The result is a fixed-length description of the
    shape that needs no manual annotation, so two leaves photographed
    at different sizes can still be compared point by point.
    """
    if mask is None:
        mask = leaf_mask(rgb)
    top, bottom, center = pcv.homology.x_axis_pseudolandmarks(
        img=_to_bgr(rgb), mask=mask)

    out = rgb.copy()
    for group, points in (("top", top), ("bottom", bottom),
                          ("center", center)):
        color = LANDMARK_COLORS[group]
        for point in np.asarray(points).reshape(-1, 2):
            cv2.circle(out, (int(point[0]), int(point[1])),
                       LANDMARK_RADIUS, color, -1)
    return out


def color_histogram(rgb: np.ndarray,
                    mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return the nine channel histograms of Figure IV.7.

    Only pixels inside the mask are counted: including the background
    would add a tall spike at whatever grey the bench happens to be,
    which says nothing about the leaf and swamps the real signal.

    Bins hold the proportion of leaf pixels rather than raw counts, so
    a big leaf and a small one can be laid over each other.
    """
    if mask is None:
        mask = leaf_mask(rgb)
    bgr = _to_bgr(rgb)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    planes = {
        "blue": bgr[:, :, 0],
        "green": bgr[:, :, 1],
        "red": bgr[:, :, 2],
        "lightness": lab[:, :, 0],
        "green-magenta": lab[:, :, 1],
        "blue-yellow": lab[:, :, 2],
        "hue": hsv[:, :, 0],
        "saturation": hsv[:, :, 1],
        "value": hsv[:, :, 2],
    }
    inside = mask > 0
    total = int(inside.sum())
    if total == 0:
        return {name: np.zeros(256) for name in planes}
    return {
        name: np.bincount(plane[inside], minlength=256)[:256] * 100.0 / total
        for name, plane in planes.items()
    }
