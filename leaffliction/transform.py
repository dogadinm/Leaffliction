"""PlantCV leaf transformations.

Careful: PlantCV wants BGR despite naming its parameter rgb_img —
rgb2gray_lab calls cv2.cvtColor(img, COLOR_BGR2LAB) internally. Since
io.py hands out RGB, BGR stays private to this module and every public
function here takes and returns RGB.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from plantcv import plantcv as pcv

from leaffliction import viz

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
    """RGB in, BGR out — the byte order PlantCV expects."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR in, RGB out — back to what the rest of the project uses."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _channel_binary(bgr: np.ndarray, channel: str, object_type: str):
    """Blur one LAB channel and split it with Otsu.

    Blur first, or JPEG noise and leaf texture each become their own
    speck in the binary. Otsu reads the cut off the image itself, so
    exposure can vary between photos.
    """
    gray = pcv.rgb2gray_lab(rgb_img=bgr, channel=channel)
    blurred = pcv.gaussian_blur(img=gray, ksize=BLUR_KSIZE)
    return pcv.threshold.otsu(gray_img=blurred, object_type=object_type)


def gaussian_blur_view(rgb: np.ndarray) -> np.ndarray:
    """Return the blurred LAB a channel, the input to the threshold."""
    gray = pcv.rgb2gray_lab(rgb_img=_to_bgr(rgb), channel="a")
    return pcv.gaussian_blur(img=gray, ksize=BLUR_KSIZE)


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep the biggest connected component.

    One photo, one leaf: anything detached from it is a shadow edge or
    a speck of background that survived the threshold.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    biggest = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return ((labels == biggest) * 255).astype(np.uint8)


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a binary leaf mask (uint8, 0 or 255).

    Neither LAB channel covers the whole data set alone, so both are
    thresholded and OR-ed: a (green-magenta) keys on chlorophyll and
    holds against any background, but fades once a leaf is mostly
    lesion; b (blue-yellow) is strong on yellowish leaves and useless
    on dark blue-green ones, where it barely separates leaf from bench.
    """
    bgr = _to_bgr(rgb)
    green = _channel_binary(bgr, "a", "dark")
    yellow = _channel_binary(bgr, "b", "light")
    joined = pcv.logical_or(bin_img1=green, bin_img2=yellow)
    # fill drops specks of background marked as leaf; fill_holes does
    # the reverse and is what makes diseased leaves work — a dark rot
    # lesion falls on the background side of the threshold and gets
    # punched out, but it is part of the leaf and the histogram needs it.
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
    """Draw the region of interest and the leaf kept inside it.

    The ROI is the whole frame — one centred leaf per photo leaves
    nothing to select. roi_type "partial" still matters: it keeps a
    blob that merely touches the region, so a leaf running off the edge
    survives instead of being dropped for not fitting.
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
    """Draw contour, convex hull, axes and centroid.

    The numbers behind the drawing come back out via shape_measures.
    """
    if mask is None:
        mask = leaf_mask(rgb)
    # analyze.size stamps the sample label across the centroid, which
    # the subject's figure does not have. text_size is the putText font
    # scale, so zero hides it; restore it, PlantCV params are global.
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

    PlantCV cuts the contour into 20 equal steps along the x axis and
    takes the topmost, bottommost and central point at each. Fixed
    length is the whole point: two leaves of different size stay
    comparable point by point, with no manual annotation.
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
            try:
                x, y = int(point[0]), int(point[1])
            except (TypeError, ValueError):
                # PlantCV yields the string "NA" instead of coordinates
                # when the object is too small to walk a contour around.
                continue
            cv2.circle(out, (x, y), LANDMARK_RADIUS, color, -1)
    return out


def color_histogram(rgb: np.ndarray,
                    mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return the nine channel histograms of Figure IV.7.

    Masked pixels only, otherwise the tallest peak is the grey of the
    bench. Bins hold percent of leaf pixels rather than raw counts, so
    a big leaf and a small one overlay.
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


def color_histogram_image(rgb: np.ndarray,
                          mask: np.ndarray | None = None) -> np.ndarray:
    """Render the colour histogram as an image.

    Rasterised so that apply() stays single-typed and the batch writer
    needs no special case for the one panel that is a plot.
    """
    histograms = color_histogram(rgb, mask)
    figure = viz.build_color_histogram(histograms, HISTOGRAM_COLORS)
    return viz.figure_to_array(figure)


_DISPATCH = {
    "GaussianBlur": lambda rgb, mask: gaussian_blur_view(rgb),
    "Mask": masked_leaf,
    "RoiObjects": roi_objects,
    "AnalyzeObject": analyze_object,
    "Pseudolandmarks": pseudolandmarks,
    "ColorHistogram": color_histogram_image,
}


def apply(rgb: np.ndarray, kind: str,
          mask: np.ndarray | None = None) -> np.ndarray:
    """Run one named transformation, always returning an RGB image.

    Pass mask when several transforms share a photo, or each call
    segments the leaf again.
    """
    if kind not in _DISPATCH:
        raise ValueError(
            f"unknown transformation {kind!r}, expected one of "
            f"{', '.join(TRANSFORMS)}"
        )
    if mask is None and kind != "GaussianBlur":
        mask = leaf_mask(rgb)
    return _DISPATCH[kind](rgb, mask)


def apply_all(rgb: np.ndarray,
              kinds: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Run several transformations, segmenting the leaf only once."""
    kinds = list(TRANSFORMS) if kinds is None else list(kinds)
    mask = leaf_mask(rgb)
    return {kind: apply(rgb, kind, mask) for kind in kinds}
