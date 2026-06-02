"""Automatic pitch landmark detection for homography-based coordinate mapping.

Algorithm:
1. Extract white line pixels from the image using an HSV mask.
2. Run Shi-Tomasi corner detection on the white-line mask — corners naturally land
   at T-junctions and L-junctions (box corners, goal-line intersections etc.).
3. Filter corners to those that lie on the green pitch area.
4. RANSAC: repeatedly sample 4 candidate pixels, randomly assign them to 4 of the
   known standard landmarks, compute the 4-point homography, then score by how many
   remaining candidate pixels map close to any known standard coordinate.
   The assignment with the most inliers wins.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Known standard coordinates (Statsbomb / Opta: x = 0-120, y = 0-80)
# for the attacking goal (x = 120 = goal line being shot at).
# ---------------------------------------------------------------------------
PITCH_LANDMARKS: list[tuple[float, float]] = [
    (102.0, 18.0),   # 18-yd box front corner, near side
    (102.0, 63.0),   # 18-yd box front corner, far side
    (114.0, 30.0),   # 6-yd box front corner, near side
    (114.0, 50.0),   # 6-yd box front corner, far side
    (120.0, 18.0),   # Goal-line × 18-yd box near side
    (120.0, 30.0),   # Goal-line × 6-yd box near side
    (120.0, 36.0),   # Near goal post (ground level)
    (120.0, 44.0),   # Far goal post (ground level)
    (120.0, 50.0),   # Goal-line × 6-yd box far side
    (120.0, 63.0),   # Goal-line × 18-yd box far side
]


def detect_landmarks(
    frame: np.ndarray,
    *,
    min_landmarks: int = 4,
    n_iterations: int = 3000,
    reproj_thresh: float = 4.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Detect pitch landmarks in a still image and match them to standard coords.

    Args:
        frame: BGR image (e.g. from cv2.imread).
        min_landmarks: Minimum matched pairs to return a result.
        n_iterations: RANSAC iterations.
        reproj_thresh: Maximum reprojection error in standard-coord units for an
                       inlier (standard pitch is 120×80, so 4.0 ≈ 4 yards).

    Returns:
        List of ((pixel_x, pixel_y), (standard_x, standard_y)) matched pairs,
        or empty list if fewer than *min_landmarks* could be matched.
    """
    pitch_mask = _green_pitch_mask(frame)
    white_mask = _white_line_mask(frame)

    # Keep only white pixels that are inside (or very near) the pitch area
    dilated_pitch = cv2.dilate(pitch_mask, np.ones((15, 15), np.uint8))
    white_on_pitch = cv2.bitwise_and(white_mask, dilated_pitch)

    candidates = _corner_candidates(white_on_pitch)
    if len(candidates) < min_landmarks:
        return []

    return _ransac_match(
        candidates,
        PITCH_LANDMARKS,
        n_iterations=n_iterations,
        reproj_thresh=reproj_thresh,
        min_inliers=min_landmarks,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _green_pitch_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask for green grass pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([25, 25, 25], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _white_line_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask for white pitch-marking pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White: very low saturation, high brightness
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 160], dtype=np.uint8),
        np.array([180, 50, 255], dtype=np.uint8),
    )
    # Remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _corner_candidates(
    white_mask: np.ndarray,
    max_corners: int = 60,
    quality: float = 0.05,
    min_dist: int = 20,
) -> list[tuple[float, float]]:
    """Use Shi-Tomasi to find corner/junction candidates on the white line mask."""
    corners = cv2.goodFeaturesToTrack(
        white_mask,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=7,
    )
    if corners is None:
        return []
    return [(float(c[0][0]), float(c[0][1])) for c in corners]


def _ransac_match(
    candidates: list[tuple[float, float]],
    standard_pts: list[tuple[float, float]],
    *,
    n_iterations: int,
    reproj_thresh: float,
    min_inliers: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """RANSAC: find best pixel ↔ standard correspondences.

    Each iteration samples 4 random candidate pixels and 4 random standard
    landmarks, computes the exact 4-point homography, then counts how many
    remaining candidates map within *reproj_thresh* of any known landmark.
    The trial producing the most inliers is returned.
    """
    if len(candidates) < 4 or len(standard_pts) < 4:
        return []

    cands = np.array(candidates, dtype=np.float32)
    stds = np.array(standard_pts, dtype=np.float32)
    rng = np.random.default_rng(42)

    n_c = len(cands)
    n_s = len(stds)
    best_inliers: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for _ in range(n_iterations):
        ci = rng.choice(n_c, 4, replace=False)
        si = rng.choice(n_s, 4, replace=False)

        src = cands[ci]
        dst = stds[si]

        # We need the 4 sampled pixels to be non-collinear for a valid homography
        try:
            H, _ = cv2.findHomography(src, dst)
        except cv2.error:
            continue
        if H is None:
            continue

        # Map ALL candidates through H and find inliers
        try:
            mapped = cv2.perspectiveTransform(
                cands.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
        except cv2.error:
            continue

        # Assign each mapped point to its nearest standard landmark
        inliers: list[tuple[tuple[float, float], tuple[float, float]]] = []
        used_std: set[int] = set()
        for k in range(n_c):
            mx, my = mapped[k]
            dists = np.sqrt((stds[:, 0] - mx) ** 2 + (stds[:, 1] - my) ** 2)
            j = int(np.argmin(dists))
            if dists[j] < reproj_thresh and j not in used_std:
                used_std.add(j)
                inliers.append(
                    (
                        (float(cands[k, 0]), float(cands[k, 1])),
                        (float(stds[j, 0]), float(stds[j, 1])),
                    )
                )

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < min_inliers:
        return []
    return best_inliers
