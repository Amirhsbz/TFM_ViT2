import os
import shutil
import itertools
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MarkerRegionProp:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def refresh_dir(dirname:str):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)

def find_marker(frame,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        mask_range=(150, 255), min_value:int=70,
        morphop_iter=1, morphclose_iter=2, dilate_iter=1):
    """find markers in the tactile iamge

    Args:
        frame (np.ndarray): input image (can be RGB or grayscale)
        morphop_kernel (cv.MatLike, optional): kernel of MORPH_OPEN. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        morphclose_kernel (cv.MatLike, optional): kernel of MORPH_CLOSE. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        dilate_kernel (cv.MatLike, optional): kernel of dilation. Defaults to cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)).
        mask_range (tuple, optional): range of mask segementation. Defaults to (150, 255).
        min_value (int, optional): minimum value to segement marker from HSV (V-chan). Defaults to 70.
        morphop_iter (int, optional): iteration of MORPH_OPEN operation. Defaults to 1.
        morphclose_iter (int, optional): iteration of MORPH_CLOSE operation. Defaults to 2.
        dilate_iter (int, optional): iteration of DILATION operation. Defaults to 1.

    Returns:
        np.ndarray: final mask (0, 255) np.unint8
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) ### use only the green channel
    value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[...,-1]
    # img_sblur = cv2.GaussianBlur(gray,(3,3),5)
    img_lblur = cv2.GaussianBlur(gray, (15,15),5)
    im_blur_sub = img_lblur - gray + 128
    blur_mask = np.logical_and(im_blur_sub >= mask_range[0], im_blur_sub <= mask_range[1])
    value_mask = value < min_value
    mask = np.logical_or(blur_mask, value_mask)
    mask255 = np.array(255 * mask,dtype=np.uint8)
    mask255_op = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, morphop_kernel, iterations=morphop_iter)
    if dilate_iter > 0:
        dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    else:
        dilate_mask = mask255_op
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=morphclose_iter)
    return morph_close

def find_marker_props(marker_img:np.ndarray):
    binary_img = (marker_img > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_img, connectivity=8
    )
    props = []
    for label in range(1, num_labels):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        props.append(
            MarkerRegionProp(
                label=label,
                area=int(stats[label, cv2.CC_STAT_AREA]),
                bbox=(top, left, top + height, left + width),
                centroid=(float(centroids[label][1]), float(centroids[label][0])),
            )
        )
    return props

def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Small dependency-free 1D k-means used to regularize marker grids."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return np.zeros(k, dtype=np.float32)
    if len(values) < k:
        lo, hi = float(values.min()), float(values.max())
        return np.linspace(lo, hi, k, dtype=np.float32)

    centers = np.percentile(values, np.linspace(0, 100, k)).astype(np.float32)
    for _ in range(max_iter):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            cluster = values[labels == idx]
            if len(cluster) > 0:
                new_centers[idx] = float(np.median(cluster))
        if np.allclose(new_centers, centers, atol=0.1):
            centers = new_centers
            break
        centers = new_centers
    centers.sort()
    return centers


def _filtered_marker_centers_from_props(
    props: list[MarkerRegionProp],
    *,
    min_area_ratio: float = 0.25,
    max_area_ratio: float = 4.0,
    max_aspect_ratio: float = 3.0,
) -> list[tuple[float, float]]:
    if not props:
        return []

    areas = np.array([prop.area for prop in props], dtype=np.float32)
    median_area = float(np.median(areas))
    if median_area <= 0:
        return []

    filtered = []
    for prop in props:
        top, left, bottom, right = prop.bbox
        height = max(1, bottom - top)
        width = max(1, right - left)
        aspect = max(width / height, height / width)
        if prop.area < median_area * min_area_ratio:
            continue
        if prop.area > median_area * max_area_ratio:
            continue
        if aspect > max_aspect_ratio:
            continue
        # MarkerRegionProp stores centroid as (y, x). Return centers as (x, y).
        filtered.append((float(prop.centroid[1]), float(prop.centroid[0])))
    return filtered


def _apply_broken_cells_fill(
    grid: np.ndarray,
    broken_cells: set[tuple[int, int]],
    row_centers: np.ndarray,
) -> np.ndarray:
    """Re-fill specific grid cells by interpolation, ignoring their current values."""
    grid = grid.copy()
    rows, cols = grid.shape[:2]
    valid = np.ones((rows, cols), dtype=bool)
    for r, c in broken_cells:
        if 0 <= r < rows and 0 <= c < cols:
            valid[r, c] = False
    col_centers = np.median(grid[:, :, 0], axis=0)
    for row in range(rows):
        if np.count_nonzero(valid[row]) >= 2:
            cols_valid = np.flatnonzero(valid[row]).astype(np.float32)
            x_valid = grid[row, valid[row], 0]
            y_valid = grid[row, valid[row], 1]
            x_fit = np.polyfit(cols_valid, x_valid, deg=1)
            y_fill = float(np.median(y_valid))
            for col in range(cols):
                if not valid[row, col]:
                    grid[row, col] = (float(np.polyval(x_fit, col)), y_fill)
        else:
            for col in range(cols):
                if not valid[row, col]:
                    grid[row, col] = (col_centers[col], row_centers[row])
    return grid


def _regularize_marker_grid(
    centers: list[tuple[float, float]],
    grid_shape: tuple[int, int],
    image_shape: tuple[int, ...] | None = None,
    broken_cells: set[tuple[int, int]] | None = None,
    n_missing_right_cols: int = 0,
) -> list[list[float]]:
    """Return a stable row-major marker grid, filling missing cells if needed.

    The GelSight marker layout used in this project is a mostly rectified grid.
    Connected-component detection can still miss dim edge markers or produce
    tiny edge fragments. This function assigns detected centers to a known
    rows x cols lattice, removes duplicate assignments, and fills empty cells
    from the inferred row/column centers. The filled points are only used as
    initial LK tracking points; later validation still rejects bad tracks.

    broken_cells: set of (row, col) 0-indexed positions that are physically
        broken and should always be filled by interpolation regardless of
        what the blob detector finds there.

    n_missing_right_cols: number of rightmost grid columns that are physically
        absent. Detection is restricted to the remaining left columns, and the
        missing right columns are extrapolated from the detected spacing.
    """
    rows, cols = grid_shape
    expected = rows * cols
    if rows <= 0 or cols <= 0 or not centers:
        return [[float(x), float(y)] for x, y in centers]

    pts = np.asarray(centers, dtype=np.float32)
    # Do not hallucinate a grid when detection is badly broken.
    if len(pts) < max(4, int(expected * 0.6)):
        order = np.lexsort((pts[:, 0], pts[:, 1]))
        return pts[order].astype(float).tolist()

    row_centers = _kmeans_1d(pts[:, 1], rows)
    row_labels = np.argmin(np.abs(pts[:, 1:2] - row_centers[None, :]), axis=1)

    # Columns actually expected from detection; right-side may be physically absent.
    effective_cols = cols - max(0, n_missing_right_cols)

    row_points: list[np.ndarray] = []
    for row in range(rows):
        row_pts = pts[row_labels == row]
        row_points.append(_select_regular_row_points(row_pts, effective_cols))

    _broken_gripper_active = bool(broken_cells) or n_missing_right_cols > 0

    # Missing-edge-column heuristic only applies when not forcing right-col fill.
    if n_missing_right_cols <= 0:
        missing_edge_grid = _regularize_grid_with_missing_edge_column(
            row_points, row_centers, rows, cols, image_shape=image_shape
        )
        if missing_edge_grid is not None:
            if broken_cells:
                missing_edge_grid = _apply_broken_cells_fill(
                    missing_edge_grid, broken_cells, row_centers
                )
            if _broken_gripper_active and image_shape is not None:
                _h, _w = image_shape[:2]
                missing_edge_grid[:, :, 0] = np.clip(missing_edge_grid[:, :, 0], 0, _w - 1)
                missing_edge_grid[:, :, 1] = np.clip(missing_edge_grid[:, :, 1], 0, _h - 1)
            return missing_edge_grid.reshape(expected, 2).astype(float).tolist()

    complete_rows = [row_pts for row_pts in row_points if len(row_pts) == effective_cols]
    if complete_rows:
        template = np.median(np.stack(complete_rows, axis=0), axis=0)
        col_centers_eff = template[:, 0]
    else:
        all_x = (
            np.concatenate([r[:, 0] for r in row_points if len(r) > 0])
            if any(len(r) > 0 for r in row_points)
            else pts[:, 0]
        )
        col_centers_eff = _kmeans_1d(all_x, effective_cols)

    # Extrapolate column centers for the physically absent right columns.
    if n_missing_right_cols > 0 and len(col_centers_eff) >= 2:
        step = float(np.median(np.diff(col_centers_eff)))
        last_x = float(col_centers_eff[-1])
        extra = np.array(
            [last_x + step * (i + 1) for i in range(n_missing_right_cols)],
            dtype=np.float32,
        )
        col_centers = np.concatenate([col_centers_eff, extra])
    else:
        col_centers = col_centers_eff

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=bool)
    for row, row_pts in enumerate(row_points):
        if len(row_pts) == 0:
            continue
        if len(row_pts) == effective_cols:
            grid[row, :effective_cols] = row_pts
            valid[row, :effective_cols] = True
            continue

        labels = np.argmin(np.abs(row_pts[:, 0:1] - col_centers[:effective_cols][None, :]), axis=1)
        for point, col_idx in zip(row_pts, labels):
            col_idx = int(col_idx)
            if col_idx < effective_cols:
                if (not valid[row, col_idx]) or abs(point[0] - col_centers[col_idx]) < abs(
                    grid[row, col_idx, 0] - col_centers[col_idx]
                ):
                    grid[row, col_idx] = point
                    valid[row, col_idx] = True

    # Force broken interior cells to be re-derived by interpolation.
    if broken_cells:
        for r, c in broken_cells:
            if 0 <= r < rows and 0 <= c < cols:
                valid[r, c] = False

    for row in range(rows):
        if np.count_nonzero(valid[row]) >= 2:
            cols_valid = np.flatnonzero(valid[row]).astype(np.float32)
            x_valid = grid[row, valid[row], 0]
            y_valid = grid[row, valid[row], 1]
            x_fit = np.polyfit(cols_valid, x_valid, deg=1)
            y_fill = float(np.median(y_valid))
            for col in range(cols):
                if not valid[row, col]:
                    grid[row, col] = (float(np.polyval(x_fit, col)), y_fill)
        else:
            for col in range(cols):
                if not valid[row, col]:
                    grid[row, col] = (col_centers[col], row_centers[row])

    if _broken_gripper_active and image_shape is not None:
        _h, _w = image_shape[:2]
        grid[:, :, 0] = np.clip(grid[:, :, 0], 0, _w - 1)
        grid[:, :, 1] = np.clip(grid[:, :, 1], 0, _h - 1)

    return grid.reshape(expected, 2).astype(float).tolist()


def _select_regular_row_points(row_pts: np.ndarray, cols: int) -> np.ndarray:
    """Pick the most grid-like subset of points in one marker row.

    Edge glare/crop artifacts often appear as an extra point near the image
    border. For rows with more than the expected number of points, selecting
    the contiguous subset with the most uniform spacing drops those outliers
    while keeping the true marker row.
    """
    row_pts = np.asarray(row_pts, dtype=np.float32).reshape(-1, 2)
    if len(row_pts) == 0:
        return row_pts.reshape(0, 2)

    row_pts = row_pts[np.argsort(row_pts[:, 0])]
    if len(row_pts) <= cols:
        return row_pts

    row_y = float(np.median(row_pts[:, 1]))
    row_y_scale = max(1.0, float(np.median(np.abs(row_pts[:, 1] - row_y))) * 1.4826)

    def score_candidate(candidate: np.ndarray) -> float:
        diffs = np.diff(candidate[:, 0])
        if np.any(diffs <= 1.0):
            return float("inf")
        median_diff = float(np.median(diffs))
        if median_diff <= 0:
            return float("inf")
        spacing_cv = float(np.std(diffs) / median_diff)
        span = float(candidate[-1, 0] - candidate[0, 0])
        y_spread = float(np.std(candidate[:, 1]))
        y_outlier = float(np.max(np.abs(candidate[:, 1] - row_y)) / row_y_scale)
        return spacing_cv + 0.01 * y_spread + 0.2 * y_outlier - 0.0001 * span

    best_score = float("inf")
    best = row_pts[:cols]
    if len(row_pts) <= cols + 4:
        candidates = (row_pts[list(indices)] for indices in itertools.combinations(range(len(row_pts)), cols))
    else:
        candidates = (row_pts[start : start + cols] for start in range(0, len(row_pts) - cols + 1))

    for candidate in candidates:
        candidate = candidate[np.argsort(candidate[:, 0])]
        score = score_candidate(candidate)
        if score < best_score:
            best_score = score
            best = candidate
    return best


def _regularize_grid_with_missing_edge_column(
    row_points: list[np.ndarray],
    row_centers: np.ndarray,
    rows: int,
    cols: int,
    image_shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    counts = np.array([len(row_pts) for row_pts in row_points], dtype=np.int32)
    if np.count_nonzero(counts == cols - 1) < max(3, rows // 2):
        return None

    usable_rows = [row_pts for row_pts in row_points if len(row_pts) >= cols - 2]
    if not usable_rows:
        return None

    diffs = []
    first_x = []
    last_x = []
    for row_pts in usable_rows:
        ordered = row_pts[np.argsort(row_pts[:, 0])]
        if len(ordered) >= 2:
            diffs.extend(np.diff(ordered[:, 0]).tolist())
            first_x.append(float(ordered[0, 0]))
            last_x.append(float(ordered[-1, 0]))
    if not diffs:
        return None

    step = float(np.median(diffs))
    if step <= 1.0:
        return None

    left_margin = float(np.median(first_x))
    right_margin = 0.0
    if image_shape is not None and len(image_shape) >= 2:
        width = float(image_shape[1])
        right_margin = width - float(np.median(last_x))

    # If a full edge column is missing, the first/last detected column leaves a
    # margin roughly larger than one marker spacing. Bias toward missing-left
    # because downstream tactile crops usually leave extra dark background on
    # the left, while right-edge artifacts are already handled by row subset
    # selection.
    missing_left = left_margin > step * 1.2
    missing_right = (not missing_left) and (right_margin > step * 1.2)
    if not missing_left and not missing_right:
        return None

    # For the missing_right case, determine the expected x position of grid col 0
    # from the rows that have exactly cols-1 detections (those rows start at col 0).
    _missing_right_x_col0 = left_margin
    if not missing_left:
        x0_list = [
            float(rp[np.argsort(rp[:, 0])][0, 0])
            for rp in row_points
            if len(rp) == cols - 1
        ]
        if x0_list:
            _missing_right_x_col0 = float(np.median(x0_list))

    grid = np.zeros((rows, cols, 2), dtype=np.float32)
    for row, row_pts in enumerate(row_points):
        ordered = row_pts[np.argsort(row_pts[:, 0])]
        if len(ordered) == 0:
            x0 = left_margin - step if missing_left else left_margin
            xs = x0 + step * np.arange(cols, dtype=np.float32)
            grid[row, :, 0] = xs
            grid[row, :, 1] = row_centers[row]
            continue

        if missing_left:
            take = ordered[-(cols - 1) :] if len(ordered) >= cols - 1 else ordered
            start_col = cols - len(take)
            grid[row, start_col:, :] = take[-(cols - start_col) :]
            for col in range(start_col - 1, -1, -1):
                grid[row, col, 0] = grid[row, col + 1, 0] - step
                grid[row, col, 1] = grid[row, col + 1, 1]
        else:
            # For missing_right, anchor the first detected marker to the correct
            # grid column rather than always assuming col 0.  Rows that are also
            # missing their leftmost column(s) would otherwise be shifted right
            # (their first detected marker placed at col 0, leaving the true col 0
            # unrepresented and producing extra fake columns on the right).
            x_first = float(ordered[0, 0])
            start_col = max(0, int(round((x_first - _missing_right_x_col0) / step)))
            start_col = min(start_col, cols - 1)
            n_place = min(len(ordered), cols - 1 - start_col)
            grid[row, start_col : start_col + n_place, :] = ordered[:n_place]
            # Use a row-specific linear fit over the placed markers so that
            # extrapolation accuracy does not depend on the global median step.
            placed_col_indices = np.arange(start_col, start_col + n_place, dtype=np.float32)
            placed_x = grid[row, start_col : start_col + n_place, 0]
            placed_y = grid[row, start_col : start_col + n_place, 1]
            if n_place >= 2:
                x_fit = np.polyfit(placed_col_indices, placed_x, deg=1)
            else:
                x_fit = np.array([step, placed_x[0] - step * start_col]) if n_place == 1 else np.array([step, _missing_right_x_col0])
            y_fill = float(np.median(placed_y)) if n_place >= 1 else row_centers[row]
            for col in range(start_col - 1, -1, -1):
                grid[row, col, 0] = float(np.polyval(x_fit, col))
                grid[row, col, 1] = y_fill
            for col in range(start_col + n_place, cols):
                grid[row, col, 0] = float(np.polyval(x_fit, col))
                grid[row, col, 1] = y_fill

    return grid


def find_marker_centers(
    marker_img: np.ndarray,
    expected_grid: tuple[int, int] | None = (6, 9),
    regularize_grid: bool = True,
    broken_cells: set[tuple[int, int]] | None = None,
    n_missing_right_cols: int = 0,
    detection_margin: int = 0,
):
    props = find_marker_props(marker_img)
    centers = _filtered_marker_centers_from_props(props)
    if not centers:
        centers = [[prop.centroid[1], prop.centroid[0]] for prop in props]
    if detection_margin > 0:
        h, w = marker_img.shape[:2]
        centers = [
            (x, y) for x, y in centers
            if detection_margin <= x < w - detection_margin
            and detection_margin <= y < h - detection_margin
        ]
    if regularize_grid and expected_grid is not None:
        return _regularize_marker_grid(
            centers,
            expected_grid,
            image_shape=marker_img.shape,
            broken_cells=broken_cells,
            n_missing_right_cols=n_missing_right_cols,
        )
    return [[float(x), float(y)] for x, y in centers]

def plot_marker_center(img:np.ndarray, centers:np.ndarray):
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        out = np.copy(img)
    for pt in centers:
        cv2.circle(out, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), thickness=-1)
    return out

def plot_marker_displacement(img:np.ndarray, st_points:np.ndarray, end_points:np.ndarray, pt_color=(0,255,0), arrow_corlor=(255,0,0)):
    out_img = np.ascontiguousarray(img).copy()
    for st_pt, end_pt in zip(st_points, end_points):
        cv2.circle(out_img, (int(st_pt[0]), int(st_pt[1])), 3, pt_color, thickness=-1)
        cv2.arrowedLine(out_img, (int(st_pt[0]), int(st_pt[1])), (int(end_pt[0]), int(end_pt[1])), arrow_corlor, thickness=1, line_type=cv2.LINE_AA, tipLength=1)
    return out_img

def plot_marker_displacement2(img:np.ndarray, mask:np.ndarray, shift:np.ndarray, pt_color=(0,255,0), arrow_corlor=(255,0,0)):
    out_img = np.ascontiguousarray(img).copy() # image need to be contiguous
    ylist, xlist = np.nonzero(mask)
    for x, y in zip(xlist, ylist):
        dx, dy = shift[y, x]
        cv2.circle(out_img, (int(x), int(y)), 3, pt_color, thickness=-1)
        cv2.arrowedLine(out_img, (int(x), int(y)), (int(x + dx), int(y + dy)), arrow_corlor, thickness=1, line_type=cv2.LINE_AA, tipLength=1)
    return out_img

def plot_marker_delta(img:np.ndarray, end_points:np.ndarray, delta_vecs:np.ndarray, pt_color=(0, 255, 0), arrow_color=(255,0,0), scale:float=3.0):
    out_img = np.ascontiguousarray(img).copy()
    for end_pt, delta in zip(end_points, delta_vecs):
        cv2.circle(out_img, (int(end_pt[0]), int(end_pt[1])), 3, pt_color, thickness=-1)
        cv2.arrowedLine(out_img, (int(end_pt[0]), int(end_pt[1])), (int(end_pt[0] + delta[0]*scale), int(end_pt[1] + delta[1]*scale)), arrow_color, thickness=2, line_type=cv2.LINE_AA, tipLength=0.2)
    return out_img
