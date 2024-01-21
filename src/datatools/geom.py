from typing import List, Tuple, Optional

import numpy as np


def find_longest_line(points: List[np.ndarray], longest: bool = True)\
        -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find the longest/shortest line from a list of points.

    Args:
        points (List[np.ndarray]): List of points to process. Each point is an
            array of two elements: x and y.
        longest (bool): If True - return the longest line, otherwise - return
            the shortest line.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Start and end points of the 
            line, or None if the list was not found.
    """
    best_distance = 0 if longest else float('inf')
    best_line = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[j] - points[i])
            better = distance > best_distance if longest\
                else distance < best_distance
            if better:
                best_distance = distance
                best_line = (points[i], points[j])
    return best_line


def point_within_img(point: Optional[Tuple[float, float]],
                     img_size: Tuple[int, int] = (960, 540),
                     within_img: bool = True,
                     margin: float = 0.0) -> Optional[Tuple[float, float]]:
    """Check if a point is within the image (with some optional margins).

    Args:
        point (Optional[Tuple[float, float]]): Point (x, y) or None.
        img_size (Tuple[int, int], optional): Size of the image (W, H).
            Defaults to (960, 540).
        within_img (bool, optional): Check, whether the point is within the
            image. Defaults to True.
        margin (float, optional): Margins to add around the border to include
            the point. Defaults to 0.0.

    Returns:
        Optional[Tuple[float, float]]: Point (x, y) if it meets the conditions.
    """
    if point is not None:
        if within_img:
            x, y = point
            if 0-margin <= x <= img_size[0]+margin\
                    and 0-margin <= y <= img_size[1]+margin:
                return point
            else:
                return None
    return point
