from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from numpy import ndarray
from src.datatools.reader import read_annot


def find_closest_points(points, x, y, any_side=False):
    distances = []
    for i, point in enumerate(points):
        distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
        distances.append((distance, i))

    distances.sort(key=lambda p: p[0])

    idx1 = distances[0][1]
    x1 = points[idx1][0]
    y1 = points[idx1][1]

    if any_side:
        idx2 = distances[1][1]
        return np.vstack((points[idx1], points[idx2]))

    for i in range(1, len(distances)):
        idx2 = distances[i][1]
        x2 = points[idx2][0]
        y2 = points[idx2][1]
        if ((x1 <= x and x <= x2) or (x1 >= x and x >= x2)) \
                and ((y1 <= y and y <= y2) or (y1 >= y and y >= y2)):
            return np.vstack((points[idx1], points[idx2]))
    return None


LINE_CLS: Dict[int, str] = {0: 'Goal left post left ',
                            1: 'Goal right post right',
                            2: 'Middle line',
                            3: 'Small rect. right top',
                            4: 'Side line bottom',
                            5: 'Goal right post left',
                            6: 'Big rect. right main',
                            7: 'Goal left crossbar',
                            8: 'Small rect. left bottom',
                            9: 'Side line left',
                            10: 'Big rect. right top',
                            11: 'Small rect. left top',
                            12: 'Side line right',
                            13: 'Big rect. left top',
                            14: 'Goal left post right',
                            15: 'Small rect. right bottom',
                            16: 'Side line top',
                            17: 'Goal right crossbar',
                            18: 'Small rect. left main',
                            19: 'Big rect. left main',
                            20: 'Big rect. right bottom',
                            21: 'Small rect. right main',
                            22: 'Big rect. left bottom'}


def map_point2pixel(line1: Tuple[float, float],
                    line2: Tuple[float, float],
                    img_size: Tuple[int, int] = (960, 540)) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Converts line endpoints to pixel coordinates in the image.

    Args:
        line1 (Tuple[float, float]): The first line's endpoints in relative
            coordinates.
        line2 (Tuple[float, float]): The second line's endpoints in relative
            coordinates.
        img_size (Tuple[int, int]): Size of the image (width, height).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the pixel
        coordinates of the two lines (extreme points).
    """
    line1_arr = np.array(line1) * img_size
    line2_arr = np.array(line2) * img_size
    return line1_arr, line2_arr


def get_extreme_points(points_data: Dict[str, List[Any]],
                       img_size: Tuple[int, int] = (960, 540)) \
        -> Dict[int, Tuple[Tuple[ndarray, ndarray], Any] | None]:
    """Processes points data to extract extreme points.

    Args:
        points_data (Dict[str, List[Any]]): Dictionary containing points data.
        img_size (Tuple[int, int]): Size of the image (width, height).

    Returns:
        Dict[int, Tuple[Tuple[ndarray, ndarray], Any] | None]: Processed data
            with extreme points and line parameters.
        - points_extreme (Tuple[np.ndarray, np.ndarray]): The (x, y)
            coordinates of the two extreme points.
        - line_param (Tuple[float, float]): The (slope, intercept) of a
            line.
    """
    res = {}
    for i, line_info in LINE_CLS.items():
        res[i] = None
        if line_info in points_data:
            # Entries are (points, (slope, intercept) )
            points = points_data[line_info][0]
            line_param = points_data[line_info][1]
            if len(points) > 1:
                points_extreme = map_point2pixel(points[0], points[-1],
                                                 img_size)
                # Add both points and line parameters
                res[i] = points_extreme, line_param
    return res


def sort_anno(annos: Dict[str, List[Tuple[float, float]]],
              img_size: Tuple[int, int] = (960, 540)) -> \
        Tuple[Dict[str, Any], bool]:
    """Sorts annotations for valid lines.

    Args:
        annos (Dict[str, List[Tuple[float, float]]]): Annotations to be sorted.
        img_size (Tuple[int, int]): Size of the image (width, height).

    Returns:
        Tuple[Dict[str, Any], bool]: Sorted annotations and a flag indicating
            usability.
    """
    annos_update = dict()
    # Used for filter out images that they are one point in a line, or slope
    # or intercept is invalid.
    usable_flag = True
    for k, points in annos.items():
        # There is no line in a circle, so ignore these part of data.
        if k == 'Circle left' or k == 'Circle right' or k == 'Circle central' \
                or k == 'Line unknown':
            pass
        else:
            # Sort the points
            if len(points) >= 2:
                points, slope, intercept = (
                    sort_points_on_line(points, input_size=img_size))
                if slope and intercept:
                    annos_update[k] = points, (slope, intercept)
                else:
                    usable_flag = False
            else:
                usable_flag = False

    return annos_update, usable_flag


def filter_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters out pairs of x, y coordinates where either is NaN or Inf.

    Args:
        x (np.ndarray): Array of x coordinates.
        y (np.ndarray): Array of y coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered arrays of x and y coordinates.
    """
    # Filter out pairs where either x or y is NaN or Inf
    filtered_pairs = [(x_i, y_i) for x_i, y_i in zip(x, y) if
                      np.isfinite(x_i) and np.isfinite(y_i)]

    # Unzip the filtered pairs back into x and y lists
    x_filtered, y_filtered = zip(*filtered_pairs)

    # Convert lists back to numpy arrays
    x_filtered = np.array(x_filtered)
    y_filtered = np.array(y_filtered)

    return x_filtered, y_filtered


def sort_points_on_line(points: List[Tuple[float, float]],
                        input_size=List[int]) -> \
        Tuple[List[Any], Any, Any]:
    """
    Sorts points based on their position on a line.

    Args:
        points (List[Tuple[float, float]]): Points to be sorted.
        input_size (Tuple[int, int]): Size of the input (width, height).

    Returns:
        Tuple[List[Any], Any, Any]: Sorted points and line parameters
            (slope, intercept).
    """
    # Extract x and y coordinates from the list of points
    w, h = input_size
    x = np.array([point[0] for point in points]) * w
    y = np.array([point[1] for point in points]) * h

    x, y = filter_xy(x, y)

    if x.size >= 2 and y.size >= 2 and np.std(x) != 0 and np.std(y) != 0:
        slope, intercept = np.polyfit(x, y, deg=1)
    else:
        # print('Invalid data for fitting')
        slope, intercept = None, None
        return points, slope, intercept

    # Calculate the projected positions of each point onto the best-fit line
    line_origin = np.array([0, intercept])
    line_direction = np.array([1, slope])
    line_direction_normalized = line_direction / (
            np.linalg.norm(line_direction) + 0.00001)

    projected_positions = []
    for point in points:
        point_vector = np.array(point) - line_origin
        projected_position = np.dot(point_vector, line_direction_normalized)
        projected_positions.append(projected_position)

    # Sort the points based on their projected positions
    sorted_points = [point for _, point in
                     sorted(zip(projected_positions, points))]

    return sorted_points, slope, intercept


def loc2img(img: np.ndarray,
            points: dict[int, tuple[tuple[ndarray, ndarray], Any] | None],
            color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Draws points and lines on an image.

    Args:
        img (np.ndarray): The image to draw on.
        points (dict[int, tuple[tuple[ndarray, ndarray], Any] | None]): Points
            to draw.
        color (Tuple[int, int, int]): Color of the points and lines.

    Returns:
        np.ndarray: The image with points and lines drawn.
    """
    for ps in points.values():
        points = []
        if ps is not None:
            for p in ps[0]:
                point = round(p[0]), round(p[1])
                cv2.circle(img, point, radius=5, color=color, thickness=2)
                points.append(point)

            if len(points) > 1:
                cv2.line(img, points[0], points[-1], color=color, thickness=1)

    return img


if __name__ == "__main__":
    # An example of reading an image and draw all its extreme points,
    # and draw lines for every extreme points pair.
    anno_path = '/workdir/data/dataset/train/00000.json'
    img_path = '/workdir/data/dataset/train/00000.jpg'
    sample = read_annot(anno_path)
    sample1 = sort_anno(sample)
    extreme_points = get_extreme_points(sample1[0])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Visualisation
    img_pred = loc2img(img, extreme_points, color=(0, 0, 255))
    cv2.imshow('Draw lines linking two extreme points', img_pred)
    #cv2.imwrite('line_example.png', img_pred)
    cv2.waitKey()
