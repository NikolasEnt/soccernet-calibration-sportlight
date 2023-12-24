import numpy as np


def find_closest_points(points, x, y, any_side=False):
    distances = []
    for i, point in enumerate(points):
        distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
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
        if ((x1 <= x and x <= x2) or (x1 >= x and x >= x2))\
                and ((y1 <= y and y <= y2) or (y1 >= y and y >= y2)):
            return np.vstack((points[idx1], points[idx2]))
    return None
