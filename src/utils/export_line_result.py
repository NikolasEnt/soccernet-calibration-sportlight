import argparse
import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from argus import load_model
from torchvision import transforms as T

from src.datatools.line import LINE_CLS
from src.models.line.metamodel import EHMMetaModel
from src.models.line.transforms import EHMPredictionTransform


def heat_loc2img(img: np.ndarray,
                 points: Dict[str, List[Optional[Tuple[float, float]]]],
                 color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    plot and individual points, are lines if both points are present.
    """
    """
    Draws circles at given points on an image, and lines connecting these 
    points if more than one point is present.

    Args:
        img (np.ndarray): The image on which to draw the points and lines.
        points (Dict[str, List[Optional[Tuple[float, float]]]]): A dictionary 
            where keys are point labels and values are lists of points (x, y).
        color (Tuple[int, int, int], optional): The color of the points and 
            lines. Defaults to red (255, 0, 0).

    Returns:
        np.ndarray: The modified image with points and lines drawn on it.
    """
    for ps in points.values():
        points = []
        for p in ps:
            point = int(p[0]), int(p[1])
            cv2.circle(img, point, radius=5, color=color, thickness=2)
            points.append(point)

        if len(points) > 1:
            cv2.line(img, points[0], points[-1], color=color, thickness=1)

    return img


def calculate_slope_intercept(point1: Tuple[float, float],
                              point2: Tuple[float, float],
                              delta: float = 0.00001) \
        -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the slope and y-intercept of the line defined by two points.

    Args:
        point1 (Tuple[float, float]): The first point (x1, y1).
        point2 (Tuple[float, float]): The second point (x2, y2).
        delta (float, optional): A small value to avoid division by zero.
            Defaults to 0.00001.

    Returns:
        Optional[float], The slope of the line. Returns None if the points are
            identical.
        Optional[float]]: The y-intercept of the line. Returns None if the
            points are identical.
    """
    # Ensure points are not the same
    if point1 == point2:
        return None, None

    # Calculate slope
    x1, y1 = point1
    x2, y2 = point2

    slope = (y2 - y1) / (x2 - x1 + delta)
    # Calculate y-intercept
    intercept = y1 - slope * x1

    return slope, intercept


def get_line_data(heat_loc: np.ndarray or torch.Tensor,
                  scale=4,
                  prob_thre: float = 0.2) \
        -> Tuple[Dict[str, Tuple[float, float]], Dict[
            str, List[Tuple[float, float, float]]]]:
    """
    Processes heat location data to calculate line parameters and valid points.

    Args:
        heat_loc (Union[np.ndarray, torch.Tensor]): Heat location data with
            shape (1, K, 2, 3).
        scale (int, optional): The scale factor for point coordinates.
            Defaults to 4.
        line_cls (List[str]): List of line class names.
        prob_thre (float, optional): Probability threshold for considering
            a point as valid. Defaults to 0.2.

    Returns:
        Dict[str, Tuple[float, float]]: List of line parameters.
        Dict[str, List[Tuple[float, float, float]]]: List of valid point
            coordinates and probability.
    """
    if isinstance(heat_loc, torch.Tensor):
        heat_loc = heat_loc.cpu().numpy()

    _, ks, num_heats, _ = heat_loc.shape

    line_paras = dict()
    final_points = dict()

    for k in range(ks):
        valid_points = []
        for n in range(num_heats):  # n<2.
            x, y, p = heat_loc[0, k, n]

            if p >= prob_thre:
                point = x * scale, y * scale, p
                valid_points.append(point)

        final_points[LINE_CLS[k]] = valid_points

        if len(valid_points) >= 2:
            line_paras[LINE_CLS[k]] = \
                calculate_slope_intercept(point1=valid_points[0][:2],
                                          point2=valid_points[1][:2])

    return line_paras, final_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str,
                        default='/workdir/data/dataset/test',
                        help='Path to image folder that will be evaluated.')
    parser.add_argument('--result-file', type=str,
                        default='results/result_on_test_set.pkl',
                        help='Path to pickle file to save.')
    parser.add_argument('--model', type=str,
                        help='Path to pytorch model.')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='The device to run the model.')
    parser.add_argument('--sigma', type=float,
                        default=3.,
                        help='Gaussian std for mask first heatpoint.')
    parser.add_argument('--prob-thre', type=float,
                        default=0.,
                        help='Probability threshold for heatpoints.')
    parser.add_argument('--scale', type=int,
                        default=4,
                        help='The downscaling factor for original input.')
    parser.add_argument('--no-vis', action='store_true',
                        help='Visualize heatpoints and line of the result.')

    args = parser.parse_args()
    print(args)

    result_dir = os.path.dirname(args.result_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pre_process_fun = T.ToTensor()
    decode = EHMPredictionTransform.mask_heat_points_gauss
    model = load_model(args.model, device=args.device)

    result_output = dict()

    jpg_files = glob.glob(os.path.join(args.image_folder, '*.jpg'))
    for i, file_path in enumerate(jpg_files):
        print(f"{i}: {file_path}")

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img_tensor = pre_process_fun(img).to(args.device)

        with torch.no_grad():
            # The pred tensor (1, 23, 135, 240)
            preds = model.nn_module(img_tensor.unsqueeze(0))[-1]
            # The heat_point_pred tensor (1, 23, 2, 3)
            heat_point_preds = decode(preds, sigma=args.sigma)
            lines, points = get_line_data(heat_point_preds, scale=args.scale,
                                          prob_thre=args.prob_thre)

        filename = os.path.basename(file_path)
        result_output[filename] = {'lines': lines, 'points': points}

        # Visualisation
        if not args.no_vis:
            if len(points) > 0:
                img_pred = heat_loc2img(img, points, color=(255, 0, 0))

            cv2.imshow('model inference', img_pred)
            cv2.waitKey()

            print(points)

    with open(args.result_file, 'wb') as f:
        pickle.dump(result_output, f)


if __name__ == "__main__":
    main()
