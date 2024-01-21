import os
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from src.datatools.line import get_extreme_points, sort_anno
from src.datatools.reader import read_annot
from torch.utils.data.dataset import Dataset


class EHMDataset(Dataset):
    """Dataset for extreme heatpoint detection model.

    Notes:
        The dataset sample includes:
            - keypoints (np.ndarray): keypoint coordinates in pixels in an
                order of x, y, flag.
            - image (np.ndarray): The original image. Shape is (H, W, C).
            - line_para (List[Tuple[float, float]]): List of line_para for
                each line (slope, intercept).
            - keypoint-maps: The target heatmaps, each heatmap corresponding
                to one line, in other words, two extreme heat points. Shape is
                (num_keypoint_pairs, H // stride, W // stride).

    Args:
        dataset_folder (str): The path to the dataset folder.
        stride (int): The factor that is used to down-sample the input.
        sigma (int): The standard deviation of the Gaussian kernel used to
            generate the heat points.
        input_size (Tuple[int, int]): (height, width) of
            the input image.
        num_keypoint_pairs (int): The number of keypoint pairs to generate
            lines.
        transform (Optional[Callable]): Data transformation applied to the
            input.
    """
    def __init__(self,
                 dataset_folder: str,
                 stride: int = 4,
                 sigma: int = 7,
                 input_size: Tuple[int, int] = (960, 540),
                 num_keypoint_pairs: int = 23,
                 transform: Optional[Callable] = None):
        super().__init__()
        self._dataset_folder = dataset_folder
        self._stride = stride
        self._sigma = sigma
        self.num_keypoint_pairs = num_keypoint_pairs
        self._transform = transform
        self._labels = []
        self._img_paths = []
        self.input_size = input_size
        for fname in os.listdir(dataset_folder):
            if 'info' not in fname:
                annot_path = os.path.join(dataset_folder, fname)
                if annot_path.endswith('.json'):
                    img_path = annot_path.replace('.json', '.jpg')
                    if os.path.exists(img_path):
                        res = read_annot(annot_path)
                        res, usable_flag = sort_anno(res,
                                                     img_size=self.input_size)
                        if usable_flag:
                            self._labels.append(
                                get_extreme_points(res,
                                                   img_size=self.input_size))
                            self._img_paths.append(img_path)
        print(f'Size of {dataset_folder} is {len(self._img_paths)}')

    def __getitem__(self, idx: int) -> Dict:
        kpts_dict = self._labels[idx]
        image = cv2.imread(self._img_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.input_size)
        keypoints = np.ones(self.num_keypoint_pairs * 3 * 2,
                            dtype=np.float32) * -1
        line_paras = []
        for idx in range(self.num_keypoint_pairs):
            if kpts_dict[idx] is not None:
                # (points, line_para)
                points = kpts_dict[idx][0]
                keypoints[idx * 6] = points[0][0]
                keypoints[idx * 6 + 1] = points[0][1]
                keypoints[idx * 6 + 2] = 1
                keypoints[idx * 6 + 3] = points[1][0]
                keypoints[idx * 6 + 4] = points[1][1]
                keypoints[idx * 6 + 5] = 1

                line_paras.append(kpts_dict[idx][1])
            else:
                keypoints[idx * 6 + 2] = 0
                keypoints[idx * 6 + 5] = 0
                line_paras.append((np.nan, np.nan))

        sample = {
            'keypoints': keypoints,
            'image': image,
            'line_para': line_paras
        }
        sample['keypoint_maps'] = self._generate_keypoint_maps(sample)
        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self._labels)

    def _generate_keypoint_maps(self, sample: Dict) -> torch.Tensor:
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(
            shape=(self.num_keypoint_pairs, int(round(n_rows / self._stride)),
                   int(round(n_cols / self._stride))), dtype=np.float32)

        keypoints = sample['keypoints']

        all_ps = []
        for id in range(len(keypoints) // 6):
            # Prepare points to be converted to heat
            points = []
            if keypoints[id * 6 + 2] == 1:
                p1 = (keypoints[id * 6], keypoints[id * 6 + 1])
                points.append(p1)
                all_ps.append(p1)
            if keypoints[id * 6 + 5] == 1:
                p2 = (keypoints[id * 6 + 3], keypoints[id * 6 + 4])
                points.append(p2)
                all_ps.append(p2)
            self._add_gaussian(keypoint_maps[id], points,
                               self._stride, self._sigma)
        return torch.tensor(keypoint_maps)

    def _add_gaussian(self, keypoint_map: np.ndarray, points: list,
                      stride: int, sigma: float = 1) -> np.ndarray:
        """
         Adds Gaussian peaks to a keypoint map at specified points.

         This method modifies the input keypoint map by adding Gaussian peaks
         centered at the provided point locations. Each point in 'points' is
         expected to be a 2D coordinate. The Gaussian peak is added such that
         its maximum is at the point's location, and it spreads out according
         to the specified 'sigma'.
         The method handles the stride and ensures the Gaussian is added
         correctly considering the scale of the heatmap.

         Args:
             keypoint_map (np.ndarray): A 2D numpy array representing the
                 keypoint map to which Gaussian peaks will be added. Its shape
                 is expected to be (img_h // stride, img_w // stride).
             points (list): A list of points, where each point is a list or
                 tuple of two elements representing the x and y coordinates,
                 respectively.
             stride (int): The stride of the keypoint map, which scales the
                point locations appropriately.
             sigma (float, optional): The standard deviation of the Gaussian
                distribution. Defaults to 1.

         Returns:
             np.ndarray: The updated keypoint map with Gaussian peaks added at
                the specified points.
                Shape is (img_h // stride, img_w // stride).
         """
        # keypoint_map shape of img_h//stride, img_w//stride
        h, w = keypoint_map.shape

        if len(points) > 0:
            x = np.arange(w).astype(float)
            y = np.arange(h).astype(float)
            x_grids, y_grids = np.meshgrid(x, y)
            for point in points:
                x, y = point[0], point[1]
                mu_x, mu_y = (min(w-1, round(x/stride)),
                              min(h-1, round(y/stride)))
                gauss = np.exp(-((x_grids - mu_x) ** 2 + (y_grids - mu_y) ** 2
                                 ) / (2 * sigma ** 2))
                max_value = np.max(gauss)
                gauss /= max_value
                keypoint_map += gauss

        return keypoint_map


if __name__ == "__main__":
    dataset_path = '/workdir/data/dataset/valid'
    data = EHMDataset(dataset_path)
    first_data = data[0]
    print(first_data)
