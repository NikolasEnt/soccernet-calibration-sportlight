import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms as T


class ColorAugment:
    """
    Applies color augmentation to images by adjusting brightness, color, and
    contrast.

    Args:
        brightness (Tuple[float, float], optional): Range for brightness
            adjustment. Defaults to (0.8, 1.2).
        color (Tuple[float, float], optional): Range for color adjustment.
            Defaults to (0.8, 1.2).
        contrast (Tuple[float, float], optional): Range for contrast
            adjustment. Defaults to (0.8, 1.2).
    """

    def __init__(self,
                 brightness: Tuple[float, float] = (0.8, 1.2),
                 color: Tuple[float, float] = (0.8, 1.2),
                 contrast: Tuple[float, float] = (0.8, 1.2)):
        self.brightness = brightness
        self.color = color
        self.contrast = contrast

    def _img_aug(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the color augmentation to an image.

        Args:
            img (np.ndarray): The input image to augment.

        Returns:
            np.ndarray: The color-augmented image.
        """
        img = img.astype(float)
        random_colors = np.random.uniform(
            self.brightness[0], self.brightness[1]) \
                        * np.random.uniform(self.color[0], self.color[1], 3)
        for i in range(3):
            img[:, :, i] = img[:, :, i] * random_colors[i]
        mean = img.mean(axis=(0, 1))
        contrast = np.random.uniform(self.contrast[0], self.contrast[1])
        img = (img - mean) * contrast + mean
        img = np.clip(img, 0.0, 255.0)
        img = img.astype(np.uint8)
        return img

    def __call__(self, sample: Dict) -> Dict:
        sample['image'] = self._img_aug(sample['image'])
        return sample


class GaussNoise:
    """Adds Gaussian noise to images.

    Attributes:
        sigma_sq (float): Variance of the Gaussian noise.

    Args:
        sigma_sq (float, optional): Variance of the Gaussian noise.
            Defaults to 30.0.
    """

    def __init__(self, sigma_sq: float = 30.0):
        self.sigma_sq = sigma_sq

    def __call__(self, sample):
        img = sample['image'].astype(int)
        w, h, c = img.shape
        gauss = np.random.normal(0, np.random.uniform(0.0, self.sigma_sq),
                                 (w, h, c))
        img = img + gauss
        img = np.clip(img, 0, 255)
        sample['image'] = img.astype(np.uint8)
        return sample


class UseWithProb:
    """Apply transform with a given probability for data augmentation.

    Args:
        transform (Callable): Transform to apply.
        prob (float, optional): Probability of the transform. Should be in
            range [0..1]. Defaults to 0.5.
    """

    def __init__(self, transform, prob: float = 0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if random.random() < self.prob:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, sample: Dict) -> Dict:
        sample['image'] = self.to_tensor(sample['image'])
        return sample


def flip_keypoints(x: np.ndarray, w: int) -> np.ndarray:
    """
    Flip keypoints horizontally in an image.

    Args:
        x (np.ndarray): The keypoints array.
        w (int): The width of the image.

    Returns:
        np.ndarray: The array with flipped keypoints.
    """

    for i in range(len(x) // 3):
        if x[i * 3] != -1 and x[i * 3 + 2] == 1:
            x[i * 3] = w - x[i * 3] - 1
    return x


class Flip:
    """Horizontally flips an image and its corresponding keypoints."""

    def __call__(self, sample):
        sample['image'] = cv2.flip(sample['image'], 1)
        sample['keypoints'] = flip_keypoints(sample['keypoints'],
                                             sample['image'].shape[1])
        sample['keypoint_maps'] = torch.flip(sample['keypoint_maps'], [2])
        return sample


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def train_transform(brightness: Tuple[float, float] = (0.8, 1.2),
                    color: Tuple[float, float] = (0.8, 1.2),
                    contrast: Tuple[float, float] = (0.8, 1.2),
                    gauss_noise_sigma: float = 30.0,
                    prob: float = 0.5):
    """
    Create a training transformation pipeline.

    Args:
        brightness (Tuple[float, float], optional): Range for brightness
            adjustment. Defaults to (0.8, 1.2).
        color (Tuple[float, float], optional): Range for color adjustment.
            Defaults to (0.8, 1.2).
        contrast (Tuple[float, float], optional): Range for contrast
            adjustment. Defaults to (0.8, 1.2).
        gauss_noise_sigma (float, optional): Variance of the Gaussian noise.
            Defaults to 30.0.
        prob (float, optional): Probability of applying each transformation.
            Defaults to 0.5.

    Returns:
        ComposeTransform: The composed transformation pipeline for training.
    """
    transforms = ComposeTransform([
        UseWithProb(ColorAugment(brightness=brightness,
                                 color=color,
                                 contrast=contrast), prob),
        UseWithProb(GaussNoise(gauss_noise_sigma), prob),
        UseWithProb(Flip(), 0.5),
        ToTensor()
    ])
    return transforms


def test_transform():
    transforms = ComposeTransform([
        ToTensor()
    ])
    return transforms


class EHMPredictionTransform:
    """
    Transforms EHM model predictions.

    Attributes:
        scale (int | float): Scaling factor for coordinates.
        sigma (int | float): Standard deviation for Gaussian mask.
        distance_threshold (int | float): Distance threshold for keypoints.

    Args:
        scale (int | float, optional): Scaling factor for coordinates.
            Defaults to 8.
        sigma (int | float, optional): Standard deviation for Gaussian mask.
            Defaults to 6.
    """

    def __init__(self,
                 scale: int | float = 8,
                 sigma: int | float = 6):
        self.scale = scale
        self.sigma = sigma
        self.distance_threshold = 2 * self.sigma

    def __call__(self, preds: torch.Tensor) -> torch.Tensor:
        # The shape of preds is (B, N, 2, 3)
        prediction = self.mask_heat_points_gauss(preds, sigma=self.sigma)
        prediction[..., 0] *= self.scale
        prediction[..., 1] *= self.scale

        return prediction

    @staticmethod
    def mask_heat_points_gauss(tensor: torch.Tensor, sigma: float = 5) \
            -> torch.Tensor:
        """
        Apply a Gaussian mask to heatmap and extract key points.

        Args:
            tensor (torch.Tensor): The input tensor (heatmap).
            sigma (float, optional): The standard deviation for the Gaussian
                mask. Defaults to 5.

        Returns:
            torch.Tensor: The tensor with key points extracted.
        """
        B, C, H, W = tensor.shape
        heat_points = -torch.ones(B, C, 2, 3, device=tensor.device)

        # Apply relu to remove negative values
        tensor = torch.relu(tensor)

        for b in range(B):
            for c in range(C):
                # Get heatmap
                heatmap = tensor[b, c]

                # Find the first peak
                max_val_1, max_idx_1 = torch.max(heatmap.view(-1), dim=0)
                max_coords_1 = torch.tensor([max_idx_1 % W, max_idx_1 // W],
                                            dtype=torch.float,
                                            device=tensor.device)

                # Assign first peak to heat_points tensor
                heat_points[b, c, 0, :2] = max_coords_1
                heat_points[b, c, 0, 2] = max_val_1

                # Create a meshgrid for the Gaussian
                x = torch.arange(0, W, dtype=torch.float,
                                 device=tensor.device)[None, :]
                y = torch.arange(0, H, dtype=torch.float,
                                 device=tensor.device)[:, None]

                # Create the Gaussian mask
                mask = torch.exp(-((x - max_coords_1[0]) ** 2 + (
                        y - max_coords_1[1]) ** 2) / (2.0 * sigma ** 2))

                # Apply the Gaussian mask and find the second peak
                heatmap = heatmap * (1 - mask)
                max_val_2, max_idx_2 = torch.max(heatmap.view(-1), dim=0)
                max_coords_2 = torch.tensor([max_idx_2 % W, max_idx_2 // W],
                                            dtype=torch.float,
                                            device=tensor.device)

                # Assign second peak to heat_points tensor
                heat_points[b, c, 1, :2] = max_coords_2
                heat_points[b, c, 1, 2] = max_val_2

        return heat_points
