from typing import Tuple

import torch
from torch import nn


def gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
    """1D Gaussian distribution. The distribution amplitude is 1.0.

    Args:
        x (torch.Tensor): 1D tensor of X values, (X,).
        mu (torch.Tensor): Mean values for gaussian (B, N).
        sigma (float): Standard deviation in scale of X axis.

    Returns:
        torch.Tensor: Resulted 1d gaussians: (B, N, X).
    """
    return torch.exp(-(torch.div(x - mu.unsqueeze(-1), sigma) ** 2) / 2.0)


def create_heatmaps(keypoints: torch.Tensor, sigma: float,
                    pred_size: Tuple[int, int] = (68, 120)) -> torch.Tensor:
    """Create Gaussian distributions heatmaps for keypoints.

    Each heatmap is drawn on an individual channel.

    Args:
        keypoints (torch.Tensor): A batch (B) of N points, each point is (x, y).
            Expected shape: (B, N, 2).
        sigma (float): Standard deviation.
        pred_size (Tuple[int, int]): Size of the 2D Gaussian distribution canvas
            (H, W). Defaults to (68, 120).

    Returns:
        (torch.Tensor): Resulted Gaussian heatmaps: (B, N, H, W).

    """
    h, w = pred_size
    device = keypoints.device
    x = keypoints[:, :, 0]
    y = keypoints[:, :, 1]

    x_range = torch.arange(0, w, device=device, dtype=torch.float32)
    y_range = torch.arange(0, h, device=device, dtype=torch.float32)
    gauss_x: torch.Tensor = gaussian(x_range, x, sigma)
    gauss_y: torch.Tensor = gaussian(y_range, y, sigma)
    heatmaps = torch.einsum("BNW, BNH -> BNHW", gauss_x, gauss_y)

    visible_points = torch.any(keypoints == 1, dim=-1, keepdim=True)
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    heatmaps = torch.where(visible_points.unsqueeze(-1), heatmaps, zero)
    return heatmaps


class HRNetLoss(nn.Module):
    def __init__(self, num_refinement_stages: int = 1,
                 sigma: float = 1.0,
                 stride: int = 1,
                 pred_size: Tuple[int, int] = (540, 960),
                 num_keypoints: int = 57,
                 l2_w: float = 1.0,
                 kldiv_w: float = 1.0,
                 awing_w: float = 0.0):
        super().__init__()
        self.sigma = sigma
        self.stride = stride
        self.pred_size = pred_size
        self.num_keypoints = num_keypoints
        self.n_losses = num_refinement_stages + 1
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.l2_w = l2_w
        self.kldiv_w = kldiv_w
        self.awing_w = awing_w
        # Adaptive wing loss
        self.alpha = 2.1
        self.omega = 14
        self.epsilon = 1
        self.theta = 0.5

    def create_target(self, keypoints: torch.Tensor) -> torch.Tensor:
        heatmaps = create_heatmaps(keypoints, sigma=self.sigma,
                                   pred_size=self.pred_size)
        # Add the last channel with inverted heatmaps values
        heatmaps = torch.cat(
            [heatmaps, (1.0 - torch.max(heatmaps, dim=1, keepdim=True)[0])], 1)
        return heatmaps

    def forward(self, pred, target, mask=None):
        keypoints = target.detach().clone().reshape(-1, self.num_keypoints, 3)
        # Scale keypoints to the prediction tensor scale
        keypoints[:, :, :2] /= self.stride
        heatmap = self.create_target(keypoints)
        if mask is not None:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            heatmap = heatmap * mask

        losses = []
        for loss_idx in range(self.n_losses):
            if mask is None:
                pred_masked = pred[loss_idx]
            else:
                pred_masked = pred[loss_idx] * mask

            pred_01 = torch.exp(pred_masked)  # As we have log on output
            loss = 0
            if self.l2_w > 0.0:
                loss += self.l2_w * self.mse(pred_01, heatmap)
            if self.kldiv_w > 0.0:
                loss += self.kldiv_w * self.kldiv_loss(pred_masked, heatmap)
            if self.awing_w > 0.0:
                loss += self.awing_w * self.adaptive_wing(pred_01, heatmap)

            losses.append(loss)

        loss = losses[0]
        for loss_idx in range(1, len(losses)):
            loss += losses[loss_idx]
        return loss

    def l2_loss(self, pred, target, mask=None):
        loss = pred - target
        if mask is not None:
            loss = loss * mask
        batch_size = target.shape[0]
        loss = (loss * loss)
        return loss.sum() / 2.0 / batch_size

    def adaptive_wing(self, pred, target):
        delta = (target - pred).abs()
        alpha_t = self.alpha - target
        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon,
                               alpha_t))) * alpha_t\
            * (torch.pow(self.theta / self.epsilon,
                         self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, alpha_t))

        losses = torch.where(delta < self.theta,
                             self.omega * torch.log(
                                 1 + torch.pow(delta / self.epsilon, alpha_t)),
                             A * delta - C)
        return torch.mean(losses)
