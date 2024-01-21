import torch
from torch import nn


class EHMLoss(nn.Module):
    """
    A class representing the Enhanced Heatmap Loss (EHM Loss) used in line
    estimation tasks. This class combines multiple loss functions to calculate
    the total loss for predicted and target heatmaps.

    Args:
        num_refinement_stages (int): Number of refinement stages for the loss
            calculation. Defaults to 1.
        gmse_w (float): Weight of the GMSE loss. Defaults to 1.
        awing_w (float): Weight of the adaptive wing loss. Defaults to 1.
        sigma (float): Standard deviation of the GMSE. Defaults to 4.
    """
    def __init__(self, num_refinement_stages: int = 1,
                 gmse_w: float = 1.0,
                 awing_w: float = 1.0,
                 sigma: float = 4):
        super().__init__()
        self.n_losses = num_refinement_stages + 1
        self.gmse_w = gmse_w
        self.awing_w = awing_w
        # GMSE loss
        self.sigma = sigma
        # Adaptive wing loss
        self.alpha = 2.1
        self.omega = 14
        self.epsilon = 1
        self.theta = 0.5

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Calculate the total loss for the current iteration.

        Args:
            pred (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The total loss calculated over all refinement stages.
        """
        losses = []
        for loss_idx in range(self.n_losses):
            loss = 0
            if self.gmse_w > 0:
                loss += self.gmse_w * self.gmse_loss(pred[loss_idx], target)
            if self.awing_w > 0:
                loss += self.awing_w * self.adaptive_wing(pred[loss_idx],
                                                          target)
            losses.append(loss)

        loss = losses[0]
        for loss_idx in range(1, len(losses)):
            loss += losses[loss_idx]

        return loss

    def gmse_loss(self, pred: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        """Gaussian Mean Squared Error loss function.

        Args:
            pred (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The calculated MSE Gaussian loss.
        """
        squared_diff = (pred - target) ** 2
        gaussian = torch.exp(-squared_diff / (2 * self.sigma ** 2))
        loss = (squared_diff * gaussian).mean()

        return loss

    def adaptive_wing(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        """
        Adaptive wing loss for robust training, particularly effective for
        heatmap-based tasks.

        Args:
            pred (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The calculated adaptive wing loss.
        """
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)
