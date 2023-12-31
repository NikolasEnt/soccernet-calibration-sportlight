import numpy as np
import torch
from argus.metrics import Metric


def angle_between_lines(m1: float, m2: float) -> float:
    """
    Calculate the absolute angle between two lines given their slopes.

    Args:
        m1 (float): The slope of the first line.
        m2 (float): The slope of the second line.

    Returns:
        float: The absolute angle between the two lines.
    """
    return np.abs(np.arctan((m2 - m1) / (1 + m1 * m2)))


class AccMetric(Metric):
    """The accuracy value of distance.

    This metric evaluates the predictions at different threshold levels and
    calculates the accuracy as a weighted value from these different
    thresholds.

    The metric evaluate the predictions at different threshold levels, defined
    as t. The A@t = TP /(TP + FP +FN). The final accuracy is a weighted value
    from different t values, similar to the metric that used by Soccernet.

    Args:
        num_keypoints (int, optional): Number of keypoints. Defaults to 23.
        conf_threshold (float, optional): Confidence threshold.
            Defaults to 0.2.
        device (str, optional): The device to use for computations.
            Defaults to 'cuda:0'.
    """
    name = 'acc'
    better = 'max'

    def __init__(self, num_keypoints: int = 23, conf_threshold: float = 0.2,
                 device: str = 'cuda:0'):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.conf_threshold = conf_threshold
        self.reset()
        self.device = device

    def reset(self):
        """Reset the accumulator for the accuracy values after each epoch."""
        self.acc = []

    def a_t_score(self, gt: torch.Tensor, pred: torch.Tensor, t: float,
                  p_threshold: float = 0.5):
        """
        Calculate the accuracy score at a specific threshold (a@t).

        Args:
            gt (torch.Tensor): Ground truth tensor of shape (B, C, 2, 3).
            pred (torch.Tensor): Prediction tensor of shape (B, C, 2, 3).
            t (float): Threshold distance to determine if a prediction is
                correct.
            p_threshold (float, optional): Confidence threshold for detection.
                Defaults to 0.5.

        Returns:
            float: The calculated accuracy score at the specified threshold.
        """

        batch, channel, _, _ = gt.shape
        tp, fp, fn = 0, 0, 0

        for b in range(batch):
            for c in range(channel):

                gt_points = gt[b, c, :, :]
                pred_points = pred[b, c, :, :2]
                pred_probs = pred[b, c, :, 2]

                gt_exists = gt_points[:, 2] == 1
                pred_exists = pred_probs >= p_threshold

                # Calculate distances between gt and pred points
                # Get the distance pairs [(gt0_p0, gt0_p1), (gt1_p0, gt1_p1)]
                distances = torch.cdist(gt_points[:, :2], pred_points)

                # Find the closest pred point for each gt point
                min_distances, _ = distances.min(dim=1)

                # Determine if closest points are within the threshold
                # distance t
                within_t = min_distances <= t

                # Calculate tp, fp, fn
                tp += (gt_exists & pred_exists & within_t).sum().item()
                fn += (gt_exists & ~pred_exists).sum().item()
                fp += (pred_exists & ~gt_exists).sum().item()
                fp += (gt_exists & pred_exists & ~within_t).sum().item()

        # Calculate a@t
        a_t = tp / (tp + fp + fn)

        return a_t

    def update(self, step_output: dict):
        """
        Update the metric with predictions and ground truth from a batch.

        Args:
            step_output (dict): A dictionary containing 'prediction' and
                'keypoints' keys.
        """
        preds = step_output['prediction']
        kpts = step_output['keypoints'].reshape(-1, self.num_keypoints, 2, 3)

        # Define weights for averaging the final accuracy
        ts = [5, 10, 20]
        ws = [0.5, 0.35, 0.15]

        acc = 0
        # Get weighted accuracy
        for i, t in enumerate(ts):
            acc = self.a_t_score(kpts, preds, t=t,
                                 p_threshold=self.conf_threshold)
            acc += acc * ws[i]

        self.acc.append(acc)
        # print('the current acc is:', self.acc)

    def compute(self) -> float:
        """Compute the final accuracy over all batches.

        Returns:
            float: The mean accuracy over all accumulated batches.
        """
        # print('the current acc for current epoch:', np.mean(self.acc))
        return np.mean(self.acc)
