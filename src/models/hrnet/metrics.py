import concurrent.futures as cf
from typing import Any, List, Tuple, Callable
from collections import defaultdict

import numpy as np
import torch
from argus.engine import State
from argus.metrics import Metric

from baseline.evaluate_camera import get_polylines, evaluate_camera_prediction
from baseline.evaluate_extremities import mirror_labels


class L2metric(Metric):
    """L2 metric, which also calculates other supplimentary metrics.

    It includes Pcks (percent of correct keypoints withing distance threshold),
    precision, recall.

    The metric calculates L2 per points. It ignores detections, which are not
        confident enough (i.e. the metric may be 0 if model predicts no points
        confidently).

    Args:
        num_keypoints (int): Number of keypoints. Defaults to 30.
        conf_threshold (float): Confidence threshold to consider point as
            detected. Should be iin range [0..1].
        pckhs_thres (List[float]): List of keypoints distance threshold (in px)
            compute the Pcks metrics. Defaults to [2.0, 5.0, 10.0, 50.0].

    """
    name = 'l2'
    better = 'min'

    def __init__(self, num_keypoints: int = 30, conf_threshold: float = 0.5,
                 pckhs_thres: List[float] = [2.0, 5.0, 10.0, 50.0]):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.conf_threshold = conf_threshold
        self.pckhs_thres = pckhs_thres
        self.reset()

    def reset(self):
        self.sum = 0
        # Number of points, presented in both: GT and preds after thresholding
        self.num_el = 0
        self.gt_points = 0  # Number of points in the GT
        self.pred_points = 0  # Number of points preds after thresholding
        self.n_fp = 0
        self.n_fn = 0
        self.pckhs = [0] * len(self.pckhs_thres)

    def update(self, step_output: dict):
        preds = step_output['prediction']
        kpts = step_output['target'].reshape(-1, self.num_keypoints, 3)
        l2 = torch.linalg.vector_norm(
            preds[:, :, :2]-kpts[:, :, :2], dim=-1)
        conf = preds[:, :, 2]
        gt_valid_points = kpts[:, :, 0] != -1
        pred_valid_points = conf > self.conf_threshold
        mask = gt_valid_points * pred_valid_points
        self.num_el += torch.count_nonzero(mask)
        l2 = l2[mask]
        self.sum += torch.sum(l2)
        self.gt_points += torch.count_nonzero(gt_valid_points)
        self.pred_points += torch.count_nonzero(pred_valid_points)
        self.n_fp = torch.count_nonzero(~gt_valid_points * pred_valid_points)
        self.n_fn = torch.count_nonzero(~pred_valid_points * gt_valid_points)
        for i, threshold in enumerate(self.pckhs_thres):
            self.pckhs[i] += torch.count_nonzero(l2[l2 < threshold])

    def compute(self) -> float:
        if self.num_el > 0:
            return self.sum / self.num_el
        else:
            return float('inf')

    def epoch_complete(self, state: State):
        with torch.no_grad():
            score = self.compute()

        precision, recall = 0.0, 0.0
        if self.num_el > 0:
            precision = self.num_el / (self.num_el + self.n_fp)
            recall = self.num_el / (self.num_el + self.n_fn)
        name_prefix = f'{state.phase}_' if state.phase else ''
        state.metrics[f'{name_prefix}precision'] = precision
        state.metrics[f'{name_prefix}recall'] = recall
        for i, pckh_sum in enumerate(self.pckhs):
            pckh = 0.0
            if self.gt_points > 0:
                pckh = pckh_sum / self.gt_points
            state.metrics[f'{name_prefix}pcks-{self.pckhs_thres[i]}'] = pckh
        state.metrics[f'{name_prefix}{self.name}'] = score


class Evaluator:
    """Support class to use with parallel execution.
    """

    def __init__(self, pred2cam: Callable, threshold: int = 5,
                 img_size: Tuple[int, int] = (960, 540)):
        self.pred2cam = pred2cam
        self.threshold = threshold
        self.img_size = img_size

    def __call__(self, x: Tuple[np.ndarray, dict, str]):
        pred, annot, name = x
        cam = self.pred2cam(pred, name)
        if cam is not None:
            img_prediction = get_polylines(
                cam, self.img_size[0], self.img_size[1],
                sampling_factor=0.9)
            confusion1, per_class_conf1, reproj_errors1 = \
                evaluate_camera_prediction(img_prediction, annot,
                                           self.threshold)
            confusion2, per_class_conf2, reproj_errors2 =\
                evaluate_camera_prediction(img_prediction,
                                           mirror_labels(annot),
                                           self.threshold)
            accuracy1, accuracy2 = 0.0, 0.0
            if confusion1.sum() > 0:
                accuracy1 = confusion1[0, 0] / confusion1.sum()
            if confusion2.sum() > 0:
                accuracy2 = confusion2[0, 0] / confusion2.sum()

            if accuracy1 > accuracy2:
                accuracy = accuracy1
                confusion = confusion1
                per_class_conf = per_class_conf1
                reproj_errors = reproj_errors1
            else:
                accuracy = accuracy2
                confusion = confusion2
                per_class_conf = per_class_conf2
                reproj_errors = reproj_errors2
            return (accuracy, confusion, per_class_conf, reproj_errors)

        return None


class EvalAImetric(Metric):
    """Metric based on the original reference baseline.

    Args:
        pred2cam (Callable): A callable object which returns Camera once called
            on a single frame predictions.
        threshold (int): Distance threshold in pixels to consider a point as
            correct reprojection. Defaults to 5.
        img_size (Tuple[int, int]): Size of the image (W, H). Defaults to
            (960, 540).
        max_workers (int): Maximum number of workers to use for parallel
            evaluation of the metric. Defaults to 16.

    """
    name = 'evalai'
    better = 'max'

    def __init__(self, pred2cam: Callable, threshold: int = 5,
                 img_size: Tuple[int, int] = (960, 540), max_workers: int = 16):
        self.pred2cam = pred2cam
        self.threshold = threshold
        self.img_size = img_size
        self.executor = cf.ProcessPoolExecutor(max_workers=max_workers)
        self.evaluator = Evaluator(pred2cam, threshold, img_size)
        self.reset()

    def reset(self):
        self.total_frames = 0
        self.missed_frames = 0
        self.tp = 0.0
        self.n_precision = 0
        self.recall = 0.0
        self.n_recall = 0
        self.accuracy = 0.0
        self.n_accuracy = 0
        self.per_class_confusion = defaultdict(lambda: np.zeros((2, 2)))
        self.l2_proj_sum = 0.0
        self.n_l2_proj = 0

    def update(self, step_output: dict):
        preds = step_output['prediction'].cpu().numpy()
        raw_annots = step_output['raw_annots']
        img_name = step_output['img_name']
        self.total_frames += preds.shape[0]
        for res in self.executor.map(self.evaluator,
                                     [(preds[i], raw_annots[i], img_name[i])
                                      for i in range(preds.shape[0])]):
            if res is not None:
                accuracy, confusion, per_class_conf, reproj_errors = res
                self.accuracy += accuracy
                self.n_accuracy += 1

                self.tp += confusion[0, 0]
                self.n_precision += confusion[0, :].sum()
                self.n_recall += confusion[0, 0] + confusion[1, 0]

                for line_class, confusion_mat in per_class_conf.items():
                    self.per_class_confusion[line_class] += confusion_mat

                if len(reproj_errors) > 0:
                    for line_type in reproj_errors:
                        self.n_l2_proj += len(reproj_errors[line_type])
                        self.l2_proj_sum += sum(reproj_errors[line_type])

            else:
                self.missed_frames += 1

    def compute(self) -> float:
        if self.total_frames > 0:
            return (self.total_frames - self.missed_frames)/self.total_frames
        else:
            return 0.0

    def epoch_complete(self, state: State):
        name_prefix = f'{state.phase}_' if state.phase else ''
        completeness = self.compute()
        precision = self.tp / self.n_precision if self.n_precision > 0 else 0.0
        recall = self.tp / self.n_recall if self.n_recall > 0 else 0.0
        accuracy = self.accuracy / self.n_accuracy\
            if self.n_accuracy > 0 else 0.0
        l2_reproj = self.l2_proj_sum / self.n_l2_proj if self.n_l2_proj > 0\
            else float('inf')
        state.metrics[f'{name_prefix}l2_reprojection'] = l2_reproj
        state.metrics[f'{name_prefix}completeness'] = completeness
        state.metrics[f'{name_prefix}eval_precision'] = precision
        state.metrics[f'{name_prefix}eval_recall'] = recall
        state.metrics[f'{name_prefix}eval_accuracy'] = accuracy
        state.metrics[f'{name_prefix}{self.name}'] = completeness * accuracy
        print(self.pred2cam.stat)
