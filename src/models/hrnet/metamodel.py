from typing import List, Optional

import torch
import torch.nn as nn
from argus import Model
from argus.utils import deep_to, deep_detach
from argus.engine import State

from src.models.hrnet.loss import HRNetLoss, create_heatmaps
from src.models.hrnet.model import HRNetHeatmap
from src.models.hrnet.transforms import HRNetPredictionTransform


class HRNetMetaModel(Model):
    nn_module = HRNetHeatmap
    loss = HRNetLoss
    optimizer = torch.optim.Adam
    prediction_transform = HRNetPredictionTransform

    def __init__(self, params):
        super().__init__(params)
        self.amp = (False if 'amp' not in self.params
                    else bool(self.params['amp']))
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()
        raw_annots = batch['raw_annot']
        img_name = batch['img_name']
        del batch['raw_annot']
        del batch['img_name']
        batch = deep_to(batch, device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            prediction = self.nn_module(batch['image'])
        loss = self.loss(prediction, batch['keypoints'], batch['mask'])

        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        prediction = deep_detach(prediction)
        prediction = self.prediction_transform(prediction[-1])
        return {
            'prediction': prediction,
            'target': deep_detach(batch['keypoints']),
            'loss': loss.item(),
            'raw_annots': raw_annots,
            'img_name': img_name
        }

    def val_step(self, batch, state: State) -> dict:
        self.eval()
        with torch.no_grad():
            raw_annots = batch['raw_annot']
            img_name = batch['img_name']
            del batch['raw_annot']
            del batch['img_name']
            batch = deep_to(batch, device=self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(batch['image'])
                # keypoints = batch['keypoints'].detach(
                # ).clone().reshape(-1, 57, 3)
                # keypoints[:, :, :2] /= 2
                # heatmaps = create_heatmaps(keypoints, 2.0, (270, 480))
                # heatmaps = torch.cat(
                #     [heatmaps, (1.0 - torch.max(heatmaps, dim=1, keepdim=True)[0])], 1)
                # prediction = [torch.log(heatmaps)]
            loss = self.loss(prediction, batch['keypoints'], batch['mask'])

            prediction = deep_detach(prediction)
            prediction = self.prediction_transform(prediction[-1])
            return {
                'prediction': prediction,
                'target': deep_detach(batch['keypoints']),
                'loss': loss.item(),
                'raw_annots': raw_annots,
                'img_name': img_name
            }

    def save(self, file_path: str, optimizer_state: bool = False):
        """Save the argus model into a file.

        The argus model is saved as a dict::

            {
                'model_name': Name of the argus model,
                'params': Argus model parameters dict,
                'nn_state_dict': torch nn_module.state_dict(),
                'optimizer_state_dict': torch optimizer.state_dict()
            }

        The *state_dict* is always transferred to CPU before saving.

        Args:
            file_path (str or :class:`pathlib.Path`): Path to the argus model
                file.
            optimizer_state (bool): Save optimizer state. Defaults to False.

        """
        nn_module = self.get_nn_module()
        state = {
            'model_name': self.__class__.__name__,
            'params': self.params,
            'nn_state_dict': deep_to(nn_module.state_dict(), 'cpu')
        }
        # This step is necessary for compiled models
        new_state_dict = {}
        for k, v in state['nn_state_dict'].items():
            k2 = k.replace("_orig_mod.", "")
            new_state_dict[k2] = v
        state['nn_state_dict'] = new_state_dict
        if optimizer_state and self.optimizer is not None:
            state['optimizer_state_dict'] = deep_to(
                self.optimizer.state_dict(), 'cpu'
            )
        torch.save(state, file_path)
        self.logger.info(f"Model saved to '{file_path}'")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self._check_predict_ready()
        with torch.no_grad():
            self.eval()
            x = deep_to(x, self.device)
            prediction = self.nn_module(x)
            prediction = self.prediction_transform(prediction[-1])
            return prediction
