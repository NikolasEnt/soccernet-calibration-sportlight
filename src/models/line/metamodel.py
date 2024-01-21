import torch
from argus import Model
from argus.engine import State
from argus.utils import deep_detach, deep_to
from src.models.line.loss import EHMLoss
from src.models.line.model import HRNetHeatmap
from src.models.line.transforms import EHMPredictionTransform


class EHMMetaModel(Model):
    nn_module = HRNetHeatmap
    loss = EHMLoss
    optimizer = torch.optim.Adam
    prediction_transform = EHMPredictionTransform

    def __init__(self, params):
        super().__init__(params)
        self.amp = (False if 'amp' not in self.params
                    else bool(self.params['amp']))
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()
        batch = deep_to(batch, device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            prediction = self.nn_module(batch['image'])
        loss = self.loss(prediction, batch['keypoint_maps'])

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
            'target': deep_detach(batch['keypoint_maps']),
            'keypoints': deep_detach(batch['keypoints']),
            'line_para': deep_detach(batch['line_para']),
            'loss': loss.item()
        }

    def val_step(self, batch, state: State) -> dict:

        self.eval()
        with torch.no_grad():
            batch = deep_to(batch, device=self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(batch['image'])

            loss = self.loss(prediction, batch['keypoint_maps'])

            prediction = deep_detach(prediction)

            prediction = self.prediction_transform(prediction[-1])
            return {
                'prediction': prediction,
                'target': deep_detach(batch['keypoint_maps']),
                'keypoints': deep_detach(batch['keypoints']),
                'line_para': deep_detach(batch['line_para']),
                'loss': loss.item()
            }
