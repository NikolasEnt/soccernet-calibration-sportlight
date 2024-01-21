import os
from typing import Callable, Optional

import hydra
from argus import load_model
from argus.callbacks import (EarlyStopping, LoggingToFile, MonitorCheckpoint,
                             ReduceLROnPlateau)
from omegaconf import DictConfig
from src.models.line.dataset import EHMDataset
from src.models.line.metamodel import EHMMetaModel, EHMPredictionTransform
from src.models.line.metrics import AccMetric
from src.models.line.transforms import test_transform, train_transform
from torch.utils.data import DataLoader

CONFIG_PATH = '/workdir/src/models/line/train_config.yaml'


def get_loader(dataset_path: str, data_params: DictConfig,
               transform: Optional[Callable] = None, shuffle: bool = False)\
        -> DataLoader:
    dataset = EHMDataset(dataset_path,
                         stride=data_params.stride,
                         sigma=data_params.sigma,
                         transform=transform)
    loader = DataLoader(
        dataset, batch_size=data_params.batch_size,
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        shuffle=shuffle)
    return loader


@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH),
            config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def train(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    train_trns = train_transform()
    val_trns = test_transform()
    train_loader = get_loader(cfg.data.train, cfg.data_params,
                              train_trns, True)
    val_loader = get_loader(cfg.data.val, cfg.data_params, val_trns, False)
    experiment_name = cfg.metadata.experiment_name
    run_name = cfg.metadata.run_name

    save_dir = f'/workdir/data/experiments/{experiment_name}_{run_name}'
    metrics = [AccMetric(cfg.data_params.num_keypoint_pairs)]
    callbacks = [
        EarlyStopping(patience=cfg.train_params.early_stopping_epochs,
                      monitor=cfg.train_params.monitor_metric,
                      better=cfg.train_params.monitor_metric_better),
        ReduceLROnPlateau(factor=cfg.train_params.reduce_lr_factor,
                          patience=cfg.train_params.reduce_lr_patience,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better),
        MonitorCheckpoint(save_dir, max_saves=3,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
    ]

    pretrain_path = cfg.model.params.pretrain
    if pretrain_path is not None:
        if os.path.exists(pretrain_path):
            model = load_model(pretrain_path,
                               device=cfg.model.params.device)
            model.set_lr(cfg.model.params.optimizer.lr)
        else:
            raise ValueError(f'Pretrain {pretrain_path} does not exist')

    # Mode may need tuning to find the optimal one for the particular model
    # model.nn_module = compile(model.nn_module)
    model.fit(train_loader, val_loader=val_loader, metrics_on_train=False,
              num_epochs=cfg.train_params.max_epochs,
              callbacks=callbacks,
              metrics=metrics)


if __name__ == "__main__":
    train()
