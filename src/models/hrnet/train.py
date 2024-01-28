import os

import hydra
from argus import load_model
from torch import compile
from omegaconf import OmegaConf, DictConfig
from torch.backends import cudnn
from argus.callbacks import Checkpoint, EarlyStopping, LoggingToFile, \
    MonitorCheckpoint, ReduceLROnPlateau

from src.utils.argus_utils import load_compatible_weights
from src.models.hrnet.dataset import get_loader
from src.models.hrnet.metrics import L2metric, EvalAImetric
from src.models.hrnet.transforms import test_transform, train_transform

CONFIG_PATH = '/workdir/src/models/hrnet/train_config.yaml'


@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH),
            config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def train(cfg: DictConfig):
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    model = hydra.utils.instantiate(cfg.model)
    aug_params = cfg.data_params.augmentations
    train_trns = train_transform(
        brightness=aug_params.brightness,
        color=aug_params.color,
        contrast=aug_params.contrast,
        gauss_noise_sigma=aug_params.gauss_noise_sigma,
        prob=aug_params.prob
    )
    val_trns = test_transform()
    train_loader = get_loader(cfg.data.train, cfg.data_params,
                              train_trns, True)
    val_loader = get_loader(cfg.data.val, cfg.data_params, val_trns, False)
    experiment_name = cfg.metadata.experiment_name
    run_name = cfg.metadata.run_name
    save_dir = f'/workdir/data/experiments/{experiment_name}_{run_name}'
    metrics = [
        L2metric(num_keypoints=cfg.data_params.num_keypoints),
        EvalAImetric(hydra.utils.instantiate(cfg.camera),
                     img_size=cfg.data_params.input_size)
    ]
    callbacks = [
        EarlyStopping(patience=cfg.train_params.early_stopping_epochs,
                      monitor=cfg.train_params.monitor_metric,
                      better=cfg.train_params.monitor_metric_better),
        ReduceLROnPlateau(factor=cfg.train_params.reduce_lr_factor,
                          patience=cfg.train_params.reduce_lr_patience,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better),
        MonitorCheckpoint(save_dir, max_saves=1,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better,
                          optimizer_state=True),
        MonitorCheckpoint(save_dir, max_saves=1, monitor='val_evalai',
                          file_format='evalai-{epoch:03d}-{monitor:.6f}.pth',
                          better='max', optimizer_state=True),
        MonitorCheckpoint(save_dir, max_saves=1, monitor='val_pcks-5.0',
                          file_format='pcks-5.0-{epoch:03d}-{monitor:.6f}.pth',
                          better='max', optimizer_state=True),
        Checkpoint(save_dir, max_saves=3, file_format='save-{epoch:03d}.pth',
                   save_after_exception=True, optimizer_state=True, period=2),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
    ]

    pretrain_path = cfg.model.params.pretrain
    if pretrain_path is not None:
        if os.path.exists(pretrain_path):
            model_pretrain = load_model(pretrain_path,
                                        device=cfg.model.params.device)
            if cfg.train_params.load_compatible:
                model_pretrain = load_model(pretrain_path,
                                            device=cfg.model.params.device)
                model = load_compatible_weights(model_pretrain, model)
            else:
                model = load_model(pretrain_path,
                                   device=cfg.model.params.device)
            model.set_lr(cfg.model.params.optimizer.lr)
        else:
            raise ValueError(f'Pretrain {pretrain_path} does not exist')
    # Model may need tuning to find the optimal one for the particular model
    if cfg.train_params.use_compile:
        model.nn_module = compile(model.nn_module)
    model.fit(train_loader, val_loader=val_loader, metrics_on_train=False,
              num_epochs=cfg.train_params.max_epochs,
              callbacks=callbacks,
              metrics=metrics)


if __name__ == "__main__":
    train()
