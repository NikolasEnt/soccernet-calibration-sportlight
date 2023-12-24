"""
To run model validation:
`python validate.py --config-name val_config`

To run optimization of camera model params with Optuna:
`python validate.py --config-name optimize_valid --multirun`
"""

import os
import time
import warnings

import hydra
import mlflow
from argus import load_model
from torch import compile
from omegaconf import DictConfig

from src.models.hrnet.dataset import get_loader
from src.models.hrnet.metrics import L2metric, EvalAImetric
from src.models.hrnet.transforms import test_transform

CONFIG_PATH = '/workdir/src/models/hrnet/val_config.yaml'

monitors = ['val_l2', 'val_loss', 'val_pcks-5.0' 'val_completeness',
            'val_eval_accuracy', 'val_l2_reprojection', 'val_evalai']


@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH))
def validate(cfg: DictConfig) -> float:
    res = 0.0
    metrics = {}
    model = hydra.utils.instantiate(cfg.model)
    val_trns = test_transform()
    val_loader = get_loader(cfg.data.val, cfg.data_params, val_trns, False)
    experiment_name = cfg.metadata.experiment_name
    run_name = cfg.metadata.run_name+'_val'
    description = cfg.metadata.description
    mlflow.set_tracking_uri(os.environ['MLFLOW_URI'])
    mlflow.set_experiment(experiment_name)
    metrics_list = [
        L2metric(num_keypoints=cfg.data_params.num_keypoints),
        EvalAImetric(hydra.utils.instantiate(cfg.camera),
                     img_size=cfg.data_params.input_size)
    ]

    with mlflow.start_run(run_name=run_name, description=description):
        try:
            mlflow.log_params({**cfg.data_params, **cfg.camera})
            mlflow.log_params({'val.images': len(val_loader.dataset)})
            mlflow.set_tag('Experimenter', cfg.metadata.experimenter)
            mlflow.set_tag('device', cfg.model.params.device)
            if 'location' in cfg.metadata:
                mlflow.set_tag('Location', cfg.metadata.location)
            mlflow.log_artifact(CONFIG_PATH)
        except mlflow.MlflowException as e:
            warnings.warn(f'Error in saving artifacts to MLflow: {e}.')

        pretrain_path = cfg.model.params.pretrain
        if os.path.exists(pretrain_path):
            model = load_model(pretrain_path,
                               device=cfg.model.params.device)
            mlflow.log_param('pretrain', pretrain_path)
        else:
            raise ValueError(f'Pretrain {pretrain_path} does not exist')
        # Mode may need tuning to find the optimal one for the particular model
        if cfg.train_params.use_compile:
            model.nn_module = compile(model.nn_module)
        s_time = time.perf_counter()
        metrics = model.validate(val_loader=val_loader, metrics=metrics_list)
        print(f'Validation complete in {time.perf_counter()-s_time:.4f} s.')
        metrics = {k: float(v) for k, v in metrics.items()}
        for k in list(metrics.keys()):
            if k in monitors:
                metrics[f'{k}_best'] = metrics[k]
        mlflow.log_metrics(metrics=metrics, step=0)
        # To avoid problems with CUDA in the case of parallel execution
        model.nn_module.to('cpu')
        del model
        del metrics_list
    if 'val_evalai' in metrics:
        res = metrics['val_evalai']
    time.sleep(3)
    return res


if __name__ == "__main__":
    validate()
