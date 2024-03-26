import hydra
from omegaconf import DictConfig
from pytorch_lightning import callbacks

from pipeline.experiment_tracker import ExperimentTracker


def get_callbacks(experiment_tracker: ExperimentTracker, config: DictConfig) -> list:
    callback_list = []
    if config.callbacks.early_stopping.enable:
        earlystopping = callbacks.EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            patience=config.callbacks.early_stopping.patience,
            mode=config.callbacks.early_stopping.mode,
            verbose=True,
            strict=True,
        )
        callback_list.append(earlystopping)
    if config.callbacks.model_checkpoint.enable:
        model_checkpoint = callbacks.ModelCheckpoint(
            dirpath=hydra.utils.get_original_cwd() + "/model/ckpt/",
            filename=experiment_tracker.tracker_config.run_name,
            monitor=config.callbacks.model_checkpoint.monitor,
            mode=config.callbacks.model_checkpoint.mode,
            save_top_k=config.callbacks.model_checkpoint.save_top_k,
            save_last=config.callbacks.model_checkpoint.save_last,
        )
        callback_list.append(model_checkpoint)

    return callback_list
