import json
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from dataset.datamodule import DataModule
from model.callbacks import get_callbacks
from model.net import Net
from pipeline.experiment_tracker import ExperimentTracker
from utils.data_manager import generate_label_map, load_df_from_csv


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # read dataframe
    train_df, val_df = load_df_from_csv(file_path=cfg.df_path)

    # generate label map
    label_map = generate_label_map(df=train_df)
    with open("outputs/label_map.json", "w") as f:
        json.dump(label_map, f)

    # set random seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    # experiment tracker
    experiment_tracker = ExperimentTracker(config=cfg)
    experiment_tracker.log_artifacts(f"config/{cfg.tracker.config_name}")
    experiment_tracker.log_artifacts(cfg.df_path)
    experiment_tracker.log_artifacts("outputs/label_map.json")

    # datamodule
    datamodule = DataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=val_df,
        label_map=label_map,
        config=cfg,
    )

    # model
    net = Net(label_map=label_map, config=cfg)

    # trainer
    trainer = pl.Trainer(
        logger=experiment_tracker.logger,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=get_callbacks(experiment_tracker=experiment_tracker, config=cfg),
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        auto_lr_find=cfg.trainer.auto_lr_find,
        deterministic=cfg.trainer.deterministic,
    )

    # train
    trainer.fit(net, datamodule=datamodule)

    # test
    trainer.test(net, datamodule=datamodule, ckpt_path="best")
    experiment_tracker.log_artifacts("outputs/classification_report.txt")
    experiment_tracker.log_artifacts("outputs/confusion_matrix.png")
    experiment_tracker.log_artifacts("outputs/roc_curve.png")

    # save artifacts
    ckpt_path = f"model/ckpt/{experiment_tracker.tracker_config.run_name}.ckpt"
    if os.path.exists(ckpt_path):
        experiment_tracker.log_artifacts(ckpt_path)


if __name__ == "__main__":
    main()