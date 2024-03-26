import shutil

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from dataset.datamodule import DataModule
from model.net import Net
from utils.data_manager import generate_label_map, load_df_from_csv


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # read dataframe
    train_df, val_df = load_df_from_csv(file_path=cfg.df_path)

    # generate label map
    label_map = generate_label_map(df=train_df)

    # set random seed
    pl.seed_everything(seed=cfg.seed, workers=True)

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
        logger=False,
        max_epochs=1,
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

    # remove checkpoints
    shutil.rmtree("checkpoints/")


if __name__ == "__main__":
    main()