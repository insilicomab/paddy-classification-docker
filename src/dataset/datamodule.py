import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from dataset.dataset import ImageDataset
from dataset.transformation import Transforms


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_map: dict,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config

        self.x_train = train_df["image_path"].to_list()
        self.y_train = train_df["label"].map(label_map).to_list()

        self.x_val = val_df["image_path"].to_list()
        self.y_val = val_df["label"].map(label_map).to_list()

        self.x_test = test_df["image_path"].to_list()
        self.y_test = test_df["label"].map(label_map).to_list()

    def setup(self, stage=None) -> None:
        self.train_dataset = ImageDataset(
            root=self.config.root,
            image_path_list=self.x_train,
            label_list=self.y_train,
            transform=Transforms(config=self.config),
            phase="train",
        )
        self.val_dataset = ImageDataset(
            root=self.config.root,
            image_path_list=self.x_val,
            label_list=self.y_val,
            transform=Transforms(config=self.config),
            phase="val",
        )
        self.test_dataset = ImageDataset(
            root=self.config.root,
            image_path_list=self.x_test,
            label_list=self.y_test,
            transform=Transforms(config=self.config),
            phase="test",
        )

    def train_dataloader(self) -> DataLoader:
        if self.config.train_dataloader.imbalancedDatasetSampler:
            print("=== Imbalanced Dataset Sampler is Running ===")
            return DataLoader(
                self.train_dataset,
                sampler=ImbalancedDatasetSampler(self.train_dataset),
                batch_size=self.config.train_dataloader.batch_size,
                num_workers=self.config.train_dataloader.num_workers,
                pin_memory=self.config.train_dataloader.pin_memory,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.train_dataloader.batch_size,
                shuffle=self.config.train_dataloader.shuffle,
                num_workers=self.config.train_dataloader.num_workers,
                pin_memory=self.config.train_dataloader.pin_memory,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_dataloader.batch_size,
            shuffle=self.config.val_dataloader.shuffle,
            num_workers=self.config.val_dataloader.num_workers,
            pin_memory=self.config.val_dataloader.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_dataloader.batch_size,
            shuffle=self.config.test_dataloader.shuffle,
            num_workers=self.config.test_dataloader.num_workers,
            pin_memory=self.config.test_dataloader.pin_memory,
        )
