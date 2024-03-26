import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from dataset.transformation import TestTransforms


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_path_list: list,
        label_list: list,
        transform: Compose,
        phase: str,
    ) -> None:
        self.root = root
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.phase = phase
        self.transform = transform

    def get_labels(self):
        return np.array(self.label_list)

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        image = Image.open(os.path.join(self.root, self.image_path_list[index]))
        image = self.transform(self.phase, image)
        label = self.label_list[index]

        return image, label


class InferenceImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_path_list: list,
        label_list: list,
        transform: Compose,
    ) -> None:
        self.root = root
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, str, str]:
        image = Image.open(os.path.join(self.root, self.image_path_list[index]))
        image = self.transform(image)

        file_path = self.image_path_list[index]
        label = self.label_list[index]

        return image, file_path, label


def get_inference_dataloader(
    root: str, df_file_path: str, image_size: int
) -> DataLoader:
    # read test data
    df = pd.read_csv(df_file_path)
    image_path_list = df["image_id"].to_list()
    label_list = df["label"].to_list()

    # test dataset
    test_dataset = InferenceImageDataset(
        root=root,
        image_path_list=image_path_list,
        label_list=label_list,
        transform=TestTransforms(image_size=image_size),
    )

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_dataloader
