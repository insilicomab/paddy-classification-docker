from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset.dataset import InferenceImageDataset
from model.model import get_model

CUDA_IS_AVAILABLE = torch.cuda.is_available()


def load_model_weights(
    config: DictConfig, download_root: str, model_path: str
) -> torch.nn.Module:
    _model_state_dict = torch.load(Path(download_root) / Path(model_path))["state_dict"]
    model_state_dict = {
        k.replace("model.", ""): v for k, v in _model_state_dict.items()
    }
    model = get_model(config)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def inference(
    dataloader: DataLoader, model: torch.nn.Module, int_to_label: dict
) -> pd.DataFrame:
    if CUDA_IS_AVAILABLE:
        model.cuda()
    paths, preds = [], []
    with torch.no_grad():
        for image, path, _ in tqdm(dataloader):
            if CUDA_IS_AVAILABLE:
                image = image.cuda()
            logits = model(image)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().detach().numpy()
            pred = [int_to_label[i] for i in pred]
            paths.extend(path)
            preds.extend(pred)
    df = pd.DataFrame({"image_id": paths, "label": preds})
    return df


class TestTimeAugmentationInference:
    def __init__(self) -> None:
        pass

    def setup_tta_transforms(self, image_size: int) -> list[Compose]:
        tta_transforms = [
            transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.functional.hflip,
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.functional.vflip,
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        ]
        return tta_transforms

    def _get_inference_dataloader(
        self,
        root: str,
        df_file_path: str,
        transform: Compose,
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
            transform=transform,
        )
        # dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return test_dataloader

    def _inference(
        self, dataloader: DataLoader, model: torch.nn.Module
    ) -> tuple[np.ndarray, list]:
        if CUDA_IS_AVAILABLE:
            model.cuda()
        preds, paths = [], []
        with torch.no_grad():
            for image, path, _ in tqdm(dataloader):
                if CUDA_IS_AVAILABLE:
                    image = image.cuda()
                logits = model(image)
                preds_proba = F.softmax(logits, dim=1)
                preds.append(preds_proba.cpu().numpy())
                paths.extend(path)
        preds = np.concatenate(preds)
        return preds, paths

    def inference_tta(
        self,
        root: str,
        df_file_path: str,
        model: torch.nn.Module,
        int_to_label: dict,
        tta_transforms: list[Compose],
    ) -> pd.DataFrame:
        with torch.no_grad():
            preds_for_tta = []
            for transform in tqdm(tta_transforms):
                test_dataloader = self._get_inference_dataloader(
                    root=root, df_file_path=df_file_path, transform=transform
                )
                preds, paths = self._inference(dataloader=test_dataloader, model=model)
                preds_for_tta.append(preds)

            tta_preds_mean = np.mean(preds_for_tta, axis=0)
            tta_preds = np.argmax(tta_preds_mean, axis=1)
            tta_preds_label = [int_to_label[i] for i in tta_preds]

        df = pd.DataFrame({"image_id": paths, "label": tta_preds_label})

        return df
