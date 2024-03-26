import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms


class Nonetransform:
    def __call__(self, image):
        return image


class Transforms:
    def __init__(self, config: DictConfig) -> None:
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomCrop(
                        (
                            config.train_transform.random_crop.image_size,
                            config.train_transform.random_crop.image_size,
                        )
                    )
                    if config.train_transform.random_crop.enable
                    else Nonetransform(),
                    transforms.RandAugment(
                        config.train_transform.randaugment.num_ops,
                        config.train_transform.randaugment.magnitude,
                    )
                    if config.train_transform.randaugment.enable
                    else Nonetransform(),
                    transforms.TrivialAugmentWide()
                    if config.train_transform.trivial_augment_wide.enable
                    else Nonetransform(),
                    transforms.AugMix(
                        severity=config.train_transform.augmix.severity,
                        mixture_width=config.train_transform.augmix.mixture_width,
                        chain_depth=config.train_transform.augmix.chain_depth,
                        alpha=config.train_transform.augmix.alpha,
                        all_ops=config.train_transform.augmix.all_ops,
                    )
                    if config.train_transform.augmix.enable
                    else Nonetransform(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        config.train_transform.normalize.mean,
                        config.train_transform.normalize.std,
                    )
                    if config.train_transform.normalize.enable
                    else Nonetransform(),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.CenterCrop(
                        (
                            config.test_transform.center_crop.image_size,
                            config.test_transform.center_crop.image_size,
                        )
                    )
                    if config.test_transform.center_crop.enable
                    else Nonetransform(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        config.test_transform.normalize.mean,
                        config.test_transform.normalize.std,
                    )
                    if config.test_transform.normalize.enable
                    else Nonetransform(),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.CenterCrop(
                        (
                            config.test_transform.center_crop.image_size,
                            config.test_transform.center_crop.image_size,
                        )
                    )
                    if config.test_transform.center_crop.enable
                    else Nonetransform(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        config.test_transform.normalize.mean,
                        config.test_transform.normalize.std,
                    )
                    if config.test_transform.normalize.enable
                    else Nonetransform(),
                ]
            ),
        }

    def __call__(self, phase: str, img: Image) -> torch.Tensor:
        return self.data_transform[phase](img)


class TestTransforms:
    def __init__(self, image_size: int):
        self.data_transform = transforms.Compose(
            [
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img: Image) -> torch.Tensor:
        return self.data_transform(img)


__all__ = ["Transforms", "TestTransforms"]
