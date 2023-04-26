import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torchvision.transforms import transforms, ToTensor

from sparsam.data_augmentation import BaseMultiCropper, Solarization, GaussianBlur
from sparsam.utils import min_max_normalize_tensor

MODEL = timm.models.xcit_small_12_p8_224_dist

CLASSIFIERS = [
    SVC(
        class_weight='balanced',
        C=1,
        gamma='scale',
        cache_size=10000,
        max_iter=-1,
        probability=True,
        break_ties=True,
        kernel='rbf',
    ),
    KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=8),
    LogisticRegression(
        class_weight='balanced',
        C=2.5,
        max_iter=10000,
        n_jobs=-1,
    ),
]


class DataCropperDINO(BaseMultiCropper):
    #  Adopted from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    def __init__(self, n_global_crops: int, n_local_crops: int, global_crops_scale, local_crops_scale, res=256):
        super(DataCropperDINO, self).__init__(n_global_crops, n_local_crops)
        # In DataAugmentation in Dataset
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.5, p=0.75),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(res, scale=global_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=1.0, radius_min=0.1, radius_max=5.0),
                ToTensor(),
            ]
        )

        # first global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(res, scale=global_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=0.1, radius_min=0.1, radius_max=5.0),
                Solarization(p=0.2),
                ToTensor(),
            ]
        )

        # transformation for the local small crops
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.5, p=0.75),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(res // 3, scale=local_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=0.5, radius_min=0.1, radius_max=5.0),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.extend(min_max_normalize_tensor(self.global_transfo1(image), 0, 1))
        crops.extend(min_max_normalize_tensor(self.global_transfo2(image), 0, 1))
        for _ in range(self.n_local_crops):
            crop = min_max_normalize_tensor(self.local_transfo(image), 0, 1)
            while torch.any(crop.isnan()):
                crop = torch.nan_to_num(crop, nan=0)
            crops.append(crop)
        return crops

