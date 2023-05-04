import random

import torch
from typing import Sequence
from PIL import ImageOps, ImageFilter, Image
from torchvision.transforms import transforms, InterpolationMode, ToTensor
from torchvision.transforms.functional import solarize
from abc import ABC, abstractmethod, abstractproperty

from sparsam.utils import min_max_normalize_tensor


class BaseMultiCropper(ABC):
    def __init__(self, n_global_crops: int, n_local_crops: int):
        self.n_crops = n_global_crops + n_local_crops
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops



class DinoAugmentationCropper(BaseMultiCropper):
    #  Adopted from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    def __init__(
        self,
        n_global_crops: int,
        n_local_crops: int,
        global_crops_scale: Sequence[int],
        local_crops_scale: Sequence[int],
        res=256,
    ):
        super().__init__(n_global_crops, n_local_crops)
        # In DataAugmentation in Dataset
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.2)], p=1.0
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(res, scale=global_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=1.0, radius_min=0.1, radius_max=5.0),
                ToTensor(),
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(res, scale=global_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=0.1, radius_min=0.1, radius_max=5.0),
                Solarization(0.2),
                ToTensor(),
            ]
        )
        # transformation for the local small crops
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.5),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(96, scale=local_crops_scale),
                flip_and_color_jitter,
                GaussianBlur(p=0.5, radius_min=0.1, radius_max=5.0),
                ToTensor(),
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.n_local_crops):
            crop = self.local_transfo(image)
            crops.append(crop)
        return crops


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        # TODO Lösung für Tensor
        blur_transforms = transforms.GaussianBlur(
            kernel_size=11, sigma=random.uniform(self.radius_min, self.radius_max)
        )
        return blur_transforms(img)
        # alt für imges
        # return img.filter(
        #     ImageFilter.GaussianBlur(
        #         radius=random.uniform(self.radius_min, self.radius_max)
        #     )
        # )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    # TODO: make it work
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)  # solarize(img, threshold=128)
        else:
            return img
