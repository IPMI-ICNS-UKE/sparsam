import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Callable, List, Tuple, Any, Sequence

import pydicom
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from sparsam.data_augmentation import DinoAugmentationCropper
from sparsam.utils import min_max_normalize_tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseSet(ABC, Dataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        :param index: which datapoint from the dataset to get
        return:
        img: the loaded and processed image
        label: if dataset is labeled returns the corresponding image label or dummy label/ None
        """
        pass


class ImageSet(Dataset):
    def __init__(
        self,
        img_paths: Sequence[Path],
        labels: Sequence = None,
        img_size: int | Sequence[int] = (256, 256),
        data_augmentation: Callable = False,
        class_names: Sequence[str] = None,
        normalize: bool = False,
    ):
        self.img_paths = img_paths
        self.labels = labels
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = sorted(list(set(labels)))
        self.data_augmentation = data_augmentation
        if not isinstance(img_size, tuple):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.normalize = normalize

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        path = self.img_paths[index]
        if path.suffix == '.dcm':
            ds = pydicom.dcmread(path)
            img = Image.fromarray(ds.pixel_array, 'RGB')
        else:
            img = Image.open(path)

        img = img.resize(self.img_size, Image.NEAREST)
        if self.data_augmentation:
            img = self.data_augmentation(img)
        if self.normalize:
            img = to_tensor(img)
            img = min_max_normalize_tensor(img, 0, 1)
        if self.labels is not None:
            label = self.labels[index]
            if self.class_names is not None:
                label = self.class_names.index(label)
        else:
            label = 0
        return img, label

