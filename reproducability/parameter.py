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

