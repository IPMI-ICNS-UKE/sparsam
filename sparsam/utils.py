from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from functools import partial
from os import PathLike
from pathlib import Path
from tqdm import tqdm
from typing import Callable, List, Iterable, Tuple, Any
import json

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.amp import autocast
from torch.optim import Optimizer
from torch.utils.data.dataset import Dataset

from sparsam.helper import trunc_normal_, recursive_dict


class ModelMode(Enum):
    EXTRACT_FEATURES = 1
    CLASSIFICATION = 2


@torch.no_grad()
@autocast('cuda')
def model_inference(
    data_loader: Iterable, model: nn.Module, mode: ModelMode, device: str | int = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = []
    labels = []
    for batch in tqdm(data_loader):
        images, label = batch
        images = images.to(device)
        if mode == ModelMode.EXTRACT_FEATURES:
            feature = model.forward_features(images)
        if mode == ModelMode.CLASSIFICATION:
            feature = torch.softmax(model(images), dim=-1)
        if feature.ndim == 3:
            feature = feature[:, 0]
        features.extend(list(feature.detach().cpu().numpy()))
        labels.extend(list(label.detach().numpy()))
    return np.array(features), np.array(labels)


class EmaTeacherUpdate:
    def __init__(self, momentum: float | Callable = 0.9):
        self.momentum = momentum

    def __call__(self, teacher: nn.Module, student: nn.Module, iteration: int = None):
        with torch.no_grad():
            if callable(self.momentum):
                momentum = self.momentum(iteration)
            else:
                momentum = self.momentum
            for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                teacher_p.data.mul_(momentum).add_((1 - momentum) * student_p.detach().data)


class CosineScheduler:
    # Adopted from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    def __init__(
        self,
        final_value: float,
        base_value: float,
        total_iterations: int,
        warm_up_iterations: int = 0,
        warm_up_starting_value: float = 0.0,
    ):
        warmup_schedule = np.array([])
        if warm_up_iterations:
            warmup_schedule = np.linspace(warm_up_starting_value, base_value, warm_up_iterations)
        cos_iters = np.arange(total_iterations - warm_up_iterations)

        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * cos_iters / len(cos_iters)))

        self.schedule = np.concatenate((warmup_schedule, schedule))
        self.step = 0

    def __call__(self, step=None) -> float:
        if step is not None:
            self.step = step

        if self.step < len(self.schedule):
            value = self.schedule[self.step]
        else:
            value = self.schedule[-1]

        self.step += 1

        return value


class BaseScheduler(ABC):
    @abstractmethod
    def step(self, *args, **kwargs) -> float:
        """Needs to implement step function for manipulation"""
        pass


class SimpleScheduler(BaseScheduler):
    def __init__(self, scheduler: Callable):
        self.scheduler = scheduler
        self.current_lr = None


class OptimizerScheduler(SimpleScheduler):
    def __init__(self, scheduler: Callable, optimizer: object = None):
        super().__init__(scheduler=scheduler)
        self.optimizer = optimizer


class LRScheduler(OptimizerScheduler):
    def step(self, step: int = None, *args, **kwargs) -> float:
        lr = self.scheduler(step)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr
        return lr


class DinoWdScheduler(OptimizerScheduler):
    def step(self, step: int = None, *args, **kwargs) -> float:
        for i, param_group in enumerate(self.optimizer.param_groups):
            wd = self.scheduler(step)
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd
                break


class GradClipWrapper:
    def __init__(self, grad_clip_function: Callable):
        self.grad_clip_f = grad_clip_function

    def __call__(self, model: nn.Module, step: int = None, *args, **kwargs):
        self.grad_clip_f(model.parameters(), *args, **kwargs)


class DinoGradClipper:
    def __init__(self, freeze_last_layer_iterations: int = None, clip_factor: float = 0.32):
        self.freeze_last_layer_iterations = freeze_last_layer_iterations
        self.step = 0
        self.grad_clipper = partial(timm.utils.adaptive_clip_grad, clip_factor=clip_factor, eps=1e-3, norm_type=2.0)

    def __call__(self, model: nn.Module, step: int = None):
        if step:
            self.step = step

        self.grad_clipper(parameters=model.parameters())

        if self.freeze_last_layer_iterations:
            self._cancel_gradients_last_layer(model)
        self.step += 1

    def _cancel_gradients_last_layer(self, model: nn.Module):
        if self.step < self.freeze_last_layer_iterations:
            for n, p in model.named_parameters():
                if "last_layer" in n:
                    if p.requires_grad:
                        p.grad = None


class BaseLogger(ABC):
    @abstractmethod
    def log(self, logs: dict, step: int | str):
        """logs a dict with key value pairs"""
        pass


class DummyLogger(BaseLogger):
    @staticmethod
    def log(logs: dict, *args, **kwargs):
        pass


class JsonLogger(BaseLogger):
    def __init__(self, save_path: PathLike):
        self.save_path = Path(save_path) / f'log_{datetime.now().strftime("%Y-%m-%d-%H-%M-%s")}.json'
        self.logger = recursive_dict()

    def log(self, logs: dict, step: int | str):
        if not step in list(self.logger.keys()):
            self.logger[step] = logs
        else:
            self.logger[step].update(logs)

        with open(self.save_path, 'w') as h:
            json.dump(self.logger, h)


class EarlyStopper:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0, warm_up=1000, decision_function=lambda x, y: x > y):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.warm_up = warm_up
        self.warm_up_counter = 0
        self.counter = 0
        self.best_value = None
        self.min_delta = min_delta
        self.decision_function = decision_function

    def __call__(self, current_value):
        early_stop = False
        if self.warm_up_counter < self.warm_up:
            self.warm_up_counter += 1
        elif self.best_value is None:
            self.best_value = max(current_value, 1e-5)
        elif self.decision_function(current_value, self.best_value):
            rel_improvement = abs(self.best_value - current_value) / self.best_value
            if rel_improvement > self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
        return early_stop


class ProjectionHead(nn.Module):
    # Adopted from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
    ):
        super().__init__()
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropModelWrapper(nn.Module):
    # Adopted from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    """dino_aug_parameter
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(
        self, backbone: nn.Module, projection_head: nn.Module = None, mode: ModelMode = ModelMode.EXTRACT_FEATURES
    ):
        super().__init__()
        # disable layers dedicated to ImageNet labels classification
        if mode == ModelMode.EXTRACT_FEATURES:
            backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = copy.deepcopy(backbone)
        self.projection_head = projection_head

    def forward(self, x: List[Tensor] | Tensor) -> Tensor:
        # convert to list
        if not isinstance(x, list):
            x = [x]

        # return start, end idxs for same sized crops
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx, output = 0, torch.empty(0).to(x[0].device)
        output = []
        for end_idx in idx_crops:
            temp_out = self.backbone(torch.cat(x[start_idx:end_idx], dim=0))
            output.append(temp_out)
            start_idx = end_idx
        output = torch.cat(output)
        if self.projection_head:
            output = self.projection_head(output)
        return output

    def forward_features(self, x: Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x



class MultiCropDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int) -> Tensor | Tuple[Tensor, Tuple[Any]]:
        data = self.dataset.__getitem__(idx)
        if not isinstance(data, tuple):
            image = data
            data = None
        else:
            # assumes image to be the first return value
            image = data[0]
            data = data[1:]
        n_views = len(image)
        data = [[d] * n_views for d in data]
        return image, *data

    def __len__(self):
        return self.dataset.__len__()


def min_max_normalize_tensor(img: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    img = img.clamp(min=min_value, max=max_value)
    img = img - min_value
    img = img / (max_value - min_value)
    img = (img * 2) - 1
    return img


def optimizer_to_device(optimizer: Optimizer, device: int | str):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
