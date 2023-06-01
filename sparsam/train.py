from datetime import datetime
import os
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, List, Iterable

import timm.utils

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor, TensorType
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from sparsam.data_augmentation import DinoAugmentationCropper
from sparsam.helper import get_params_groups
from sparsam.loss import DINOLoss
from sparsam.utils import (
    ProjectionHead,
    MultiCropModelWrapper,
    EmaTeacherUpdate,
    DummyLogger,
    BaseLogger,
    DinoGradClipper,
    CosineScheduler,
    LRScheduler,
    DinoWdScheduler,
    model_inference,
    optimizer_to_device,
    BaseScheduler,
    ModelMode,
    MultiCropDatasetWrapper,
)
from sparsam.dataset import BaseSet


class BaseGym(ABC):
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader | None,
            loss_function: Callable,
            optimizer: Optimizer = None,
            device: int | str = 'cuda',
            lr_scheduler: BaseScheduler = None,
            weight_decay_scheduler: BaseScheduler = None,
            grad_clipper: Callable = None,
            n_trainings_epochs: int = 100,
            starting_step: int = 0,
            metrics: Callable | List[Callable] = balanced_accuracy_score,
            metrics_parameters: dict | List[dict] = None,
            metrics_require_probabilities: bool | List[bool] = False,
            model_saving_frequency: int = 250,
            eval_frequency: int = None,
            save_path: Path = None,
            logger: BaseLogger = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss = loss_function

        self.scaler = GradScaler()
        self.grad_clipper = grad_clipper
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = weight_decay_scheduler

        self.n_trainings_epochs = n_trainings_epochs
        self.starting_step = starting_step
        self.device = device

        self.logger = logger or DummyLogger

        metrics_parameters = metrics_parameters or {}
        if not isinstance(metrics, list):
            metrics = [metrics]
        if not isinstance(metrics_parameters, list):
            metrics_parameters = [metrics_parameters] * len(metrics)
        self.metrics = [partial(metric, **params) if not isinstance(metric, partial)
                        else metric for metric, params in zip(metrics, metrics_parameters)]
        if not isinstance(metrics_require_probabilities, list):
            metrics_require_probabilities = [metrics_require_probabilities] * len(self.metrics)
        self.metrics_require_probabilities = metrics_require_probabilities

        if save_path:
            self.save_path = Path(save_path) / datetime.now().strftime("%Y-%m-%d-%H-%M-%s")
            self.save_path.mkdir()
        else:
            self.save_path = False
        self.eval_f = eval_frequency or float('Nan')
        self.save_f = model_saving_frequency

    @abstractmethod
    def train(self):
        """implements the actual training and validation algorithm"""
        pass

    @abstractmethod
    def _predict_val_samples(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """Implements the extraction of validation prediction_probabilities and validation labels and returns them"""
        pass

    def _eval_model(self, model: nn.Module) -> dict:
        predictions_prob, val_labels = self._predict_val_samples(model)
        if np.allclose(predictions_prob.sum(-1), 1):
            predictions = np.argmax(predictions_prob, axis=-1)
        else:
            predictions = np.round(predictions_prob)
        metric_dict = dict()
        for metric, requires_probability in zip(self.metrics, self.metrics_require_probabilities):
            if isinstance(metric, partial):
                metric_name = metric.func.__name__
            else:
                metric_name = metric.__name__
            if requires_probability:
                metric_dict[metric_name] = metric(val_labels, predictions_prob)
            else:
                metric_dict[metric_name] = metric(val_labels, predictions)
        return metric_dict

    def _model_backprop(self, loss: TensorType):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.grad_clipper:
            self.grad_clipper(self.model)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(True)

    def _update_lr_wd(self, step=None, *args, **kwargs):
        if self.lr_scheduler:
            self.lr_scheduler.step(step, *args, **kwargs)
        if self.wd_scheduler:
            self.wd_scheduler.step(step, *args, **kwargs)

    @staticmethod
    def _save_state_dict(model: nn.Module | Optimizer, model_path: Path):
        state_dict = model.state_dict()
        torch.save(state_dict, model_path)

    def _init_tqdm_bar(self) -> tqdm:
        starting_epoch = self.starting_step // len(self.train_loader)
        last_step = self.starting_step + len(self.train_loader) * (self.n_trainings_epochs - starting_epoch)
        epoch_bar = tqdm(
            range(self.starting_step, last_step),
            initial=self.starting_step,
            total=last_step,
        )
        return epoch_bar


class StudentTeacherGym(BaseGym):
    def __init__(
            self,
            student_model: nn.Module,
            train_loader: DataLoader,
            loss_function: Callable,
            teacher_model: nn.Module = None,
            teacher_update_function: Callable = None,
            student_slicing: slice = slice(0, -1, 1),
            teacher_slicing: slice = slice(0, 2, 1),
            optimizer: Optimizer = None,
            lr_scheduler: BaseScheduler = None,
            weight_decay_scheduler: BaseScheduler = None,
            grad_clipper: Callable = None,
            device: int | str = 'cuda',
            n_trainings_epochs: int = 100,
            starting_step: int = 0,
            val_loader: DataLoader = None,
            labeled_train_loader: DataLoader = None,
            classifier: ClassifierMixin = KNeighborsClassifier(),
            eval_frequency: int = None,
            model_saving_frequency: int = 250,
            metrics: Callable | List[Callable] = balanced_accuracy_score,
            metrics_parameters: dict | List[dict] = None,
            metrics_require_probabilities: bool | List[bool] = False,
            save_path: Path = None,
            logger: BaseLogger = None,
    ):
        """
        :param student_model: Complete student model (backbone + projection head)
        :param train_loader: Dataloader for the Self Supervised trainings data must return 2 values: img, meta.
        img maybe a list or tensor and ist then sliced according to the provided slicing for different student and
        teacher views
        :param loss_function: criterion to be optimized
        :param teacher_model: the teacher model, if None is provided the student architecture is used.
        :param teacher_update_function: a function on how to update the teacher. Must be callable and take the
        following arguments: teacher_update_function(teacher, student)
        :param student_slicing: often different views are used for student and teacher. This slicing is used to provide
        views for the student model
        :param teacher_slicing: teacher model slicing. Fore more info see student_slicing
        :param optimizer: the already initialized optimizer for the student
        :param lr_scheduler: a callable lr scheduler, must take step argument and return lr
        :param weight_decay_scheduler: callable weight decay scheduler, must take step argument and return wd
        :param grad_clipper: callable that online clips gradients, must take parameters as input
        :param device: the device to train on: either GPU index or "cpu"/ "cuda"
        :param n_trainings_epochs: How many epochs to train
        :param starting_step: if the training is continued from where to start
        :param val_loader: if provided online validation will be performed, requires labeled_train_loader and classifier
        :param labeled_train_loader: if val_loader is provided this provides the trainings data
        :param classifier: classifier to use for val data, must follow scikit learn api
        :param eval_frequency: How often to evaluate
        :param model_saving_frequency: How often to save the model
        :param metrics: Which metrics to use, should be callable and follow scikit learn api (Label, pred). If a list is
        provided, multiple metrics are calculated.
        :param metrics_parameters: additional parameter used while calculation the metrics. Should be a dict or list of
        dicts, if multiple metrics
        :param metrics_require_probabilities: whether to use probabilities or predictions for a given metric
        :param save_path: where to save the models/ optimizer to
        :param logger: Object to log loss and metrics. Must provide log function and take step argument. wanndb works
        """
        super().__init__(
            model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            lr_scheduler=lr_scheduler,
            weight_decay_scheduler=weight_decay_scheduler,
            grad_clipper=grad_clipper,
            n_trainings_epochs=n_trainings_epochs,
            starting_step=starting_step,
            metrics=metrics,
            metrics_parameters=metrics_parameters,
            metrics_require_probabilities=metrics_require_probabilities,
            eval_frequency=eval_frequency,
            model_saving_frequency=model_saving_frequency,
            save_path=save_path,
            logger=logger,
        )

        self.train_loader = train_loader
        if labeled_train_loader and not val_loader:
            Warning(
                'Evaluation will be skipped: '
                'Labeled training data has no effect, since no validation data is provided'
            )
        if not labeled_train_loader and val_loader:
            Warning(
                'Evaluation will be skipped: '
                'Validation training data has no effect, since no labeled training data is provided'
            )
        self.labeled_train_loader = labeled_train_loader
        self.val_loader = val_loader

        self.classifier = classifier

        self.loss = loss_function

        self.student_model = self.model
        self.teacher_model = teacher_model or deepcopy(student_model)
        self.student_slicing = student_slicing
        self.teacher_slicing = teacher_slicing
        self.teacher_update = teacher_update_function

    def train(self) -> Tuple[nn.Module, nn.Module]:
        self.student_model.to(self.device)
        self.teacher_model.to(self.device)
        optimizer_to_device(self.optimizer, self.device)
        if isinstance(self.loss, nn.Module):
            self.loss.to(self.device)

        epoch_bar = self._init_tqdm_bar()
        while epoch_bar.n < epoch_bar.total:
            for iteration, batch in enumerate(self.train_loader):
                step = epoch_bar.n
                self.optimizer.zero_grad(True)
                images = self._prepare_trainings_batch(batch)
                loss = self._model_update(images)
                if not isinstance(loss, dict):
                    loss = dict(loss=loss)
                self._update_lr_wd(step=step)

                if step % self.eval_f == 0 and self.val_loader and self.labeled_train_loader:
                    metrics_dict = self.eval_student_teacher()
                    self.logger.log(metrics_dict, step=step)
                # TODO save best model
                if self.save_path and step % self.save_f == 0:
                    self._save_training_state(step)

                self.logger.log(dict(loss=loss), step=step)
                epoch_bar.update(1)
                epoch_bar.set_description(desc=f"loss={loss['loss']:.4f}")
        self.student_model.to('cpu')
        self.teacher_model.to('cpu')
        return self.student_model, self.teacher_model

    def _model_update(self, images: Tensor | List) -> float:
        teacher_out, student_out = self._teacher_student_forward(images)
        with autocast():
            loss = self.loss(student_out, teacher_out)
        self._model_backprop(loss)
        self.teacher_update(
            self.teacher_model,
            self.student_model,
        )
        return loss.item()

    @autocast()
    def _teacher_student_forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        self.teacher_model.train()
        self.student_model.train()
        with torch.no_grad():
            teacher_out = self.teacher_model(images[self.teacher_slicing])
        student_out = self.student_model(images[self.student_slicing])
        return teacher_out, student_out

    def _prepare_trainings_batch(
            self, batch: Tuple[any, List[Tensor] | Tensor] | List[Tensor]
    ) -> List[Tensor] | Tensor:
        # assumes first elem of return values to be the image or list of images
        if isinstance(batch, list) and not isinstance(batch[0], Tensor):
            images, _ = batch
        else:
            images = batch
        if isinstance(images, list):
            images = [views.to(self.device) for views in images]
        else:
            images = images.to(self.device)
        return images

    def eval_student_teacher(self):
        models = [self.student_model, self.teacher_model]
        keys = ['student', 'teacher']
        results = dict()
        for model, key in zip(models, keys):
            metric_values = self._eval_model(model)
            results[key] = metric_values
        return results

    @torch.no_grad()
    @autocast()
    def _extract_features(self, data_loader: Iterable, model: nn.Module):
        return model_inference(
            data_loader=data_loader, model=model, mode=ModelMode.EXTRACT_FEATURES, device=self.device
        )

    def _predict_val_samples(self, model: nn.Module) -> Tuple[np.array, np.array]:
        train_features, train_labels = self._extract_features(self.labeled_train_loader, model)
        val_features, val_labels = self._extract_features(self.val_loader, model)
        self.classifier.fit(train_features, train_labels)
        predictions_prob = np.asarray(self.classifier.predict_proba(val_features))
        if predictions_prob.ndim > 2:
            predictions_prob = predictions_prob[..., -1].transpose()
        return predictions_prob, val_labels

    def _save_training_state(self, step: int | str):
        step = str(step)
        save_path = self.save_path / step
        save_path.mkdir()
        self._save_state_dict(self.student_model, save_path / f'student_{step}.pt')
        self._save_state_dict(self.teacher_model, save_path / f'teacher_{step}.pt')
        self._save_state_dict(self.optimizer, save_path / f'optimizer_{step}.pt')
        self._save_state_dict(self.loss, save_path / f'loss_{step}.pt')
        self._save_state_dict(self.scaler, save_path / f'scaler_{step}.pt')


def create_dino_gym(
        # if Dataset is provided, standard Dino DataAugmentation is applied. train_loader parameter must be given
        unalabeled_train_set: BaseSet,
        labeled_train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        backbone_model: nn.Module = None,  # default: XCiT_small_12_p8
        classifier: ClassifierMixin = None,
        n_trainings_epochs: int = 250,
        save_path: Path = None,
        device: int | str = 'cuda',
        eval_frequency: int = 100,
        model_saving_frequency: int = 250,
        logger: BaseLogger = None,
        metrics: Callable | List[Callable] = balanced_accuracy_score,
        metrics_parameters: dict | List[dict] = None,
        metrics_requires_probability: bool | List[bool] = False,
        # only used in combination with train_set
        image_resolution: int = None,  # default is 224
        # if train_loader is given it is assumed the loader returns augmented images. This omits all other DA!
        unlabeled_train_loader_parameters: dict = None,
        unlabeled_train_loader: DataLoader = None,
        # used to resume training
        resume_training_from_checkpoint: int | Path | str = False,
        optimizer_state_dict: dict = None,
        # more optional parameters, that might be useful in special cases, but are generally well performing with defaults
        n_global_crops: int = 2,
        n_local_crops: int = 5,
        student_slicing: slice = slice(None),
        teacher_slicing: slice = slice(0, 2, 1),
        global_crops_scale: Tuple[float, float] = (0.5, 1),
        local_crops_scale: Tuple[float, float] = (0.1, 0.5),
        projection_head_out_dim: int = 65536,
        projection_head_hidden_dim: int = 2048,
        projection_head_bottleneck_dim: int = 256,
        projection_head_n_layers: int = 4,
        warmup_teacher_temp: float = 0.02,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_iterations: int = None,  # default: 2 epochs
        student_temp=0.1,
        center_momentum=0.9,
        grad_clip_factor: float = 0.32,
        freeze_last_layer_iterations: int = None,  # default: 1 epoch
        teacher_momentum: float = 0.9995,
        final_lr: float = 0,
        lr_scheduler_warm_up_iterations: int = None,  # default: 2 epochs
        final_weight_decay: float = 0.4,
        optimizer: Optimizer = AdamW,  # default: AdamW
        optimizer_parameters: dict = None,  # default lr: 0.0005, wd: 0.04
        grad_clipper: Callable = None,  # default adaptive grad clipping
) -> StudentTeacherGym:
    """
    :param unalabeled_train_set:
    :param labeled_train_loader:
    :param val_loader:
    :param backbone_model:
    :param classifier:
    :param n_trainings_epochs:
    :param save_path:
    :param device:
    :param eval_frequency:
    :param model_saving_frequency:
    :param logger:
    :param metrics:
    :param metrics_parameters:
    :param metrics_requires_probability:
    :param image_resolution:
    :param unlabeled_train_loader_parameters:
    :param unlabeled_train_loader:
    :param resume_training_from_checkpoint:
    :param optimizer_state_dict:
    :param n_global_crops:
    :param n_local_crops:
    :param student_slicing:
    :param teacher_slicing:
    :param global_crops_scale:
    :param local_crops_scale:
    :param projection_head_out_dim:
    :param projection_head_hidden_dim:
    :param projection_head_bottleneck_dim:
    :param projection_head_n_layers:
    :param warmup_teacher_temp:
    :param teacher_temp:
    :param warmup_teacher_temp_iterations:
    :param student_temp:
    :param center_momentum:
    :param grad_clip_factor:
    :param freeze_last_layer_iterations:
    :param teacher_momentum:
    :param final_lr:
    :param lr_scheduler_warm_up_iterations:
    :param final_weight_decay:
    :param optimizer:
    :param optimizer_parameters:
    :param grad_clipper:
    :return:
    """
    if unalabeled_train_set and unlabeled_train_loader:
        Warning('train_set and train_loader arguments are provided. Only train_loader is used')
    if unlabeled_train_loader and image_resolution:
        Warning('image_res argument has no effect and will only be used if only "train_set" is provided.')
    if unalabeled_train_set and not unlabeled_train_loader_parameters:
        raise NotImplementedError('parameters for building a pytorch dataloader e.g. batch size must be provided')
    if resume_training_from_checkpoint:
        Warning('Initial Student, teacher, optimizer state dicts will be overwritten by loaded state dicts.')
    if optimizer_state_dict and isinstance(resume_training_from_checkpoint, os.PathLike):
        Warning('optimizer_state_dict will be overwritten by loaded state_dict')

    data_augmentation = DinoAugmentationCropper(
        n_global_crops=n_global_crops,
        n_local_crops=n_local_crops,
        global_crops_scale=global_crops_scale,
        local_crops_scale=local_crops_scale,
        res=image_resolution or 224,
    )
    unalabeled_train_set.set_data_augmentation(data_augmentation)
    unlabeled_train_set = MultiCropDatasetWrapper(unalabeled_train_set)

    unlabeled_train_loader = DataLoader(dataset=unlabeled_train_set, **unlabeled_train_loader_parameters)

    classifier = classifier or KNeighborsClassifier(n_neighbors=10, weights='distance')

    if isinstance(resume_training_from_checkpoint, str):
        resume_training_from_checkpoint = Path(resume_training_from_checkpoint)

    if isinstance(resume_training_from_checkpoint, os.PathLike):
        dict_dic = Path(resume_training_from_checkpoint)
        step = resume_training_from_checkpoint.stem
        student_state_dict = torch.load(dict_dic / f'student_{step}.pt')
        teacher_state_dict = torch.load(dict_dic / f'teacher_{step}.pt')
        optimizer_state_dict = torch.load(dict_dic / f'optimizer_{step}.pt')
        loss_state_dict = torch.load(dict_dic / f'loss_{step}.pt')
        scaler_state_dict = torch.load(dict_dic / f'scaler_{step}.pt')
        step = int(step) + 1
    elif isinstance(resume_training_from_checkpoint, int) and not isinstance(resume_training_from_checkpoint, bool):
        step = resume_training_from_checkpoint + 1
    else:
        step = 0


    backbone = backbone_model or timm.models.xcit_small_12_p8_224_dist(in_chans=3, num_classes=0, pretrained=True)
    projection_head = ProjectionHead(
        in_dim=backbone.embed_dim,
        out_dim=projection_head_out_dim,
        hidden_dim=projection_head_hidden_dim,
        bottleneck_dim=projection_head_bottleneck_dim,
        n_layers=projection_head_n_layers,
    )
    student_model = MultiCropModelWrapper(backbone=backbone, projection_head=projection_head)
    # workaround because weight norm in proj head is not deepcopyable
    projection_head = ProjectionHead(
        in_dim=backbone.embed_dim,
        out_dim=projection_head_out_dim,
        hidden_dim=projection_head_hidden_dim,
        bottleneck_dim=projection_head_bottleneck_dim,
        n_layers=projection_head_n_layers,
    )
    teacher_model = MultiCropModelWrapper(backbone=deepcopy(backbone), projection_head=projection_head)
    teacher_update_function = EmaTeacherUpdate(teacher_momentum)
    lr = (unlabeled_train_loader.batch_size / 256) * 0.0005
    optimizer_parameters = optimizer_parameters or dict(lr=lr, weight_decay=0.04)
    optimizer = optimizer(get_params_groups(student_model), **optimizer_parameters)

    if isinstance(resume_training_from_checkpoint, os.PathLike):
        student_model.load_state_dict(student_state_dict)
        teacher_model.load_state_dict(teacher_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

    warmup_teacher_temp_iterations = warmup_teacher_temp_iterations or len(unlabeled_train_loader) * 10

    loss_function = DINOLoss(
        n_crops=n_global_crops + n_local_crops,
        student_temp=student_temp,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
        warmup_teacher_temp_iterations=warmup_teacher_temp_iterations,
        center_momentum=center_momentum,
        out_dim=projection_head_out_dim,
    )
    if resume_training_from_checkpoint:
        loss_function.step = step
    if isinstance(resume_training_from_checkpoint, os.PathLike):
        loss_function.load_state_dict(loss_state_dict)

    base_lr = optimizer.param_groups[0]['lr']
    base_wd = optimizer.param_groups[0]['weight_decay']
    warm_up_iterations = lr_scheduler_warm_up_iterations or len(unlabeled_train_loader) * 2
    total_iterations = len(unlabeled_train_loader) * n_trainings_epochs
    lr_cosine_scheduler = CosineScheduler(
        final_value=final_lr,
        base_value=base_lr,
        warm_up_iterations=warm_up_iterations,
        warm_up_starting_value=0,
        total_iterations=total_iterations,
    )
    lr_scheduler = LRScheduler(optimizer=optimizer, scheduler=lr_cosine_scheduler)
    wd_cosine_scheduler = CosineScheduler(
        final_value=final_weight_decay, base_value=base_wd, total_iterations=total_iterations
    )
    wd_scheduler = DinoWdScheduler(optimizer=optimizer, scheduler=wd_cosine_scheduler)

    freeze_last_layer_iterations = freeze_last_layer_iterations or len(unlabeled_train_loader)
    grad_clipper = grad_clipper or DinoGradClipper(
        freeze_last_layer_iterations=freeze_last_layer_iterations, clip_factor=grad_clip_factor
    )
    if resume_training_from_checkpoint:
        grad_clipper.step = step

    dino_gym = StudentTeacherGym(
        train_loader=unlabeled_train_loader,
        student_model=student_model,
        teacher_model=teacher_model,
        student_slicing=student_slicing,
        teacher_slicing=teacher_slicing,
        optimizer=optimizer,
        loss_function=loss_function,
        n_trainings_epochs=n_trainings_epochs,
        starting_step=step,
        lr_scheduler=lr_scheduler,
        weight_decay_scheduler=wd_scheduler,
        grad_clipper=grad_clipper,
        teacher_update_function=teacher_update_function,
        eval_frequency=eval_frequency,
        classifier=classifier,
        labeled_train_loader=labeled_train_loader,
        val_loader=val_loader,
        metrics=metrics,
        metrics_parameters=metrics_parameters,
        metrics_require_probabilities=metrics_requires_probability,
        device=device,
        save_path=save_path,
        model_saving_frequency=model_saving_frequency,
        logger=logger,
    )
    if isinstance(resume_training_from_checkpoint, os.PathLike):
        dino_gym.scaler.load_state_dict(scaler_state_dict)

    return dino_gym


class SuperGym(BaseGym):
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            model: nn.Module,
            loss_function: Callable = CrossEntropyLoss(),
            optimizer: Optimizer = AdamW,
            lr_scheduler: Callable = None,
            weight_decay_scheduler: Callable = None,
            grad_clipper: Callable = None,
            n_trainings_epochs: int = 100,
            starting_step: int = 0,
            metrics: Callable | List[Callable] = balanced_accuracy_score,
            metrics_parameters: dict | List[dict] = None,
            metrics_require_probabilities: bool | List[bool] = False,
            early_stopper: Callable = None,
            device: int = 'cuda',
            eval_frequency: int = 100,
            model_saving_frequency: int = 250,
            save_path: os.PathLike = None,
            finetune: bool = False,
            class_names: List[str] = None,
            logger: BaseLogger = None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            weight_decay_scheduler=weight_decay_scheduler,
            grad_clipper=grad_clipper,
            n_trainings_epochs=n_trainings_epochs,
            starting_step=starting_step,
            metrics=metrics,
            metrics_parameters=metrics_parameters,
            metrics_require_probabilities=metrics_require_probabilities,
            device=device,
            eval_frequency=eval_frequency,
            model_saving_frequency=model_saving_frequency,
            save_path=save_path,
            logger=logger,
        )
        self.finetune = finetune

        if finetune:
            for param_name, param in model.named_parameters():
                param_name = param_name.split('.')
                if not ('head' in param_name or 'norm' in param_name):
                    param.requires_grad = False
                else:
                    pass

        self.class_names = class_names
        self.early_stopper = early_stopper

    def train(self) -> nn.Module:
        self.model.to(self.device)
        optimizer_to_device(self.optimizer, self.device)
        if isinstance(self.loss, nn.Module):
            self.loss.to(self.device)

        best_metric = None
        early_stopping = False
        best_model = deepcopy(self.model).cpu()
        epoch_bar = self._init_tqdm_bar()
        smooth_loss = deque([], maxlen=len(self.train_loader))
        while epoch_bar.n < len(epoch_bar):
            for trainings_batch in self.train_loader:
                step = epoch_bar.n
                self.optimizer.zero_grad(True)
                loss_val = self.model_update_step(trainings_batch)
                smooth_loss.append(loss_val)
                self._update_lr_wd(step=step, metric=np.mean(smooth_loss))

                if step % self.eval_f == 0 and self.val_loader:
                    metrics_dict = self._eval_model(self.model)
                    # uses first metric as primary metric for early stopping
                    if self.early_stopper:
                        early_stopping = self.early_stopper(list(metrics_dict.values())[0])

                if not best_metric:
                    best_metrics = deepcopy(metrics_dict)
                    metric_improvement = [False]
                else:
                    metric_improvement = [
                        best_metric > metric_value for best_metric, metric_value in zip(best_metrics, metrics_dict)
                    ]
                    best_metrics = {
                        key: max(best_value, value)
                        for best_value, value, key in zip(best_metrics.values, metrics_dict.items())
                    }

                if self.save_path and step % self.save_f == 0 or any(metric_improvement):
                    self._save_training_state(step)

                self.logger.log(dict(loss=loss_val, metrics=metrics_dict), step=step)
                epoch_bar.set_description(desc=f"loss={loss_val:.4f}")
                epoch_bar.update(1)

                if early_stopping:
                    break
            if early_stopping:
                break
        return best_model

    def model_update_step(self, trainings_batch: Iterable[torch.Tensor]) -> float:
        images, labels = trainings_batch
        if not isinstance(images, torch.TensorType):
            images = [image.to(self.device) for image in images]
            labels = [label.to(self.device) for label in labels]
            labels = torch.concat(labels, dim=0)
        self.model.train()
        with autocast():
            outputs = self.model(images)
            loss_val = self.loss(outputs, labels)
        self._model_backprop(loss_val)
        return loss_val.item()

    def _predict_val_samples(self, model) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        predictions = []
        label_list = []
        for val_batch in self.val_loader:
            images, labels = val_batch
            images = images.to(self.device)
            with autocast() and torch.no_grad():
                outputs = self.model(images)
                predictions.append(outputs.cpu().numpy())
                label_list.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(label_list, axis=0)
        model.train()
        return predictions, labels

    def _save_training_state(self, step: int | str):
        step = str(step)
        save_path = self.save_path / step
        save_path.mkdir()
        self._save_state_dict(self.model, save_path / f'model_{step}.pt')
        self._save_state_dict(self.optimizer, save_path / f'optimizer_{step}.pt')
