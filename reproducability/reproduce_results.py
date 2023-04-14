import json
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Tuple, List, Sequence

import click
import numpy as np
import pandas as pd
import torch.utils.data
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from timm.optim import AdamW
from timm.scheduler import PlateauLRScheduler
from timm.utils import adaptive_clip_grad
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import git

from Dataset import ImageSet, MultiCropDatasetWrapper
from main.parameter import MODEL, CLASSIFIERS

import wandb

from sparsam.data_augmentation import DinoAugmentationCropper
from sparsam.helper import uniform_train_test_splitting, recursive_dict, _sort_class_names, get_large_classes, \
    filter_classes, dict_merge
from sparsam.train import create_dino_gym, SuperGym
from sparsam.utils import JsonLogger, model_inference, MultiCropModelWrapper, EarlyStopper, ModelMode, GradClipWrapper


@click.command()
@click.argument('image_root_path', type=click.Path(exists=True))
@click.argument(
    'data_set',
    type=click.STRING,
)
@click.option(
    '--trainings_mode',
    type=click.STRING,
    default='ssl+classifier',
    help='Which results to reproduce. The mode is provided as composed string. If "ssl" is part of the string'
         'The model is going to be pretrained with DINO otherwise precomputed weights are used. Valid options are: '
         '"ssl-classifier", "ssl-linear", "classifier", "linear", and "supervised". In principal all combinations '
         'are valid and will be executed.',
)
@click.option(
    '--eval_class_sizes',
    '-e',
    type=click.STRING,
    help='Which class sizes to evaluate. Multiple are possible. e.g. "-e 10,25 (no space between numbers!!!). Evaluates the test set for 10 and '
         '250 samples per class. Default is "1, 5, 10, 25, 50, 100, 250, 500, 1000"',
)
@click.option(
    '--n_iterations',
    type=click.INT,
    default=1,
    help='How often to repeat experiments (does not include self supervised training)',
)
@click.option(
    '--min_class_size',
    type=click.INT,
    help='only consider classes, where the training_samples exceed the' 'provided min class_size. Default is None.',
)
@click.option('--save_path', type=click.Path(exists=True), help='Where to save models and logs')
# machine specific parameters, you may also choose to just adjust the corresponding yaml file
@click.option('--device', type=click.INT, default=0, help='Which GPU to train on')
@click.option('--batch_size', type=click.INT, default=256)
@click.option('--num_workers', type=click.INT, default=8, help='How many workers to use for data loading')
@click.option(
    '--gradient_checkpointing',
    type=click.BOOL,
    default=True,
    help='Whether to use gradient checkpointing. GC trades memory for computation time, which leads to a '
         'smaller memory footprint with a slower training. Resulting in a larger possible batch size.',
)
@click.option(
    '--load_machine_param_from_yaml',
    type=click.STRING,
    help='Path to the corresponding yaml. If "True" the default path is used.',
)
def run(
        image_root_path: os.PathLike,
        data_set: str,
        trainings_mode: str,
        eval_class_sizes: Tuple[int] = None,
        n_iterations: int = 1,
        min_class_size: int = None,
        save_path: os.PathLike = None,
        device: int | str = 'cuda',
        batch_size: int = 256,
        num_workers: int = 8,
        gradient_checkpointing: bool = True,
        load_machine_param_from_yaml: bool | Path = False,
):
    """
    :param image_root_path: Path to the root image folder
    :param data_set: Which data set to train. Must be one of the following: ["isic", "bone_marrow", "endo"]
    :param trainings_mode: Which results to reproduce. The mode is provided as composed string. If "ssl" is part of the
                            string. The model is going to be pretrained with DINO otherwise precomputed weights are
                            used. Valid useful options are: "ssl,classifier", "ssl,linear", "classifier",
                            "linear", and "supervised". In principle all combinations are valid and will be
                            executed.
    :param min_class_size: only consider classes, where the training_samples exceed the provided min class_size. Default
                           is None.
    :param eval_class_sizes: Which class sizes to evaluate. Multiple are possible. e.g. "-e 10 -e 250. Evaluates the
                             test set for 10 and 250 samples per class. Default is "1, 5, 10, 25, 50, 100, 250, 500,
                             1000"
    :param n_iterations: How often to repeat experiments (does not include self supervised training).
    :param device: Which GPU to train on
    :param batch_size: batch size on gpu
    :param num_workers: How many workers to use for data loading
    :param gradient_checkpointing: Whether to use gradient checkpointing. GC trades memory for computation time, which
     leads to a smaller memory footprint with a slower training. Resulting in a larger possible batch size
    :param load_machine_param_from_yaml: Path to the corresponding yaml. If "True" the default path is used.
    :return:
    """
    repo = git.Repo('./', search_parent_directories=True)
    save_path = Path(save_path)
    current_path = Path(repo.working_tree_dir)
    cfg_path = Path('configs/dino_config.yml')
    with open(current_path / cfg_path) as f:
        config = yaml.load(f, yaml.Loader)
    cfg_path = Path(f'configs/{data_set}.yml')
    with open(current_path / cfg_path) as f:
        data_set_config = yaml.safe_load(f)
    dict_merge(config, data_set_config)

    if isinstance(eval_class_sizes, int):
        eval_class_sizes = [eval_class_sizes]
    if isinstance(eval_class_sizes, str):
        eval_class_sizes = list(map(int, eval_class_sizes.strip().split(',')))
    trainings_mode = trainings_mode.strip().split(',')
    eval_class_sizes = eval_class_sizes or [1, 5, 10, 25, 50, 100, 250, 500, 1000]

    if load_machine_param_from_yaml:
        if load_machine_param_from_yaml == 'True':
            cfg_path = Path('configs/machine_config.yml')
        else:
            cfg_path = Path(load_machine_param_from_yaml)
        with open(current_path / cfg_path) as f:
            machine_config = yaml.safe_load(f)
        dict_merge(config, machine_config)
    config['device'] = config.get('device', device)
    config['data_loader_parameter']['num_workers'] = config['data_loader_parameter'].get('num_workers', num_workers)
    config['data_loader_parameter']['batch_size'] = config['data_loader_parameter'].get('batch_size', batch_size)

    # dataset initialization
    image_root_path = Path(image_root_path)
    sample_csvs_root_path = current_path / Path(f'splits/{data_set}')
    sample_csvs = [path.stem for path in sample_csvs_root_path.glob('*.csv')]
    unlabeled_train_csv_path = None
    labeled_train_csv_path = None
    test_csv = None
    if 'unlabeled_train' in sample_csvs:
        unlabeled_train_csv_path = sample_csvs_root_path / 'unlabeled_train.csv'
    if 'labeled_train' in sample_csvs:
        labeled_train_csv_path = sample_csvs_root_path / 'labeled_train.csv'
    if 'test' in sample_csvs:
        test_csv = sample_csvs_root_path / 'test.csv'
    if not labeled_train_csv_path and not unlabeled_train_csv_path:
        raise ValueError(
            'no training data is provided.'
            ' Please provide a valid path to a labeled and/ or unlabeled csv containing image paths.'
        )
    if test_csv and not labeled_train_csv_path:
        raise ValueError('no training data is provided, so test performance can not be evaluated')
    class_names = config.get('class_names', None)

    data_loader_parameter = config['data_loader_parameter']
    test_loader_parameter = deepcopy(data_loader_parameter)
    test_loader_parameter['drop_last'] = False
    if save_path:
        logger = JsonLogger(save_path)
    else:
        logger = None

    labeled_train_frame = pd.read_csv(labeled_train_csv_path)
    labeled_train_paths = labeled_train_frame['image']
    labeled_train_paths = [image_root_path / Path(img) for img in labeled_train_paths]
    train_labels = np.array(labeled_train_frame['label'])

    if min_class_size:
        class_names = get_large_classes(train_labels, min_amount=min_class_size)
        labeled_train_paths, train_labels = filter_classes(labeled_train_paths, train_labels, class_names)

    test_loader = create_test_loader(
        test_csv,
        image_root_path,
        class_names=class_names,
        data_set_parameter=dict(img_size=config['dataset']['image_resolution']),
        data_loader_parameter=test_loader_parameter,
    )

    model = MODEL(**config['model_parameter'])
    model.to(device)
    if gradient_checkpointing:
        model.set_grad_checkpointing()
        model.to(torch.float32)

    if 'ssl' in trainings_mode:
        unlabeled_train_set = create_unlabeled_train_set(
            unlabeled_train_csv_path=unlabeled_train_csv_path,
            labeled_train_csv_path=labeled_train_csv_path,
            image_root_path=image_root_path,
            class_names=class_names,
            data_set_parameter=dict(img_size=config['dataset']['image_resolution']),
        )
        dino_gym = create_dino_gym(
            unalabeled_train_set=unlabeled_train_set,
            backbone_model=model,
            logger=logger,
            unlabeled_train_loader_parameters=data_loader_parameter,
            save_path=save_path,
            **config['gym'],
        )
        _, model = dino_gym.train()
        dino_gym._save_state_dict(model, dino_gym.save_path / 'ssl_best_model.pt')

    elif 'classifier' in trainings_mode or 'linear_layer' in trainings_mode:
        map_location = None
        if device != 'cpu':
            map_location = 'cuda'
        pretrained_weights = torch.load(current_path / f'models/pretrained_{data_set}.pt', map_location=map_location)
        pretrained_weights = {
            key.replace('backbone.', ''): weight for key, weight in pretrained_weights.items() if 'backbone.' in key
        }
        model.load_state_dict(pretrained_weights, strict=True)
        test_features, test_labels = model_inference(test_loader, model, mode=ModelMode.EXTRACT_FEATURES, device=device)
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

    results_dict = recursive_dict()
    if 'classifier' in trainings_mode:
        train_set = ImageSet(
            img_paths=labeled_train_paths,
            labels=train_labels,
            class_names=class_names,
            normalize=True,
            img_size=config['dataset']['image_resolution'],
        )
        train_loader = DataLoader(
            train_set,
            **test_loader_parameter,
        )
        train_features, train_labels = model_inference(
            train_loader, model=model, mode=ModelMode.EXTRACT_FEATURES, device=device
        )
        classifier_results = fit_classifier(
            totaL_train_features=train_features,
            total_train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            eval_class_sizes=eval_class_sizes,
            class_names=class_names,
            n_iterations=n_iterations,
        )
        results_dict['classifier'] = classifier_results
        with open(save_path / 'results.json', 'w') as h:
            json.dump(results_dict, h)

    base_patience = config['early_stopping_parameter']['patience']

    for class_size in eval_class_sizes:
        for iteration in range(n_iterations):
            sub_sampled_train_paths, sub_sampled_train_labels, _, _ = uniform_train_test_splitting(
                labeled_train_paths, train_labels, n_samples_class=class_size, seed=iteration
            )
            sub_sampled_train_paths, sub_sampled_train_labels, _, _ = uniform_train_test_splitting(
                labeled_train_paths, train_labels, n_samples_class=class_size, seed=iteration
            )
            data_19 = train_test_split(
                sub_sampled_train_paths,
                sub_sampled_train_labels,
                train_size=0.7,
                stratify=sub_sampled_train_labels,
                random_state=iteration,
            )

            sub_sampled_train_paths = data_19[0]
            val_paths = data_19[1]
            sub_sampled_train_labels = data_19[2]
            val_labels = data_19[3]

            train_set = ImageSet(
                img_paths=sub_sampled_train_paths,
                labels=sub_sampled_train_labels,
                class_names=class_names,
                normalize=False,
                img_size=config['dataset']['image_resolution'],
            )
            train_set = MultiCropDatasetWrapper(
                dataset=train_set,
                data_cropper=DinoAugmentationCropper(
                    res=config['dataset']['image_resolution'], **config['augmentor_parameter']
                ),
            )
            data_loader_parameter['batch_size'] = min(len(train_set), config['data_loader_parameter']['batch_size'])
            train_loader = DataLoader(train_set, **data_loader_parameter)

            val_set = ImageSet(
                img_paths=val_paths,
                labels=val_labels,
                class_names=class_names,
                normalize=True,
                img_size=config['dataset']['image_resolution'],
            )
            val_loader = DataLoader(val_set, **test_loader_parameter)

            if 'linear' in trainings_mode:
                gym = create_supervised_gym(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    save_path=save_path,
                    finetune=True,
                    class_size=class_size,
                    class_names=class_names,
                    base_patience=base_patience,
                    device=device,
                    config=config,
                )
                best_model = gym.train()
                best_model.to(device)
                test_probas, test_labels = model_inference(
                    test_loader, best_model, mode=ModelMode.CLASSIFICATION, device=device
                )
                test_preds = np.argmax(test_probas, axis=-1)
                report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)

                results_dict[class_size]['last_layer'][iteration] = report
                with open(save_path / 'results.json', 'w') as h:
                    json.dump(results_dict, h)
            if 'supervised' in trainings_mode:
                gym = create_supervised_gym(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    save_path=save_path,
                    finetune=False,
                    class_size=class_size,
                    class_names=class_names,
                    base_patience=base_patience,
                    device=device,
                    config=config,
                )
                best_model = gym.train()
                best_model.to(device)
                test_probas, test_labels = model_inference(
                    test_loader, best_model, mode=ModelMode.CLASSIFICATION, device=device
                )
                test_preds = np.argmax(test_probas, axis=-1)
                report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)

                results_dict[class_size]['last_layer'][iteration] = report
                with open(save_path / 'results.json', 'w') as h:
                    json.dump(results_dict, h)


def create_unlabeled_train_set(
        unlabeled_train_csv_path: os.PathLike = None,
        labeled_train_csv_path: os.PathLike = None,
        image_root_path: os.PathLike = None,
        class_names: Sequence[str] = None,
        data_set_parameter: dict = None,
) -> Dataset:
    if not unlabeled_train_csv_path and not labeled_train_csv_path:
        raise ValueError('No training data is provided')
    total_train_paths = list()
    if unlabeled_train_csv_path:
        unlabeled_train_frame = pd.read_csv(labeled_train_csv_path)
        total_train_paths.extend(unlabeled_train_frame['image'])

    if labeled_train_csv_path:
        labeled_train_frame = pd.read_csv(labeled_train_csv_path)
        total_train_paths.extend(labeled_train_frame['image'])
    if image_root_path:
        total_train_paths = [image_root_path / Path(img) for img in total_train_paths]
    data_set_parameter = data_set_parameter or {}
    unlabeled_train_set = ImageSet(
        img_paths=total_train_paths, class_names=class_names, normalize=False, **data_set_parameter
    )
    return unlabeled_train_set


def create_test_loader(
        test_csv_path: os.PathLike,
        image_root_path: os.PathLike,
        class_names: Sequence[str] = None,
        data_set_parameter: dict = None,
        data_loader_parameter: dict = None,
) -> DataLoader:
    test_frame = pd.read_csv(test_csv_path)
    test_paths = test_frame['image']
    test_labels = test_frame['label']
    test_paths = [image_root_path / Path(img) for img in test_paths]
    test_paths, test_labels = filter_classes(test_paths, test_labels, filter_labels=class_names)
    test_set = ImageSet(
        img_paths=test_paths, labels=test_labels, class_names=class_names, normalize=True, **data_set_parameter
    )
    test_loader = DataLoader(test_set, **data_loader_parameter)
    return test_loader


def fit_classifier(
        totaL_train_features: np.ndarray,
        total_train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        eval_class_sizes: Sequence[int],
        class_names: Sequence[str],
        n_iterations: int = 1,
) -> dict:
    results = recursive_dict()
    for eval_class_size in eval_class_sizes:
        for iteration in range(n_iterations):
            train_features, train_labels, _, _ = uniform_train_test_splitting(
                totaL_train_features, total_train_labels, n_samples_class=eval_class_size, seed=iteration
            )
        train_features = np.array(train_features)
        train_labels = np.array(train_labels)
        for classifier in CLASSIFIERS:
            if isinstance(classifier, KNeighborsClassifier):
                classifier.n_neighbors = 5 * len(class_names)
            pca = PCA()
            standardizer = PowerTransformer()
            classifier_pipeline = Pipeline([('standardizer', standardizer), ('pca', pca), ('classifier', classifier)])
            classifier_pipeline.fit(train_features, train_labels)
            test_probas = classifier_pipeline.predict_proba(test_features)
            test_preds = np.argmax(test_probas, axis=-1)
            report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
            results[eval_class_size][classifier.__class__.__name__][iteration]['report'] = report
    return results


def fit_linear_layer():
    pass


def create_supervised_gym(
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: os.PathLike,
        finetune: bool,
        class_size: int,
        class_names: Sequence,
        base_patience: int,
        device: str | int,
        config: dict,
) -> SuperGym:
    if save_path:
        logger = JsonLogger(save_path)
    else:
        logger = None
    config['model_parameter']['num_classes'] = len(class_names)
    multi_crop_model = MultiCropModelWrapper(backbone=MODEL(**config['model_parameter']), mode=ModelMode.CLASSIFICATION)
    optimizer = AdamW(multi_crop_model.parameters(), **config['optimizer_parameter'])
    metrics = balanced_accuracy_score
    metrics_require_probabilities = False
    model_saving_frequency = 1000
    grad_clipper = GradClipWrapper(partial(adaptive_clip_grad, clip_factor=0.16))
    lr_scheduler = PlateauLRScheduler(optimizer, **config['lr_scheduler_parameter'])
    sub_sampled_train_labels = train_loader.dataset.dataset.labels
    unique_class_names, label_counts = np.unique(sub_sampled_train_labels, return_counts=True)
    unique_class_names = list(unique_class_names)
    label_counts = _sort_class_names(class_names, unique_class_names, label_counts)
    label_counts = np.asarray(label_counts)
    weights = 1 / label_counts
    weights = weights * len(class_names) / weights.sum()
    loss_function = CrossEntropyLoss(weight=torch.Tensor(weights).to(device), **config['loss_parameter'])
    patience = base_patience * class_size * len(class_names) // config['gym']['eval_frequency']
    patience = max(250 // config['gym']['eval_frequency'], patience)
    config['early_stopping_parameter']['patience'] = min(patience, 2500 // config['gym']['eval_frequency'])
    early_stopper = EarlyStopper(**config['early_stopping_parameter'])
    gym = SuperGym(
        train_loader=train_loader,
        val_loader=val_loader,
        model=multi_crop_model,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_clipper=grad_clipper,
        early_stopper=early_stopper,
        metrics=metrics,
        metrics_require_probabilities=metrics_require_probabilities,
        device=device,
        model_saving_frequency=model_saving_frequency,
        logger=logger,
        save_path=save_path,
        finetune=finetune,
        class_names=class_names,
        **config['gym'],
    )
    return gym


if __name__ == "__main__":
    run()
