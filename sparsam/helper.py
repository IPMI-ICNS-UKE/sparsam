import math
import warnings
from collections import defaultdict
from random import random, shuffle, Random
from typing import Mapping, Sequence, List, Tuple

import numpy as np
import torch
from torch import Tensor


def recursive_dict():
    return defaultdict(recursive_dict)


def dict_merge(dct: dict, merge_dct: dict):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def get_large_classes(labels: Sequence, min_amount: int) -> List:
    class_names, n_samples_labels = np.unique(labels, return_counts=True)
    n_samples_labels = n_samples_labels >= min_amount
    large_labels = [label for label, label_bool in zip(class_names, n_samples_labels) if label_bool]
    return large_labels


def filter_classes(features: Sequence, labels: Sequence, filter_labels: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    labels = np.asarray(labels)
    filtered_features = []
    filtered_labels = []
    # label_mapping = {label: new_label for new_label, label in enumerate(filter_labels)}
    for label in filter_labels:
        valid_samples = labels == label
        filtered_features.append(features[valid_samples])
        valid_labels = labels[valid_samples]
        # valid_labels[:] = label_mapping[label]
        filtered_labels.append(valid_labels)
    filtered_features = np.concatenate(filtered_features, axis=0)
    filtered_labels = np.concatenate(filtered_labels, axis=0)
    return filtered_features, filtered_labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]


def cancel_gradients_last_layer(epoch, model, n_epochs_freeze_last_layer):
    if epoch < n_epochs_freeze_last_layer:
        for n, p in model.named_parameters():
            if "last_layer" in n:
                p.grad = None


def uniform_train_test_splitting(feature, labels, n_samples_class=100, seed=None):
    index = list(range(len(feature)))
    if seed:
        random.Random(seed).shuffle(index)
    else:
        shuffle(index)
    split_feature = []
    split_label = []
    rest_feature = []
    rest_label = []
    counter = {label: 0 for label in set(labels)}
    for i in index:
        label = labels[i]
        if counter[label] < n_samples_class:
            split_label.append(label)
            split_feature.append(feature[i])
            counter[label] = counter[label] + 1
        else:
            rest_label.append(label)
            rest_feature.append(feature[i])
    return split_feature, split_label, rest_feature, rest_label


def uniform_subsampling(feature, labels, n_samples=100, seed=None):
    index = list(range(len(feature)))
    if seed:
        Random(seed).shuffle(index)
    else:
        shuffle(index)
    new_feature = []
    new_label = []
    counter = np.zeros(max(labels) + 1)
    for i in index:
        label = labels[i]
        if counter[label] < n_samples:
            new_label.append(label)
            new_feature.append(feature[i])
            counter[label] = counter[label] + 1
    return new_feature, new_label


def _sort_class_names(org_class_names, new_class_order, class_freq):
    new_clas_freq = []
    for class_name in org_class_names:
        class_idx = new_class_order.index(class_name)
        new_clas_freq.append(class_freq[class_idx])
    return new_clas_freq


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Copied from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    # Copied from Facebook, Inc. and its affiliates https://github.com/facebookresearch/dino
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
