from functools import partial
from pathlib import Path

import numpy as np
import timm
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from sparsam.loss import DINOLoss
from sparsam.train import create_dino_gym, StudentTeacherGym
from sparsam.utils import DummyLogger, model_inference, ModelMode
from sparsam.dataset import BaseSet

# First step creating your Datasets / loader, must be following the api defined in BaseSet (returns: img, label), the e
# easiest way to achieve this is to inherit directly from the BaseSet
# This dataset does not require labels and will be used for self supervised training NOTE: this will be internally
# converted to dataloader object
unlabeled_train_set = BaseSet()

# These Datasets are optional: the labeled_train_set and val_set may be used to track the process during SSL
# These Datasets are required, if a classifier is fitted after SSL to a specific task.
labeled_train_loader = DataLoader(BaseSet())
val_loader = DataLoader(BaseSet())
# This is the independent test set
test_loader = DataLoader(BaseSet())

# Second step: choosing the Backbone, here we choose the timm implementation of ViT, but any model works
backbone = timm.models.VisionTransformer()

# Option One: creating the DINO training Setup with default parameters:
# TODO: provide machine specific dataloader parameter
data_loader_parameter = dict(batch_size=256)
# TODO: choose your classifier, any one following the sklearn api works
classifier_pipeline = Pipeline([
    ('standardizer', PowerTransformer()),
    ('pca', PCA()),
    ('classifier', SVC(probability=True))
])
# TODO: chooes logging tool: any logger with a log function works. We tested our JsonLogger and wandb.
logger = DummyLogger()
# TODO: save_path: base path were to save models
save_path = Path()
# TODO: choose device
device = 'cuda'
# TODO choose metrics: they will be logged, multiple metrics are allowed via a list
metrics = [
    partial(classification_report, output_dict=True, zero_division=0),
],
# set if metrics require probability scores
metrics_requires_probability = [False],

################################ FIRST OPTION #########################################

# There are further specific options like the scheduler, optimizer, data_augmentation etc. which may be chosen by the
# user, but are not covered in this example. To see the options please inspect the "create_dino_gym" function
gym = create_dino_gym(
    unalabeled_train_set=unlabeled_train_set,
    labeled_train_loader=labeled_train_loader,
    val_loader=val_loader,
    backbone_model=backbone,
    classifier=classifier_pipeline,
    logger=logger,
    unlabeled_train_loader_parameters=data_loader_parameter,
    save_path=save_path,
    device=device,
    metrics=[
        partial(classification_report, output_dict=True, zero_division=0),
    ],
    metrics_requires_probability=[True],
    # If you wish to continue the training set the resume_training_from_checkpoint argument with the patch to the checkpoint folder in "create_dino_gym"
    resume_training_from_checkpoint=False,
    # Note: should be scaled down for larger batch sizes (0.996 for a batch size of 512) and up for smaller ones (see
    # https://arxiv.org/abs/2104.14294 for details)
    teacher_momentum=0.9995
)

################################ SECOND OPTION #########################################
# directly initialize the class, this adds a lot of custom options, but does not provide defaults. Here only the three
# required options are listed, for more please see the documentation
gym = StudentTeacherGym(
    student_model=backbone,  # for most setups a projection head is required, this can be found in utils
    train_loader=DataLoader(unlabeled_train_set),
    loss_function=DINOLoss(),
    # for more options please see the class docs
)

#######################################################################################

# gym returns models after training. Also the models, optimizers etc are checkpointed regularly.
student, teacher = gym.train()

# extract train / test feature and label
train_features, train_labels = model_inference(
    labeled_train_loader, model=teacher, mode=ModelMode.EXTRACT_FEATURES, device=device
)
test_features, test_labels = model_inference(test_loader, teacher, mode=ModelMode.EXTRACT_FEATURES, device=device)
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# fit classifier, predict test features, generate classification report
classifier_pipeline.fit(train_features, train_labels)
preds = classifier_pipeline(test_features)
report = classification_report(test_labels, preds, output_dict=True, zero_division=0)
