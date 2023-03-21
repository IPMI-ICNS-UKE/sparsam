# Sparsam: Deep Learning on SPARSely Annotated Medical-data

*The name of the repository comes from the German word "[sparsam](https://en.wiktionary.org/wiki/sparsam)" [ˈʃpaːɐ̯zaːm], meaning "economical" or "sparing", which reflects the key feature of our approach regarding annotated samples for medical image data.*

This repository is a placeholder for the code and data used in the experiments described in the corresponding publication: "Self-supervision for medical image classification". Please note that the code and data will only be made available after the publication has been accepted. We appreciate your understanding in this matter. Once the publication has been accepted, we will update this repository with the necessary code and data to reproduce the experiments. Until then, solely additional information regarding model training and implementation details will be provided here. 

## Implementation Details

In the following, parameters that have to be instantiated to reproduce the presented results are detailed. The naming follows the DINO implementation.

### Image cropping and augmentation
**Global crops:**
- Number N of crops: 2
- Crop size relative to input image: 0.5 - 1.0
- Network image input size (crops are rescaled if necessary): 256 x 256 pixel

**Local crops:**
- N: 5
- Crop size relative to input image: 0.1 - 0.5
- Network image input size: 96 x 96

**Augmentation:**
- Horizontal / vertical flip: p=0.5
- Rotation: random selection of rotation between 0 and 180 degrees
- Color jitter (brightness, contrast, saturation and hue): p=0.8
- Greyscaling: p=0.2
- Gaussian blurring: random selection of kernel radius between 0.1 - 5
- Solarization (50% threshold; only global crops): p=0.1

### Network architecture and related settings

**Backbone model:**
- XCiT small pretrained on ImageNet, patch size: 8
- Representation dimensionality: 384

**Projection head:**
- Number of fully connected layers: 4
- Hidden dimension size: 2,048
- Bottleneck dimension size: 256
- Projection size: 65,536

**Optimization process:**
- Student model optimization by stochastic gradient descent
- Adam with weight decay (cosine incline from 0.05 to 1)
- Learning rate: cosine decline with warm up from 0.0005 to 1e-6
- Batch size: 256
- Training epochs: 300
- Teacher model EMA momentum: Cosine incline from 0.998 to 1 

### Classifier settings

The parameter description follows the naming of the scikit-learn implementation. 

**Support vector machine (SVM):**
- Kernel: Radial basis function
- class weights: balanced
- C: 1
- gamma: 0.1

**Logistic regression (LR):**
- class weights: balanced
- C: 2.5

**K-Nearest Neighbors (KNN):**
- Number of neighbours: 10
- weights: distance

**Linear layer (LL) and DL-Baseline:**
- Only global crops 
- Learning rate: 1e-4
- Batch size: 128
- Training epochs: 300

