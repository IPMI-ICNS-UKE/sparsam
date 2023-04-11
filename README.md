# Sparsam: Deep Learning on SPARSely Annotated Medical-data

*The name of the repository comes from the German word "[sparsam](https://en.wiktionary.org/wiki/sparsam)" [ˈʃpaːɐ̯zaːm], meaning "economical" or "sparing", which reflects the key feature of our approach regarding annotated samples for medical image data.*

This repository is a placeholder for the code and data used in the experiments described in the corresponding publication: "Self-supervision for medical image classification". Please note that the code and data will only be made available after the publication has been accepted. We appreciate your understanding in this matter. Once the publication has been accepted, we will update this repository with the necessary code and data to reproduce the experiments. Until then, solely additional information regarding model training and implementation details will be provided here. 

## Data sets

The following data sets were used in this study:

- **Bone Marrow (BM) data set:** A highly unbalanced data set of single bone marrow cell microscopy images that can be found [here](example.com/dataset1) and is provided by [Matek et al. (2021)](https://doi.org/10.1182/blood.2020010568).
- **Endoscopic (Endo) image data set:** The so-called HyperKvasir data set, consisting of labeled and unlabeled endoscopic images of the upper and lower gastrointestinal tract that can be found [here](example.com/dataset2) and is provided by [Borgli et al. (2020)](https://doi.org/10.1038/s41597-020-00622-y).
- **Dermoscopic lesion (ISIC) data set:**  A collection of dermoscopic skin lesion data sets that were released as part of the annual Grand Challenges organized by the International Skin Lesion Collaboration (ISIC) that can be found [here](example.com/dataset3) (provided by ).


## Results

![Classification balanced accuracy, full data](imgs/balanced_acc_250.png "Classification balanced accuracy, full data")

![Classification balanced accuracy, 250+](imgs/balanced_acc_250+.png "Classification balanced accuracy, 250+")

![confusion matrix for bone marrow data set](imgs/cfm_BM.png "Confusion matrix for the BM data set")
