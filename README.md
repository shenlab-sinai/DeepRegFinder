Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# DeepRegFinder: *Deep* Learning based *Reg*ulatory Elements *Finder*
by Li Shen, Aarthi Ramakrishnan, George Wangensteen, Sarah Kim

Shen Lab Website: http://labs.neuroscience.mssm.edu/project/shen-lab/

Icahn School of Medicine at Mount Sinai, New York, NY, USA

**DeepRegFinder** is a deep learning based program to identify DNA regulatory elements using ChIP-seq. It uses the deep learning framework PyTorch. 

## Overview
Identifying DNA regulatory elements such as enhancers and promoters has always been an important topic in the epigenomics field. Although certain histone marks are known to exhibit characteristic binding patterns at enhancers and promoters, the exact rules to make the call do not exist. Using machine learning models that are trained on known enhancers to predict at other regions using histone mark ChIP-seq data has been found to be the most successful method so far.

Many machine learning algorithms for enhancer identification exist. However, most of them are designed for reproducing results only. It's a hassle to apply them to your own data considering the most time-consuming part of a machine learning project is often data cleaning and formatting. We developed DeepRegFinder to be a modularized pipeline for you to build training data from aligned reads and genomic annotation easily so that you can use them to train models and make predictions. DeepRegFinder uses two deep neural networks: convolutional neural net (CNN) and recurrent neural net (RNN).

### Benchmarks
DeepRegFinder predicts one of five classes for a genomic region: poised enhancer (PE), active enhancer (AE), poised promoter (PT), active promoter (AT) and background (Bdg). Mean average precision (mAP) of the four non-background classes is used as the model evaluation metric. DeepRegFinder compares favorably with two established methods on three different cell types (mAP with 95% confidence intervals):
|                                    |     K562                    |     H1                      |     GM12878                 |
|------------------------------------|-----------------------------|-----------------------------|-----------------------------|
|     DeepRegFinder (CNN)            |     0.721 [0.711, 0.732]    |     0.694 [0.680, 0.708]    |     0.679 [0.671, 0.686]    |
|     DeepRegFinder  (RNN)           |     0.717 [0.707, 0.728]    |     0.686 [0.672, 0.700]    |     0.673 [0.666, 0.681]    |
|     EP-DNN (Kim,S.G. et al.)       |     0.619 [0.609, 0.629]    |     0.650 [0.637, 0.664]    |     0.634 [0.626, 0.642]    |
|     RFECS (Rajagopal,N. et al.)    |     0.639 [0.627, 0.653]    |     0.665 [0.655, 0.676]    |     0.633 [0.625, 0.640]    |

Additionally, DeepRegFinder is more parameter efficient than EP-DNN. The CNN and RNN have about 26K and 12K weight parameters, respectively, while EP-DNN has about 500K weight parameters. 

**References**:
1. Kim,S.G. et al. (2016) EP-DNN: A Deep Neural Network-Based Global Enhancer Prediction Algorithm. Scientific Reports, 6, 38433.
2. Rajagopal,N. et al. (2013) RFECS: A Random-Forest Based Algorithm for Enhancer Identification from Chromatin State. PLoS Comput Biol, 9, e1002968.

### Visualization of first convolution layer filters
One of the advantages of using convolution layers is that they tend to be easy to interpret. The following figure shows the activation patterns of the 32 1D filters of the first convolution layer of the CNN for the five classes (left panel) and the weights for a few example filters (right panel). As you can see, the activations show clear clusterings among the 32 filters and some filters are distinctly associated with certain classes. For example, filter 12 is exclusively associated with the active enhancer class and there is a clear peak detector for the H3K27ac histone mark.

<img src="./figures/conv1%20filters%20acts%20and%20weights.png" alt="conv1_filters"/>

### Network structures
Here is the CNN structure (click to see enlarged version):

<img src="./figures/CNN_structure.png" alt="cnn_struct" width="500"/>

Here is the RNN structure (click to see enlarged version):

<img src="./figures/RNN_structure.png" alt="rnn_struct" width="200"/>

### Multi-class classification analysis
Previous studies tend to focus on binary classifications between enhancer (as positive) and the rest (as negative), where promoters are lumped into the negative class. This can lead to an issue that the performance to distinguish enhancers from other regulatory elements becomes unclear. 5-way classification is used in DeepRegFinder. The following confusion matrix is from CNN predictions on the K562 cell type. We can see that for the active enhancer and promoter classes, the classifier is fairly accurate. However, the poised enhancer and promoter classes are often mistaken with other classes (click to see enlarged version):

<img src="./figures/test_confusion_matrix_convnet.png" alt="k562_confmat_cnn" width="300"/>

## Run DeepRegFinder using the Docker image (recommended)
A Docker Image for DeepRegFinder is available at https://hub.docker.com/r/aarthir239/deepregfinder. Please read the documentation on the Docker Hub page to use the image.

## Installation
DeepRegFinder relies on Python 3 (>=3.6) so make sure that's the Python you are using. There are a number of dependencies that DeepRegFinder needs. You can install them as follows.

### Install dependencies using Anaconda
You may install the dependencies using [Anaconda](https://www.anaconda.com/). Download the project repository onto your workstation. Change into the downloaded repository and run the following command:

`conda env create -f environment.yaml`

This will create a conda environment called *deepregfinder*. Next, you may activate the environment using the following command:

`conda activate deepregfinder`

### Install dependencies using pip
First, download the project repository to your workstation. The dependencies and their versions in our development environment are listed in the `requirements.txt`. You may try to automatically install them by:

`pip install -r requirements.txt`

However, this approach may fail due to software incompatibility. In that case, you can manually install each package. If a particular version is incompatible or becomes unavailable, you may install the current default version and it shall work just fine.

### Install featureCounts
The preprocessing module relies on a program called `featureCounts` from the [Subread](http://subread.sourceforge.net/) package for short read counting. If you install the dependencies using Anaconda, the **Subread** package is already installed and you don't need to do anything. If you install the dependencies using pip, you'll need to install the **Subread** package manually and make sure `featureCounts` is in your `PATH`.

### Install DeepRegFinder
After all the dependencies have been installed, go to the project folder and run the following command to install DeepRegFinder:

`pip install -e .`

### About Operating Systems
DeepRegFinder has been tested under Linux and Mac. We never tested it under Windows. You may have to use a simulated terminal such as [Cygwin](https://www.cygwin.com/). 

## Running the pipeline
The pipeline has three modules: preprocessing, training and prediction. You can execute each module separately, which provides a lot of flexibility. The basic procedure for running each step is to first gather all the required input files, fill out a YAML configuration file and then run the corresponding program. We have provided example configuration files for you to easily follow. If you installed DeepRegFinder properly, the three `drfinder-xxx.py` scripts shall already be in your `PATH`. You can go to your own project folder and issue commands from there. Use the configuration files in DeepRegFinder's repository as your starting points.

### An example project
I understand how important it is to have an example for people to follow. Therefore I have created an example project with all annotations, bam files and configurations so that you can see how a project shall be structured. It can be accessed at this Google Dirve [folder](https://drive.google.com/drive/folders/1sW9KM9TnK6nqquf7nQniEpfTtiKtWVni?usp=sharing).

### Preprocessing
Fill out the configuration file: `preprocessing_data.yaml` and run this command:

`drfinder-preprocessing.py preprocessing_data.yaml <NAME OF OUTPUT FOLDER>`

To get histone mark ChIP-seq from ENCODE easily, a script (`create_histones_folder.py`) has been provided in the `scripts` folder. In the script, edit the section marked **edit the following** and run the Python script in background as follows:

`nohup python create_histones_folder.py &`

For your own ChIP-seq data, just follow the same file structure and put your BAM files under corresponding folders. Data for the folders `peak_lists` and `tfbs` may be obtained from [ENCODE](https://www.encodeproject.org/) and data for the `genome` folder may be obtained from [gencode](https://www.gencodegenes.org/). GRO-seq data can be found on [GEO](https://www.ncbi.nlm.nih.gov/geo/). You may have to process the GRO-seq data yourself to obtain the bam files to be used with DeepRegFinder.

After preprocessing finished, the following files will be generated and required for training and prediction:

Under `tensor_data` directory:
- all_datasets.pth.tar
- chann_stats.csv

Under `histone_data` directory:
- alltogether_notnormed.txt.gz
- alltogether_notnormed.txt.gz.tbi

Under `tss_data` directory:
- enhancer_slopped_tss.bed

Under `tpms_data` directory:
- final_tpms.bed

### Training
Fill out the configuration file: `training_data.yaml` and run this command:

`drfinder-training.py training_data.yaml <NAME OF OUTPUT FOLDER>`

The trained model, training result summary and confusion matrix are under the `model` directory.

### Prediction
Fill out the configuration file: `wg_prediction_data.yaml` and run this command:

`drfinder-prediction.py wg_prediction_data.yaml <NAME OF OUTPUT FOLDER>`

A prediction summary and the predicted enhancers and promoters are under the `predictions` directory.

### Running time
Approximate time to run the three modules (assume you have a not-too-old GPU and a multi-core CPU):
- Preprocessing: 2-8h
- Training: 5 min
- Prediction: 20 min

### Cleaning up disk space
DeepRegFinder may generate a lot of intermediate files that take up a large amount of disk space. If you are running short on disk space, you may want to delete them. Particularly, two directories - `genome_data` and `histone_data` use the most space. Feel safe to delete the `genome_data` directory. For the `histone_data` directory, only two files are needed: `alltogether_notnormed.txt.gz` and `alltogether_notnormed.txt.gz.tbi`; you may delete everything else.

# TERMS OF USE
All data is free to use for non-commercial purposes. For commercial use please contact [MSIP](https://www.ip.mountsinai.org/).


