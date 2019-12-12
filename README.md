# DeepRegFinder: *Deep* Learning based *Reg*ulatory Elements *Finder*
by Li Shen, George Wangensteen, Sarah Kim
Icahn School of Medicine at Mount Sinai

**DeepRegFinder** is a deep learning program used to identify transcriptional regulatory elements on the genome using histone mark ChIP-seq based on PyTorch. 

## Installation
DeepRegFinder relies on Python 3 (>=3.6). First, download the project repository to your workstation. After all the dependencies have been installed, go to the project folder and run the following command:

`pip install -e .`

The dependencies are listed in the `requirements.txt`. You may automatically install them by:

`pip install -r requirements.txt`

Or, you can manually install them.

## Running the pipeline
The pipeline has three modules: preprocessing, training and prediction. You shall follow the three steps 1-by-1. The basic procedure for running each step is to first get all the required input files organized, fill out a configuration file and then run the corresponding program. You don't have to run the programs under the DeepRegFinder's repository because they shall already be in your PATH. You can go to your own project folder and issue commands from there. Use the configuration files in DeepRegFinder's repository as your starting points.

### Preprocessing:
Fill out the configuration file: preprocessing_data.yaml and run this command:

`drfinder-preprocessing.py preprocessing_data.yaml \<NAME OF OUTPUT FOLDER\>`

### Training:
Fill out the configuration file: training_data.yaml and run this command:

`drfinder-training.py training_data.yaml \<NAME OF OUTPUT FOLDER\>`

### Prediction:
Fill out the configuration file: wg_prediction_data.yaml and run this command:

`drfinder-prediction.py wg_prediction_data.yaml \<NAME OF OUTPUT FOLDER\>`

### Running time
Approximate time to run the three modules (assume you have a not-too-old GPU and a multi-core CPU):
- Preprocessing: 2h
- Training: 5 min
- Prediction: 20 min




