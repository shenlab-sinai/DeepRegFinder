# DeepRegFinder: *Deep* Learning based *Reg*ulatory Elements *Finder*
by Li Shen, Aarthi Ramakrishnan, George Wangensteen, Sarah Kim

Icahn School of Medicine at Mount Sinai, New York, NY, USA

**DeepRegFinder** is a deep learning based program to identify DNA regulatory elements using ChIP-seq. It uses the deep learning framework PyTorch. 

## Installation
DeepRegFinder relies on Python 3 (>=3.6) so make sure that's the Python you are using. There are a number of dependencies that DeepRegFinder needs. You can install them as follows.

### Install dependencies using Anaconda (recommended)
You may install the dependencies using [Anaconda](https://www.anaconda.com/). Download the project repository onto your workstation. Change into the downloaded repository and run the following command:

`conda env create -f environment.yaml`

This will create a conda environment called *deepregfinder*. Next, you may activate the environment using the following command:

`conda activate deepregfinder`

### Install dependencies using pip
First, download the project repository to your workstation. The dependencies and their versions in our development environment are listed in the `requirements.txt`. You may try to automatically install them by:

`pip install -r requirements.txt`

However, this approach may fail due to software incompatibility. In that case, you can manually install each package. If a particular version is incompatible or becomes unavailable, you may install the current default version and it shall work just fine.

### Install DeepRegFinder
After all the dependencies have been installed, go to the project folder and run the following command to install DeepRegFinder:

`pip install -e .`

## Running the pipeline
The pipeline has three modules: preprocessing, training and prediction. You can execute each module separately, which provides a lot of flexibility. The basic procedure for running each step is to first gather all the required input files, fill out a YAML configuration file and then run the corresponding program. We have provided example configuration files for you to easily follow. If you installed DeepRegFinder properly, the three `drfinder-xxx.py` scripts shall already be in your `PATH`. You can go to your own project folder and issue commands from there. Use the configuration files in DeepRegFinder's repository as your starting points.

### An example project
I understand how important it is to have an example for people to follow. Therefore I have created an example project with all annotations, bam files and configurations so that you can see how a project shall be structured. It can be accessed at this Google Drive [folder](https://drive.google.com/drive/folders/1sW9KM9TnK6nqquf7nQniEpfTtiKtWVni?usp=sharing).

### Preprocessing
Fill out the configuration file: `preprocessing_data.yaml` and run this command:

`drfinder-preprocessing.py preprocessing_data.yaml <NAME OF OUTPUT FOLDER>`

To get histone mark ChIP-seq from ENCODE easily, a script (`create_histones_folder.py`) has been provided in the `scripts` folder. In the script, edit the section marked **edit the following** and run the Python script in background as follows:

`nohup python create_histones_folder.py &`

For your own ChIP-seq data, just follow the same file structure and put your BAM files under corresponding folders. Data for the folders `peak_lists` and `tfbs` may be obtained from [ENCODE](https://www.encodeproject.org/) and data for the `genome` folder may be obtained from [gencode](https://www.gencodegenes.org/). GRO-seq data can be found on [GEO](https://www.ncbi.nlm.nih.gov/geo/). You may have to process the GRO-seq data yourself to obtain the bam files to be used with DeepRegFinder.

### Training
Fill out the configuration file: `training_data.yaml` and run this command:

`drfinder-training.py training_data.yaml <NAME OF OUTPUT FOLDER>`

### Prediction
Fill out the configuration file: `wg_prediction_data.yaml` and run this command:

`drfinder-prediction.py wg_prediction_data.yaml <NAME OF OUTPUT FOLDER>`

### Running time
Approximate time to run the three modules (assume you have a not-too-old GPU and a multi-core CPU):
- Preprocessing: 2-8h
- Training: 5 min
- Prediction: 20 min




