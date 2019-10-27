## Preprocessing Pipeline:
### To Run:
Fill out input file: preprocessing_data.yaml
In command line type:
`python preprocessing.py preprocessing_data.yaml\<NAME OF OUTPUT FOLDER\>`

Files necessary
-preprocessing.py
-preprocessing_data.yaml
-preprocessing_functions.py
-format_tss.sh
-index_file.sh
-log_transform.sh
-remove_header.sh



## Training/Net Pipeline:
### To Run:
#### Histone net:
Fill out: histone_net_data.yaml
In command line type:
`python histone_net.py histone_net_data.yaml\<NAME OF OUTPUT FOLDER\>`

Files necessary:
-histone_net.py
-histone_net_data.yaml
-nn_models.py
-net_functions.py

#### Sequence net: 
Fill out: sequence_net_data.yaml
In command line type:
`python sequence_net.py sequence_net_data.yaml\<NAME OF OUTPUT FOLDER\>`

Files necessary:
-sequence_net.py
-sequence_net_data.yaml
-nn_models.py
-net_functions.py

#### Combined net:
Fill out: combined_net_data.yaml
In command line type:
`python combined_net.py combined_net_data.yaml\<NAME OF OUTPUT FOLDER\>`

Files necessary:
-combined_net.py
-combined_net_data.yaml
-nn_models.py
-net_functions.py



## Prediction Pipeline:
### To Run:
Fill out: wg_prediction_data.yaml
In command line type:
`python wg_prediction.py wg_prediction_data.yaml\<NAME OF OUTPUT FOLDER\>`

Files necessary:
-wg_prediction.py
-wg_prediction_data.yaml
-wg_prediction_functions.py
-nn_models
-index_file.sh




