# NextX: Forecasting Wind Speed for the Next X Hours

This repository contains various state-of-the-art models and custom models for predicting the next 12 hours of wind speed from a weather station. The aim of this project is to evaluate the performance of different models on the same dataset, as well as to test some novel models that have not been explored in existing research. At this point, all the models are designed in a multivariate-to-univariate format.

##  Dataset
The dataset used in this project is collected from a weather station, containing historical wind speed (target variable) and 11 other climatological data measurements with a 1-hour resolution, for 4 years from 2010 to 2013. The dataset is preprocessed and split into training , validation, and test sets for model training and evaluation. Data was collected from [here](https://www.ncei.noaa.gov/data/local-climatological-data/).

##  Hardware
All the models are trained/tested on a local machine with the following config:
- Model: Alienware Alienware m15 R4
- CPU: Intel Core i7-10870H
- GPU: NVIDIA GeForce RTX 3060 Laptop - 6 GB
- Memory: 16 GB - DDR4 SDRAM
- OS: Microsoft Windows 11 Professional (x64)

## Dependencies
All the models are implemented in python (v3.9.16) with the following key dependencies:

- pytorch: 2.0.0
- numpy: 1.24.1
- pandas: 1.5.3
- scikit-learn: 1.2.2
- tqdm: 4.65.0

All the dependecies are provided in the requirement.txt file.

### Features
[dataprep](www.TODO.com) config file can be manually adjusted for different settings. The deafult setting used for this project is:
- TARGET =  windspeed
- WINDOWSIZE = 168
- HORIZON = 12
- SKIP = 0 (in case you desire to skip some hours between lookback and the forecasting horizon)
- TEST_SIZE = 0.2
- VALID_SIZE = 0.2

## Models:
I will implement some custom models as well as some well-known published models and evaluate their performance.

### Benchmarks and SOTA models that will be implemented:
- [x] VanillaLSTM: Custom design
- [x] LSTMDENSE: Custom design
- [TODO] CNNLSTM: Custom design
- [TODO] [Vanilla-Transformer](https://arxiv.org/abs/1706.03762) (NeuIPS 2017)
- [TODO] [Informer](https://arxiv.org/abs/2012.07436) (AAAI 2021 Best paper)
- [TODO] [Autoformer](https://arxiv.org/abs/2106.13008) (NeuIPS 2021)
- [TODO] [FEDformer](https://arxiv.org/abs/2201.12740) (ICML 2022)

## Comparison of different models
<!-- We provide all experiment script files in `./scripts`: -->
| Model         | TEST MSE            | Number of parameters   | Number of epochs        | AVG epoch time (s)     |
| ------------- | --------------------| -----------------------| ------------------------| -----------------------|
| VanillaLSTM   | 0.0117              |                        | 52                      | 35.0                   |
| LSTMDENSE     | 0.0119              |                        | 34                      | 35.0                   |


## Getting Started

### Project structure

```
.
├── data
│   ├── training                    # (auto-generate)
│   ├── clean                       # (auto-generate)
│   └── raw                         # The raw data
│       └── WTH.csv                 # The weather forecast data
│
├── xplr_notebooks                  # Data exploration jupyter notebooks
│
├── config                          # config files for the pipeline
│
├── src                             # Contains the codes of the main pipeline
│   ├── utils                       # Includes utils.py module
│   │   ├── EarlyStopping.py
│   │   └── utils.py.csv
│   ├── clean
│   │   └── CleanWTH.py
│   ├── dataprep
│   └── models
│       └── univariate
│           └── LSTMbased.py
│
├── logs                    
│   ├── debug.log               
│   └── info.log
│
├── requirements.txt                # Required dependencies of the project
├── README.md
├── .gitignore
│
├── main.py                         # Entry point for the preprocesing pipeline
└── train.py                        # Entry point for the training pipeline
```
### Environment Requirements

First, please make sure you have installed Conda. Then, the environment can be installed using:
```
conda create -n nextX python=3.9.16
conda activate nextX
pip install -r requirements.txt
```

### Data Preparation

The required raw weather file (WTH.csv) is already placed in `./data/raw`.
In order to preprocess the data:
1. Open the terminal
2. Activate the environment
3. execute the following commands:
```
python main.py clean_wth                
python main.py prepare_target           
python main.py prepare_features
python main.py prepare_time_features
```

These commands will automatially clean and process the weather data. And due to the purpose of experimenting different form of datasets and models, the following datasets will be created in `data/training/wth/{windowsize}_{horizon}_{skip}` folder and for each {train/test/validation} set.
- x_feat_{}.npy
- x_{}.npy
- x_{}_fut_time.npy
- x_{}_hist_time.npy
- y_{}.npy

### Training Example
As this repository is intended for experimentation purposes, allowing for exploration of various model architectures and dataset combinations, the training process is not fully automated. Therefore, each training module can be executed following these steps:

1. Set the desired config in `config/training.cfg`
2. Import the desired model, in the corresponding module and set the model
3. Open the terminal and run the module. 
Example:
```
python trainLSTMBased.py
```







