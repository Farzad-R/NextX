# NextX: Forecasting Wind Speed for the Next 12 Hours

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
- [x] VanillaLSTM: custom architecture
- [x] LSTMDENSE: custom architecture
- [TODO] CNNLSTM: custom architecture
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
### Environment Requirements

First, please make sure you have installed Conda. Then, the environment can be installed using:
```
conda create -n nextX python=3.9.16
conda activate nextX
pip install -r requirements.txt
```

### Data Preparation

You can obtain all the nine benchmarks from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) provided in Autoformer. All the datasets are well pre-processed and can be used easily.

```
mkdir data
cd data
mkdir raw
```

**Please put them in the `./data/raw` directory**

### Training Example
- 






