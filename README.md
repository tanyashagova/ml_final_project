Final project for RS School Machine Learning course.

This project uses [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction).


## Usage
This package allows you to train model for solving the task of forest cover type prediction.
1. Clone this repository to your machine.
2. Download [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction), save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.13).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
 ![MLFlow experiments example](https://github.com/tanyashagova/ml_final_project/blob/main/mlflow_experiments_runs.png)

  ![Best parameters train](https://github.com/tanyashagova/ml_final_project/blob/main/best_param_train.png)