# MLZOOMCAMP Project 1

## Introduction

In this project I chose the diamnods dataset from Kaggle. The dataset contains 53940 observations and 
10 variables. The variables are: carat, cut, color, clarity, depth, table, price, x, y, z. 
The target variable is price. The dataset is available at https://www.kaggle.com/shivam2503/diamonds.

## Exploratory Data Analysis

The exploratory data analysis can be found in the notebook with the link 
[diamond_regression_with_eda.ipynb](https://github.com/gladomat/mlzoomcamp_project_1/blob/master/diamond_regression_with_eda.ipynb).

I also trained three different types of models: 
* random forest regressor,
* SDG regressor,
* xgboost regressor.

The results of the models can be found in the same notebook. I furthremore used hyperparamter tuning to improve the 
results of the models.

I used [`mlflow`](https://mlflow.org) to track the experiments. 

## Scripts
I then turned the preprocessing and training into scripts:
* `preprocessor.py` contains the preprocessing steps
* `training.py` contains the training steps

To test the scripts you need to first create a virtual environment and install the requirements:
```
pipenv install -r requirements.txt
```
Open a terminal and start the mlflow server:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then you can run the scripts in another terminal:
```
pipenv run python preprocessor.py
pipenv run python training.py
```

## Deployment using BentoML
I then deployed the model using BentoML. The deployment can be found under the `./deployment` directory.

You can test the deployment by building a bento service:
```
pipenv run bentoml build ./deployment
```
and then running the service:
```
pipenv run bentoml serve diamond-price-prediction:latest
```

In a different terminal you can then send a request to the service:
```
pipenv run python service_request.py
```

## Docker
I then created a docker image for the deployment. Make sure you have docker installed.

To build the docker image run:
```
pipenv run bentoml containerize diamond-price-prediction:latest
```

To run the docker image run:
```
docker run -it --rm -p 3000:3000 diamond-price-prediction:[docker-tag]
```

You can then try the service by sending a request:
```
pipenv run python service_request.py
```