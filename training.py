import argparse
import os
import pickle

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

import xgboost as xgb
import bentoml


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path: str, num_trials: int):
    """
    Run hyperparameter optimization on xgboost model."""

    EXPERIMENT_NAME = "xgb-hyperopt"
    
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog(disable=True)
    #mlflow.log_artifacts("./artifacts")

    # Load pickled files.
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "X_val.pkl"))

    def objective(params):
        with mlflow.start_run() as run:
            artifact_uri = run.info.artifact_uri

            mlflow.set_tag("model", "xgb-regressor")
            mlflow.log_params(params)

            xgb_model = xgb.XGBRegressor(    
                n_estimators = params['n_estimators'],
                max_depth = int(params['max_depth']),
                learning_rate = params['learning_rate'],
                gamma = params['gamma'],
                min_child_weight = params ['min_child_weight'],
                subsample = params['subsample'],
                colsample_bytree = params['colsample_bytree'],
                objective='reg:squarederror'
            )

            cat_cols = ['cut', 'color', 'clarity']
            num_cols = ['carat', 'volume']

            transformer = ColumnTransformer(
            [('categorical', OneHotEncoder(sparse=False), cat_cols), 
            ('numerical', StandardScaler(), num_cols)]
            )

            full_pipeline = Pipeline([
                ('preprocessing', transformer),
                ('regression', xgb_model)
            ])
            tt = TransformedTargetRegressor(regressor=full_pipeline, transformer=StandardScaler())

            tt.fit(X_train, y_train)
            y_pred = tt.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # log model
            mlflow.sklearn.log_model(tt, "model")
            
            mlflow.log_metric("rmse", rmse)   

        return {"loss": rmse, "status": STATUS_OK}


    search_space = {
    'max_depth': hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
    'gamma': hp.quniform('gamma', 0, 0.5, 0.01),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    num_trials = 2
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

    # Search the best model by ascending order of metric score:
    from mlflow.entities import ViewType

    #MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    runs = client.search_runs(
        experiment_ids='2',    # Experiment ID we want
        filter_string="metrics.rmse < 600",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )    

    # Get model URI from runs:
    best_model_uri = runs[0].info.artifact_uri
    print(f"Best model URI: {best_model_uri}")
    
    # Save id of the best model to the bentoml repository.
    best_model_uri = best_model_uri[7:]
    full_dir = os.path.join(os.getcwd(), best_model_uri+"/model")
    bento_model = bentoml.mlflow.import_model("xgb-hyperopt", full_dir)


def trainer(data_path: str, max_evals: int=1000):
    """
    Run hyperparameter optimization on xgboost model."""

    run(data_path, max_evals)

if __name__ == "__main__":
    trainer("./processed_data")