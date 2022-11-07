import numpy as np
import bentoml
from bentoml.io import JSON
import pandas as pd


BENTO_MODEL_TAG = "xgb-hyperopt:latest"

model_ref = bentoml.mlflow.get(BENTO_MODEL_TAG)
model_runner = model_ref.to_runner()

svc = bentoml.Service("diamond-price-prediction", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(json_data):
    df = pd.DataFrame(json_data, index=[0])
    # Preprocess the data
    df['volume'] = df['x'] * df['y'] * df['z']
    # Drop unnecessary columns
    df.drop(['price', 'x', 'y', 'z', 'depth', 'table', 'Unnamed: 0'], 
            axis=1, inplace=True)
    
    return model_runner.predict.run(df)