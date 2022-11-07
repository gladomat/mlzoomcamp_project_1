import numpy as np
import requests
import pandas as pd
import os
import json

SERVICE_URL = "http://localhost:3000/predict"

def sample_random_data(data_path: str):
    """Sample random data from the data path.
    
    :param data_path: The path to the data.
    :return: The dataframe.
    """
    i = np.random.randint(0, 20)
    fname = os.path.join(data_path, f"Diamonds_Prices2022_split_{i}.csv")

    df = pd.read_csv(fname)
    return df.sample(1)


def make_request_to_bento_service(df: pd.DataFrame):
    """Make a request to the BentoML service.
    
    :param df: The dataframe to make the request with.
    :return: The response.
    """

    response = requests.post(
        SERVICE_URL, 
        data=json.dumps(df.to_dict(orient="records")),
        headers={"Content-Type": "application/json"},)
    return response.json()

def main():
    """Main function."""
    data_path = "./data"
    df = sample_random_data(data_path)
    response = make_request_to_bento_service(df)
    print("Prediction", response)
    print("Actual", df["price"].values[0])


if __name__ == "__main__":
    main()