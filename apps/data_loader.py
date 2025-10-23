import pandas as pd
import numpy as np
import os

def data_load(path: str = f"{os.path.dirname(__file__)}/../data/customer_churn.csv") -> pd.DataFrame:

    """
    Load customer data from a CSV file.

    Parameters:
    path (str): The file path to the CSV file. Default is "../data/customer_data.csv".

    Returns:
    pd.DataFrame: A DataFrame containing the customer data.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    data = data_load()
    print(data)