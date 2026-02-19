import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

import os
import yaml
from sklearn.model_selection import train_test_split
import logging
from src.logger import logging
from src.connection import s3_connection

def load_params(params_path: str)-> dict:
    try:
        with open (params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug(f"Parameter file has been retrieved succesfully from {params_path}")
        return params

    except FileNotFoundError as e:
        logging.error("File not found at give file path %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML Error %s", e)
        raise
    except Exception as e:
        logging.error(f"Unexpected Error encountered while retrieving parameters file from {params_path} with Error :{e}")
        raise

def load_data(data_url: str)-> pd.DataFrame:
    "Loading csv data"
    try:
        df = pd.read_csv(data_url)
        logging.info("csv file loaded successfully from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failer to pase the csv file %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading csv data %s", e)
        raise

def preprocess_data(df: pd.DataFrame)->pd.DataFrame:
    "preprocess te data"
    try:
        logging.info("Preprocessing ...")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive':0, 'negative':1})
        logging.info("Preprocessing completed.")
        return final_df
        
    except KeyError as e:
        logging.error("Missing column in DataFrame %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error occured %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:
    "Saving the preprocessed and splitted data"
    try:
        logging.info("Saving...")
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index = False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index = False)
        logging.info('Train and Test csv saved to %s', raw_data_path)

    except Exception as e:
        logging.error("Unecpected error occured while saving the data: %s ", e)
        raise

def main():
    try:
        params = load_params(params_path = 'params.yaml')
        test_size = params['data_ingestion']['test_size'] # 0.2

        # load data
        # df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        
        s3 = s3_connection.s3_operations(
            'review-end-to-end-bucket',
            region = 'us-east-1'
            )
        df = s3.fetch_file_from_s3('data.csv')

        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size = test_size, random_state=42)
        save_data(train_data, test_data, data_path = './datas')

    except Exception as e:
        logging.error("Failed to complete data ingestion %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
