import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
from src.logger import logging
import pickle

def load_params(params_path: str) -> dict:
    """"Load parameters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Parameters Retrieved from %s", params_path)
        return params
    
    except FileNotFoundError as e:
        logging.error("File Not Found at: %s", e)
        raise
    except yaml.YAMLError as e:
        logging.error("TAML error: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected Error Occured: %s",e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    "Load data from a csv"
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace = True)
        logging.info("Data filled and Nans filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected Error occured while loading data: %s", e)
        raise

def apply_bow(train_data: pd.DataFrame, test_data:pd.DataFrame, max_features: int) -> tuple:
    """Apply Count vectorizer to the data"""
    try:
        logging.info("Applying now...")
        vectorizer = CountVectorizer(max_features = max_features)

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

        x_train = train_data['review'].values
        x_test = test_data['review'].values
        y_train = train_data['sentiment'].values
        y_test = test_data['sentiment'].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        # pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        logging.info('Bag of words applied and data transformed')
        
        return train_df, test_df
    
    except Exception as e:
        logging.error("Error during Bag of words tranformation: %s", e)
        raise

def save_data(df: pd.DataFrame, file_path: str)-> None:
    """Saving DataFrame to csv"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        df.to_csv(file_path, index = False)
        logging.info('Data saved to: %s', file_path)

    except Exception as e:
        logging.error("Errro occured while saving the data: %s", e)
        raise

def main():
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./datas/interim/train_processed.csv')
        test_data = load_data('./datas/interim/test_processed.csv')

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./datas", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./datas", "processed", "test_bow.csv"))
    except Exception as e:
        logging.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()