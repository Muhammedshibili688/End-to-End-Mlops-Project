import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
import yaml
from src.logger import logging

def load_data(file_path: str)->pd.DataFrame:
    """"Load data from a csv file"""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV file %s",e)
        raise 
    except Exception as e:
        logging.error("Unexpected error occured %s", e)
        raise

def train_model(x_train: np.ndarray, y_train:np.ndarray)->pd.DataFrame:
    "train the Naive bayes model"
    try:
        clf = MultinomialNB(alpha = 1, fit_prior=True)
        clf.fit(x_train, y_train)
        logging.info("Model training completed")
        return clf
    except Exception as e:
        logging.error("Error during model training")
        raise

def save_model(model, file_path:str)->None:
    """Save the trained model to a file"""
    try:
        with open(file_path,'wb') as file:
            pickle.dump(model, file)
        logging.info("model saved to %s", file_path)
    except Exception as e:
        logging.error("Failed to complete the model training process: %s", e)
        print(f"Error: {e}")

def main():
    try:
        train_data = load_data("./datas/processed/train_bow.csv")
        x_train = train_data.iloc[:,:-1]#.values
        y_train = train_data.iloc[:,-1]#.values
        print("Class distribution:")
        print(y_train.value_counts(normalize=True))

        clf = train_model(x_train, y_train)
        save_model(clf, './models/model.pkl')
    except Exception as e:
        logging.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()