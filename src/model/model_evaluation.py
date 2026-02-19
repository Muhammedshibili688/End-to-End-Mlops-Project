import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json
from src.logger import logging
import pickle
import os
import dagshub
import mlflow
import mlflow.sklearn

# code for production
# -----
# Setting Daghub crendiatals for mlflow tracking

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment is not set")

os.environ["CAPSTONE_TRACKING_USERNAME"] = dagshub_token
os.environ["CAPSTON_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Muhammedshibili688"
repo_name = "End-to-End-Mlops-Project"

# set up mlflow tracking  uri

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# ------------------------------------------------------------------------------------------
# Below code is for local use
# ------------------------------------------------------------------------------------------
# mlflow.set_tracking_uri("https://dagshub.com/Muhammedshibili688/End-to-End-Mlops-Project.mlflow")
# dagshub.init(repo_owner = "Muhammedshibili688", repo_name = "End-to-End-Mlops-Project",mlflow = True)

def configure_tracking():
    token = os.getenv("CAPSTONE_TEST")
    if not token:
        raise EnvironmentError("CAPSTONE_TEST is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(
        "https://dagshub.com/Muhammedshibili688/End-to-End-Mlops-Project.mlflow"
    )


def load_model(file_path:str):
    """Load the trained model from a file"""
    try:
        with open (file_path,"rb") as file:
            model = pickle.load(file)
        logging.info("Model loaded from %s", file_path)
        return model
    except FileNotFoundError as e:
        logging.error("File not found on given file path : %s ", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error founded : %s ", e)
        raise

def load_data(file_path:str)-> pd.DataFrame:
    """Load the trained model from a file"""
    try:
        df = pd.read_csv(file_path)
        logging.info("data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error("faile tp parse the CSV file : %s ", e)
        raise
    except Exception as e:
        logging.error("Unexpected error founded while loading data : %s ", e)
        raise

def evaluate_model(clf, x_test:np.array, y_test:np.array)-> dict:
    "Evaluating the model and return the evaluation metrics"
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall":recall,
            "auc": auc
        }
        logging.info("Model Metrics Calculated")
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics:dict, file_path:str)-> None:
    """Save the evaluation metrics to a JSON file"""
    try:
        with open(file_path,"w") as file:
            json.dump(metrics, file,indent = 4)
        logging.info("Metrics saved to %s: ", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the metrics: %s", e)
        raise

def save_model_info(run_id:str, model_uri:str, file_path:str)->None:
    "Saving the model"
    try:
        model_info = {"run_id" : run_id, "model_uri": model_uri}
        with open (file_path,'w') as file:
            json.dump(model_info, file, indent = 4)
        logging.debug("Model info saved into : %s", file_path)

    except Exception as e:
        logging.error("Error occured while saving the model info: %s", e)
        raise

def main():
    configure_tracking()
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run: # Starting mlflow run
        try:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./datas/processed/test_bow.csv")
            
            x_test = test_data.iloc[:,:-1]
            y_test = test_data.iloc[:,-1]
            metrics = evaluate_model(clf, x_test, y_test)

            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to mlflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, "get_params"):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log model to mlflow
            model_info = mlflow.sklearn.log_model(clf, name = "model")

            # Save the mode info
            save_model_info(run.info.run_id, model_info.model_uri, "reports/experiment_info.json")

            # Log the metrics file to MLflow
            mlflow.log_artifact("reports/metrics.json")

        except Exception as e:
            logging.error("Failed to complete the model evaluations : %s", e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()