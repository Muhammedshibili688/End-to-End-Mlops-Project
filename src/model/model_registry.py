import os
import mlflow
import logging
from src.logger import logging
import dagshub
import json

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# For production use
# set up dagshub credentials

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Muhammedshibili688"
repo_name = "End-to-End-Mlops-Project"

# set up mlflow tracking  uri
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# ---------------------------------------------------------------------------
# For local Run
# ----------------------------------------------------------------------------

# mlflow.set_tracking_uri("https://dagshub.com/Muhammedshibili688/End-to-End-Mlops-Project.mlflow")
# dagshub.init(repo_owner = "Muhammedshibili688", repo_name = "End-to-End-Mlops-Project",mlflow = True)

# ----------------------------------------------------------------------------

def load_model_info(file_path:str)-> dict:
    """load the model info from a JSON file"""
    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)
        logging.info("Model info loaded from : %s", file_path)
        return model_info
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error found : %s", e)
        raise

def register_model(model_name:str, model_info:dict):
    """Register model to the mlflow registry"""
    try:
        # model_url = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_info["model_uri"], model_name)

        # Transition the model to "staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Staging"
        )
        logging.debug(f"Model {model_name} version {model_version.version} registered and transitioned to staging.")
    except Exception as e:
        logging.error("Error during model regisration: %s", e)
        raise

def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = 'my_model'
        register_model(model_name, model_info)
    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()