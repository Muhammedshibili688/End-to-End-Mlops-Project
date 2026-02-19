import boto3
import pandas as pd
from src.logger import logging
import logging
from io import StringIO

# Configure logging
# logging.Basic config(level = logging.info)
# logger = logging.getlogger(__name__)

class s3_operations:
    "Initialize s3 operations with aws credentials"
    def __init__(self, bucket_name, region="us-east-1"):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            region_name=region
        )
        logging.info("Data Ingestion from s3 bucket initiated")

    def fetch_file_from_s3(self, file_key):
        """
        Fetch file from s3 and perform  it as pandas dataframe
        !params file_key = s3.file_path (eg. data/data.csv)
        !return pandas datframe
        """
        try:    
            logging.info(f"Fetching file from {file_key} s3 bucket {self.bucket_name} ...")
            obj = self.s3_client.get_object(Bucket = self.bucket_name, Key = file_key)
            df = pd.read_csv(StringIO(obj["Body"].read().decode('utf-8')))
            logging.info(f"successfully fetched and loaded {file_key} from s3 bucket has {len(df)} records." )
            return df
        except Exception as e:
            logging.exception(f"Failed to fetch file from {file_key} from s3 {e}")
            return None

# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual AWS credentials and S3 details
#     BUCKET_NAME = "bucket-name"
#     AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
#     AWS_SECRET_KEY = "AWS_SECRET_KEY"
#     FILE_KEY = "data.csv"  # Path inside S3 bucket

#     data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame
