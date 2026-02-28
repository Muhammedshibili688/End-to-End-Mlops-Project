import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download("wordnet")
nltk.download("stopwords")

def preprocess_dataframe(df, col = 'text'):
    """
    Preprocessing dtaframe by appy preprocessing techniques to specific columns
    returns preprocessed DataFrame
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    # negation_words = {"not", "no", "nor", "dont", "didnt", "isnt", "wasnt", "shouldnt", "wouldnt", "couldnt"}
    # stop_words = stop_words - negation_words

    def preprocess_text(text):
        "Help to process single text"
        
        text = re.sub(r"https?://\S+|www\.\S+",'',text)
        
        text = ''.join((char for char in text if not char.isdigit()))
        
        text = text.lower()
        
        text = re.sub('[%s]' %re.escape(string.punctuation),'',text)
        text = text.replace(',', '')
        re.sub(r'\s+', '', text)

        text = ' '.join(word for word in text.split() if word not in stop_words)

        text =  ' '.join(lemmatizer.lemmatize(word) for word in text.split())
        return text
    
    # Applying preprocessing to specific column
    df[col] = df[col].apply(preprocess_text)

    # droping col with Nan values
    df = df.dropna(subset = [col])
    logging.info("Preprocessing completed")
    return df

def main():
    try:
        "Fetch the data from main data frame"
        train_data = pd.read_csv("./datas/raw/train.csv")
        test_data = pd.read_csv("./datas/raw/test.csv")
        logging.info("Data loaded properly.")

        # preprocessing
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        data_path = os.path.join('./datas/', 'interim')
        os.makedirs(data_path, exist_ok = True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index = False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index = False)

        logging.info("Processed data  saved to %s", data_path)

    except Exception as e:
        logging.error("Failed to complete Preprocessing: %s", e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()



