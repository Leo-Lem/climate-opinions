from datasets import Dataset
from os import path
from pandas import DataFrame, read_csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from __params__ import RESULTS_PATH, MODEL, DATA_PATH, SAMPLE


def predict(file: str) -> DataFrame:
    """ Predict the sentiment of each message in a file. """
    DATA_FILE = path.join(DATA_PATH, f"{file}.csv")
    MODEL_DIR = path.join(RESULTS_PATH, MODEL)
    PRED_FILE = path.join(RESULTS_PATH, f"{SAMPLE}{file}-predictions.csv")

    df = read_csv(DATA_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    df["prediction"] = df["message"].apply(pipeline)

    df.to_csv(PRED_FILE, index=False)
    return df
