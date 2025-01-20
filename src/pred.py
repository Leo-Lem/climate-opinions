from os import path
from pandas import DataFrame, read_csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm

from __params__ import RESULTS_PATH, MODEL_NAME, DATA_PATH, SAMPLE


def predict(file: str) -> DataFrame:
    """ Predict the sentiment of each message in a file. """
    DATA_FILE = path.join(DATA_PATH, f"{SAMPLE}{file}.csv")
    MODEL_DIR = path.join(RESULTS_PATH, MODEL_NAME)
    PRED_FILE = path.join(RESULTS_PATH, f"{SAMPLE}{file}-predictions.csv")

    df = read_csv(DATA_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    print("Predicting…")
    # truncate to max length of model
    df["prediction"] = tqdm(pipeline(
        df["text"].apply(lambda x: x[:tokenizer.model_max_length]).tolist()))
    df["score"] = df["prediction"]\
        .apply(lambda x: x["score"])
    df["prediction"] = df["prediction"].apply(lambda x: x["label"]
                                              .replace("LABEL_", "")
                                              .replace("NEG", "0")
                                              .replace("NEU", "1")
                                              .replace("POS", "2"))

    df.to_csv(PRED_FILE, index=False)

    print(f"Predictions saved to {PRED_FILE}.")

    return df
