from datetime import datetime
from datasets import Dataset
from os import path
from pandas import read_csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback

from __params__ import RESULTS_PATH, DATA_PATH, SEED, MODEL, BATCH_SIZE, EPOCHS, SAMPLE


def preprocess(tokenizer: AutoTokenizer) -> tuple[Dataset, Dataset, Dataset]:
    """ Load from csv, preprocess, tokenize, and split into train, val, test. """
    df = read_csv(path.join(DATA_PATH, f"{SAMPLE}data.csv"))\
        .rename(columns={'sentiment': 'label', 'message': 'text'})

    # remove factual label
    df = df[df["label"] != 2]

    # remove links
    df['text'] = df['text'].str.replace(r"https://t\.co/\S+", "",
                                        regex=True)

    # lower case
    df['text'] = df['text'].str.lower()

    # shift labels to positive numbers
    df = df.replace({"label": {1: 2, 0: 1, -1: 0}})

    preprocessed = Dataset.from_pandas(df)

    tokenized = preprocessed.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
                                 batched=True)

    train_val = tokenized.train_test_split(test_size=.2, seed=SEED)
    val_test = train_val["test"].train_test_split(test_size=.5, seed=SEED)
    return train_val["train"], val_test["train"], val_test["test"]
