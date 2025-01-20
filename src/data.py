from datetime import datetime
from datasets import Dataset
from os import path
from pandas import read_csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback

from __params__ import RESULTS_PATH, DATA_PATH, SEED, MODEL, BATCH_SIZE, EPOCHS


def preprocess(tokenizer: AutoTokenizer) -> tuple[Dataset, Dataset, Dataset]:
    """ Load from csv, preprocess, tokenize, and split into train, val, test. """
    df = read_csv(path.join(DATA_PATH, "data.csv"))\
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


class ClimateSentimentModel:
    """ A model for sentiment analysis on climate change posts. """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

        self.train, self.val, self.test = self.preprocess()

    def trainer(self) -> Trainer:
        class SaveBest(TrainerCallback):
            def __init__(self, model, tokenizer):
                self.MODEL_DIR = path.join(
                    RESULTS_PATH, f"{MODEL}-{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                self.tokenizer = tokenizer
                self.model = model

                self.best_eval_loss = float('inf')

            def on_evaluate(self, args, state, control, **kwargs):
                eval_loss = kwargs['metrics']['eval_loss']

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.tokenizer.save_pretrained(self.MODEL_DIR)
                    self.model.save_pretrained(self.MODEL_DIR)

                    print(
                        f"New best model saved with eval_loss: {eval_loss:.4f}")

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=RESULTS_PATH,
                eval_strategy="epoch",
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=EPOCHS
            ),
            train_dataset=self.train,
            eval_dataset=self.val,
            compute_metrics=self.metrics,
            callbacks=[SaveBest(self.model, self.tokenizer)]
        )
