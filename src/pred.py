from os import path
from pandas import read_csv, DataFrame
from torch import load
from tqdm import tqdm

from __params__ import SAMPLE, DATA_PATH, OUT_PATH
from src.model import Bert


class BertPredictor:
    CORPUS_FILE = path.join(DATA_PATH,
                            f"{'sample-' if SAMPLE else ''}corpus.csv")

    def __init__(self, model: Bert):
        self.model = model
        self.__load_best__()

        self.FILE = path.join(OUT_PATH,
                              f"{'sample-' if SAMPLE else ''}{model.__class__.__name__}-predictions.csv")

    def __load_best__(self):
        """ Load the best model from file. """
        if path.exists(self.model.BEST_FILE) and (best := load(self.model.BEST_FILE, weights_only=False)).get("model") is not None:
            self.model.load_state_dict(best["model"])

    def __call__(self, message: str) -> int:
        """ Predict the sentiment of a message. """
        encoding = self.model.tokenizer.encode_plus(message,
                                                    add_special_tokens=True,
                                                    padding="max_length",
                                                    truncation=True,
                                                    return_token_type_ids=False,
                                                    return_attention_mask=True,
                                                    return_tensors="pt")
        prediction = self.model.predict(encoding["input_ids"],
                                        encoding["attention_mask"])
        return prediction.argmax(dim=1).item()

    def corpus(self) -> DataFrame:
        """ Predict the sentiment of each message in the corpus. """
        predictions = read_csv(self.CORPUS_FILE,
                               usecols=["message"],
                               dtype={"message": str})
        for i, message in tqdm(enumerate(predictions["message"]), total=len(predictions), desc="Predicting", unit="message"):
            predictions.at[i, "prediction"] = self(message)
            predictions["prediction"] = predictions["prediction"]\
                .astype("Int64")
        predictions["target"] = None
        predictions.to_csv(self.FILE, index=False)
        print(f"Stored predictions in '{self.FILE}'.")
