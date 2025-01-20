from datasets import Dataset
from os import path
from pandas import DataFrame, read_csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from __params__ import RESULTS_PATH, MODEL, DATA_PATH

MODEL_FILE = path.join(RESULTS_PATH, MODEL)


def predict(file: str) -> DataFrame:
    """ Predict the sentiment of each message in a file. """
    df = read_csv(path.join(DATA_PATH, f"{file}.csv"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_FILE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_FILE)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    df["prediction"] = df["message"].apply(pipeline)

    return df


# class BertPredictor:
#     CORPUS_FILE = path.join(DATA_PATH,
#                             f"{'sample-' if SAMPLE else ''}corpus.csv")

#     def __init__(self, model: Bert):
#         self.model = model
#         self.__load_best__()

#         self.FILE = path.join(RESULTS_PATH,
#                               f"{'sample-' if SAMPLE else ''}{model.__class__.__name__}-predictions.csv")

#     def __load_best__(self):
#         """ Load the best model from file. """
#         if path.exists(self.model.BEST_FILE) and (best := load(self.model.BEST_FILE, weights_only=False)).get("model") is not None:
#             self.model.load_state_dict(best["model"])

#     def __call__(self, message: str) -> int:
#         """ Predict the sentiment of a message. """
#         encoding = self.model.tokenizer.encode_plus(message,
#                                                     add_special_tokens=True,
#                                                     padding="max_length",
#                                                     truncation=True,
#                                                     return_token_type_ids=False,
#                                                     return_attention_mask=True,
#                                                     return_tensors="pt")
#         prediction = self.model.predict(encoding["input_ids"],
#                                         encoding["attention_mask"])
#         return prediction.argmax(dim=1).item()

#     def corpus(self) -> DataFrame:
#         """ Predict the sentiment of each message in the corpus. """
#         predictions = read_csv(self.CORPUS_FILE,
#                                usecols=["message"],
#                                dtype={"message": str})
#         messages = predictions["message"].tolist()
#         results = []

#         for i in trange(0, len(messages), BATCH_SIZE, desc="Predicting", unit="batch", leave=False):
#             batch = messages[i:i+BATCH_SIZE]
#             encodings = self.model.tokenizer(batch,
#                                              padding=True,
#                                              truncation=True,
#                                              return_tensors="pt")
#             batch_predictions = self.model.predict(encodings["input_ids"].to(DEVICE),
#                                                    encodings["attention_mask"].to(DEVICE))
#             results.extend(batch_predictions.argmax(dim=1).tolist())

#         predictions["prediction"] = results
#         predictions["prediction"] = predictions["prediction"].astype("Int64")
#         predictions["actual"] = None
#         predictions.to_csv(self.FILE, index=False)
#         print(f"Stored predictions in '{self.FILE}'.")
