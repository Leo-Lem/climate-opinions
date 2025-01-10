from pandas import read_csv, DataFrame
from os import path
from torch import Tensor, tensor, long
from torch.utils.data import Dataset
from transformers import BertTokenizer

from __params__ import DATA_PATH, OUT_PATH, SAMPLE, SEED


class ClimateOpinions(Dataset):
    DATA_FILE = path.join(DATA_PATH, f"{'sample-' if SAMPLE else ''}data.csv")
    PREPROCESSED_FILE = path.join(OUT_PATH,
                                  f"{'sample-' if SAMPLE else ''}preprocessed.csv")

    MAX_LENGTH = 150

    def __init__(self, tokenizer: BertTokenizer, data: DataFrame = None):
        self.tokenizer = tokenizer

        self.data = self.__preprocess__() if data is None else data
        self.encoded = self.__encode__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        """ Return the input_ids, attention_mask and sentiment of the i-th message as a tensor. """
        return self.encoded[i]

    def __preprocess__(self) -> DataFrame:
        """ Load from csv, remove factual label, and shift to positive numbers. """
        if path.exists(self.PREPROCESSED_FILE):
            return read_csv(self.PREPROCESSED_FILE,
                            dtype={"sentiment": int,
                                   "message": str,
                                   "tweetid": int})

        data = read_csv(self.DATA_FILE,
                        dtype={"sentiment": int,
                               "message": str,
                               "tweetid": int})
        preprocessed = data[data["sentiment"] != 2]
        preprocessed = preprocessed.replace({"sentiment": {1: 2, 0: 1, -1: 0}})
        preprocessed["message"] = preprocessed["message"].str.lower()
        preprocessed.to_csv(self.PREPROCESSED_FILE, index=True)
        return preprocessed

    def __encode__(self) -> list[tuple[Tensor, Tensor, int]]:
        """ Encode all messages into input_ids, attention_mask and sentiment. """
        encoding = self.tokenizer.batch_encode_plus(self.data["message"],
                                                    add_special_tokens=True,
                                                    max_length=self.MAX_LENGTH,
                                                    padding="max_length",
                                                    truncation=True,
                                                    return_token_type_ids=False,
                                                    return_attention_mask=True,
                                                    return_tensors="pt")
        encoded = [(input_ids, attention_mask, self.__target__(sentiment))
                   for input_ids, attention_mask, sentiment in zip(encoding["input_ids"],
                                                                   encoding["attention_mask"],
                                                                   self.data["sentiment"])]
        return encoded

    def __target__(self, sentiment: int) -> Tensor:
        """ Return the target tensor for the sentiment. """
        return tensor(sentiment, dtype=long)

    def split(self, train_frac: float = .8, val_frac: float = .1, test_frac: float = .1) -> tuple["ClimateOpinions", "ClimateOpinions", "ClimateOpinions"]:
        """ Split the dataset into three parts. """
        assert train_frac + val_frac + test_frac == 1
        assert 0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1

        train = self.data.sample(frac=train_frac, random_state=SEED)
        val = self.data.drop(train.index).sample(
            frac=val_frac/(1-train_frac), random_state=SEED)
        test = self.data.drop(train.index).drop(val.index)
        return ClimateOpinions(self.tokenizer, train.reset_index(drop=True)), \
            ClimateOpinions(self.tokenizer, val.reset_index(drop=True)), \
            ClimateOpinions(self.tokenizer, test.reset_index(drop=True))
