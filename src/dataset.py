from pandas import read_csv, DataFrame, Series
from os import path
from torch import Tensor, tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer

from __params__ import DATA_PATH, OUT_PATH


class ClimateChangeOpinions(Dataset):
    DATA_FILE = path.join(DATA_PATH, "data.csv")
    PREPROCESSED_FILE = path.join(OUT_PATH, "preprocessed.csv")
    ENCODED_FILE = path.join(OUT_PATH, "encoded.csv")

    MAX_LENGTH = 128

    def __init__(self, model: str, data: DataFrame = None):
        self.model = model
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model)

        self.data = self.__preprocess__() if data is None else data

        if ("input_ids", "attention_mask") not in self.data.columns:
            self.data = self.__encode__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        """ Return the input_ids, attention_mask and sentiment of the i-th message as a tensor. """
        return tensor(self.data.loc[i, "input_ids"]), \
            tensor(self.data.loc[i, "attention_mask"]), \
            tensor(self.data.loc[i, "sentiment"])

    def __preprocess__(self) -> DataFrame:
        """ Load from csv and remove factual label. """
        if path.exists(self.PREPROCESSED_FILE):
            return read_csv(self.PREPROCESSED_FILE, dtype={"sentiment": int,
                                                           "message": str,
                                                           "tweetid": int})

        preprocessed = read_csv(self.DATA_FILE, dtype={"sentiment": int,
                                                       "message": str,
                                                       "tweetid": int})
        preprocessed.replace({"sentiment": {2: 1}}, inplace=True)
        preprocessed.to_csv(self.PREPROCESSED_FILE)
        return preprocessed

    def __encode__(self) -> DataFrame:
        """ Encode the messages using the tokenizer. """
        if path.exists(self.ENCODED_FILE):
            return read_csv(self.ENCODED_FILE, dtype={"sentiment": int,
                                                      "message": str,
                                                      "tweetid": int,
                                                      "input_ids": Tensor,
                                                      "attention_mask": Tensor})

        encoded = self.data.copy()
        encoded[["input_ids", "attention_mask"]] = self.data["message"]\
            .apply(lambda x: self.tokenizer.encode_plus(x,
                                                        add_special_tokens=True,
                                                        max_length=self.MAX_LENGTH,
                                                        padding=True,
                                                        truncation=True,
                                                        return_token_type_ids=False,
                                                        return_attention_mask=True,
                                                        return_tensors="pt"))\
            .apply(Series)
        encoded.to_csv(self.ENCODED_FILE)
        return encoded

    def split(self, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1) -> tuple["ClimateChangeOpinions", "ClimateChangeOpinions", "ClimateChangeOpinions"]:
        """ Split the dataset into three parts. """
        assert train_frac + val_frac + test_frac == 1
        assert 0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1

        train = self.data.sample(frac=train_frac)
        val = self.data.drop(train.index).sample(frac=val_frac/(1-train_frac))
        test = self.data.drop(train.index).drop(val.index)
        return ClimateChangeOpinions(self.model, train), \
            ClimateChangeOpinions(self.model, val), \
            ClimateChangeOpinions(self.model, test)
