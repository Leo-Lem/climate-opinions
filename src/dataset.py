from pandas import read_csv, DataFrame
from os import path
from torch import Tensor, tensor, long
from torch.utils.data import Dataset
from transformers import BertTokenizer

from __params__ import DATA_PATH, OUT_PATH


class ClimateChangeOpinions(Dataset):
    DATA_FILE = path.join(DATA_PATH, "data.csv")
    PREPROCESSED_FILE = path.join(OUT_PATH, "preprocessed.csv")

    MAX_LENGTH = 128

    def __init__(self, model: str, data: DataFrame = None):
        self.model = model
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model)

        self.data = self.__preprocess__() if data is None else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        """ Return the input_ids, attention_mask and sentiment of the i-th message as a tensor. """
        input_ids, attention_mask = self.__encode__(self.data.at[i, "message"])
        target = self.__target__(self.data.at[i, "sentiment"])
        return input_ids, attention_mask, target

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
        preprocessed = data.replace({"sentiment": {2: 1}})
        preprocessed = preprocessed.replace({"sentiment": {1: 2, 0: 1, -1: 0}})
        preprocessed.to_csv(self.PREPROCESSED_FILE, index=True)
        return preprocessed

    def __encode__(self, message: str) -> tuple[Tensor, Tensor]:
        """ Encode a single message into input_ids and attention_mask. """
        encoding = self.tokenizer.encode_plus(message,
                                              add_special_tokens=True,
                                              max_length=self.MAX_LENGTH,
                                              padding="max_length",
                                              truncation=True,
                                              return_token_type_ids=False,
                                              return_attention_mask=True,
                                              return_tensors="pt")
        return encoding["input_ids"].flatten(), encoding["attention_mask"].flatten()

    def __target__(self, sentiment: int) -> Tensor:
        """ Return the target tensor for the sentiment. """
        return tensor(sentiment, dtype=long)

    def split(self, train_frac: float = .8, val_frac: float = .1, test_frac: float = .1) -> tuple["ClimateChangeOpinions", "ClimateChangeOpinions", "ClimateChangeOpinions"]:
        """ Split the dataset into three parts. """
        assert train_frac + val_frac + test_frac == 1
        assert 0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1

        train = self.data.sample(frac=train_frac)
        val = self.data.drop(train.index).sample(frac=val_frac/(1-train_frac))
        test = self.data.drop(train.index).drop(val.index)
        return ClimateChangeOpinions(self.model, train.reset_index(drop=True)), \
            ClimateChangeOpinions(self.model, val.reset_index(drop=True)), \
            ClimateChangeOpinions(self.model, test.reset_index(drop=True))
