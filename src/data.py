from pandas import read_csv, DataFrame, Series
from os import path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from __params__ import DATA_PATH, OUT_PATH, BLANK_MODEL


class ClimateChangeOpinions(Dataset):
    DATA_FILE = path.join(DATA_PATH, "data.csv")
    PREPROCESSED_FILE = path.join(OUT_PATH, "preprocessed.csv")

    TOKENIZER: BertTokenizer = BertTokenizer.from_pretrained(BLANK_MODEL)

    MAX_LENGTH = 128

    def __init__(self, data: DataFrame = None):
        if data is not None:
            self.data = data
        else:
            self.data = self.__preprocess__() \
                if not path.exists(self.PREPROCESSED_FILE) \
                else self.__load__()

        if ("input_ids", "attention_mask") not in self.data.columns:
            self.__encode__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def __load__(self) -> DataFrame:
        return read_csv(self.PREPROCESSED_FILE)

    def __preprocess__(self) -> DataFrame:
        data = read_csv(self.DATA_FILE,
                        dtype={"sentiment": int,
                               "message": str,
                               "tweetid": int})
        data.set_index("tweetid", inplace=True)
        # remove factual classification
        data.replace({"sentiment": {2: 1}}, inplace=True)
        data.to_csv(self.PREPROCESSED_FILE)
        return data

    def __encode__(self):
        self.data[["input_ids", "attention_mask"]] = self.data["message"]\
            .apply(lambda x: self.TOKENIZER.encode_plus(x,
                                                        max_length=self.MAX_LENGTH,
                                                        padding=True,
                                                        truncation=True,
                                                        return_token_type_ids=False,
                                                        return_attention_mask=True,
                                                        return_tensors="pt"))\
            .apply(Series)

    def loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """ Return a DataLoader for the dataset. """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def split(self, train_frac: float = 0.8, val_frac: float = 0.1, test_frac: float = 0.1) -> tuple["ClimateChangeOpinions", "ClimateChangeOpinions", "ClimateChangeOpinions"]:
        """ Split the dataset into three parts. """
        assert train_frac + val_frac + test_frac == 1
        assert 0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1

        train = self.data.sample(frac=train_frac)
        val = self.data.drop(train.index).sample(frac=val_frac/(1-train_frac))
        test = self.data.drop(train.index).drop(val.index)
        return ClimateChangeOpinions(train), ClimateChangeOpinions(val), ClimateChangeOpinions(test)
