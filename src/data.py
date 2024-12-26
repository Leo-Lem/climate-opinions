from pandas import read_csv, DataFrame
from os import path

from __params__ import DATA_PATH


def prepare() -> DataFrame:
    data = load()
    data.rename(columns={"message": "text", "tweetid": "id"}, inplace=True)
    data.set_index("id", inplace=True)
    # TODO: normalize labels to -1|0|1
    data.replace({"sentiment": {2, 1}}, inplace=True)
    # TODO: train/val split
    ...
    return data


def load() -> DataFrame:
    return read_csv(path.join(DATA_PATH, "data.csv"))
