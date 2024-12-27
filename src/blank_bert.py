from os import path
from pandas import read_csv, DataFrame, Series
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import BertTokenizer, BertConfig, BertModel

from __params__ import EPOCHS, BLANK_MODEL, OUT_PATH, BATCH_SIZE

tokenizer: BertTokenizer = BertTokenizer.from_pretrained(BLANK_MODEL)
config: BertConfig = BertConfig.from_pretrained(BLANK_MODEL)
model: BertModel = BertModel.from_pretrained(BLANK_MODEL, config=config)


def finetune_bert(data: DataFrame) -> BertModel:
    encoded = encode(data)
    print(encoded.head())
    train = DataLoader(encoded, batch_size=BATCH_SIZE, shuffle=True)
    ...
    return model


def encode(data: DataFrame) -> DataFrame:
    """ Encode the tokens. """
    encoded = data
    encoded[["input_ids", "attention_mask"]] = data["message"].apply(lambda x: tokenizer.encode_plus(x,
                                                                                                     max_length=128,
                                                                                                     padding=True,
                                                                                                     truncation=True,
                                                                                                     return_token_type_ids=False,
                                                                                                     return_attention_mask=True,
                                                                                                     return_tensors="pt")).apply(Series)
    return encoded


def split(data: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """ Split the data into train (.8), validation (.1), and test (.1) sets randomly. """
    train = data.sample(frac=.8)
    data = data.drop(train.index)
    validation = data.sample(frac=.5)
    test = data.drop(validation.index)
    return train, validation, test
