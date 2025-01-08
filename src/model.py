from os import path
from torch import Tensor
from torch.nn import Module, Linear, Softmax
from transformers import BertConfig, BertModel, BertTokenizer, BertForSequenceClassification

from __params__ import RESULTS_PATH, OUT_PATH, SAMPLE, MODEL


class Bert(Module):
    def __init__(self, name: str, hidden_size: int):
        super().__init__()
        self.to_3_classes = Linear(hidden_size, 3)
        self.softmax = Softmax(dim=1)

        self.MODEL_NAME = name
        self.CHECKPOINT_FILE = path.join(OUT_PATH,
                                         f"{'sample-' if SAMPLE else ''}{self.__class__.__name__}-checkpoint.pt")
        self.BEST_FILE = path.join(RESULTS_PATH,
                                   f"{'sample-' if SAMPLE else ''}{self.__class__.__name__}-best.pt")

        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)

    def predict(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.softmax(self.forward(input_ids, attention_mask))

    @classmethod
    def create(self) -> "Bert":
        if MODEL == "baseline":
            return BaselineBert()
        elif MODEL == "blank":
            return BlankBert()
        elif MODEL == "sentiment":
            return SentimentBert()
        else:
            raise ValueError(f"Model {MODEL} not recognized.")


class BlankBert(Bert):
    def __init__(self, config=None):
        config = config or BertConfig.from_pretrained("bert-base-uncased")
        super().__init__("bert-base-uncased", config.hidden_size)
        self.model = BertModel.from_pretrained(self.MODEL_NAME, config=config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        pooled_output = outputs[1]
        logits = self.to_3_classes(pooled_output)
        return logits


class BaselineBert(BlankBert):
    def __init__(self):
        super().__init__()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return super().forward(input_ids, attention_mask)


class SentimentBert(Bert):
    def __init__(self, config=None):
        config = config or BertConfig\
            .from_pretrained("bert-base-uncased", num_labels=3)
        super().__init__("bert-base-uncased", config.hidden_size)
        self.model = BertForSequenceClassification\
            .from_pretrained(self.MODEL_NAME, config=config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        logits = outputs[0]
        return logits
