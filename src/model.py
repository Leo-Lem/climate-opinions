from torch import Tensor
from torch.nn import Module, Linear, Softmax
from transformers import BertConfig, BertModel

from __params__ import BLANK_MODEL


class Bert(Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.to_3_classes = Linear(hidden_size, 3)
        self.softmax = Softmax(dim=1)

    def predict(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.softmax(self.forward(input_ids, attention_mask))


class BlankBert(Bert):
    def __init__(self, config=BertConfig.from_pretrained(BLANK_MODEL)):
        super().__init__(config.hidden_size)
        self.model = BertModel.from_pretrained(BLANK_MODEL, config=config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        pooled_output = outputs[1]
        logits = self.to_3_classes(pooled_output)
        return logits
