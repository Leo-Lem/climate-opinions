from torch import Tensor
from torch.nn import Module, Linear, Softmax
from transformers import BertConfig, BertModel

from __params__ import BLANK_MODEL


class BlankBert(Module):
    def __init__(self):
        super().__init__()
        self.config: BertConfig = BertConfig.from_pretrained(BLANK_MODEL)
        self.model = BertModel.from_pretrained(BLANK_MODEL, config=self.config)
        self.to_3_classes = Linear(self.config.hidden_size, 3)
        self.softmax = Softmax(dim=1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        pooled_output = outputs[1]
        logits = self.to_3_classes(pooled_output)
        return self.softmax(logits)

    def predict(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.softmax(self.forward(input_ids, attention_mask))
