from torch import Tensor
from torch.nn import Module, Linear, Softmax
from transformers import BertConfig, BertModel, BertTokenizer


class Bert(Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.to_3_classes = Linear(hidden_size, 3)
        self.softmax = Softmax(dim=1)

    def predict(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.softmax(self.forward(input_ids, attention_mask))


class BlankBert(Bert):
    MODEL_NAME = "bert-base-uncased"

    def __init__(self, config=None):
        config = config or BertConfig.from_pretrained(self.MODEL_NAME)
        super().__init__(config.hidden_size)
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = BertModel.from_pretrained(self.MODEL_NAME, config=config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        pooled_output = outputs[1]
        logits = self.to_3_classes(pooled_output)
        return logits
