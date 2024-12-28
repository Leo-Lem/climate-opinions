from torch import no_grad

from src.data import ClimateOpinions
from src.model import Bert


class BertEvaluator:
    def __init__(self, model: Bert):
        self.model = model

    def __call__(self, data: ClimateOpinions) -> tuple[float, float, float, float]:
        """ Evaluate the model using accuracy, precision, recall, and f1-score. """
        self.model.eval()
        with no_grad():
            results = [
                (self.model.predict(input_ids, attention_mask).argmax(dim=1), sentiment)
                for input_ids, attention_mask, sentiment in data
            ]

        accuracy = sum(pred == sentiment for pred, sentiment in results) \
            / len(results)
        precision = sum(pred == sentiment == 1 for pred, sentiment in results) \
            / sum(pred == 1 for pred, _ in results)
        recall = sum(pred == sentiment == 1 for pred, sentiment in results) \
            / sum(sentiment == 1 for _, sentiment in results)
        f1 = 2 * precision * recall / (precision + recall)

        return accuracy, precision, recall, f1
