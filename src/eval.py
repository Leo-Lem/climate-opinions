from pandas import DataFrame
from os import path
from datetime import datetime
from torch import no_grad, load
from torch.utils.data import DataLoader
from tqdm import tqdm

from __params__ import RESULTS_PATH, SAMPLE, BATCH_SIZE
from src.data import ClimateOpinions
from src.model import Bert


class BertEvaluator:
    FILE = path.join(RESULTS_PATH,
                     f"{'sample-' if SAMPLE else ''}evaluation.csv")

    def __init__(self, model: Bert):
        self.model = model
        self.__load_best__()

    def __load_best__(self):
        """ Load the best model from file. """
        if path.exists(self.model.BEST_FILE) and (best := load(self.model.BEST_FILE, weights_only=False)).get("model") is not None:
            self.model.load_state_dict(best["model"])

    def __store__(self, accuracy: float, precision: dict[int, float], recall: dict[int, float], f1: dict[int, float]):
        """ Store the evaluation results. """
        DataFrame({
            "Model": [self.model.__class__.__name__],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
            "Timestamp": [datetime.now().strftime("%Y-%m-%d/%H:%M:%S")]
        }).to_csv(self.FILE, mode="a", header=not path.exists(self.FILE), index=False)
        print(f"Stored evaluation results in '{self.FILE}'.")

    def __call__(self, data: ClimateOpinions) -> tuple[float, dict[int, float], dict[int, float], dict[int, float]]:
        """ Evaluate the model using accuracy, precision, recall, and f1-score. """
        loader = DataLoader(data, batch_size=BATCH_SIZE)
        self.model.eval()
        results = []
        with no_grad():
            for input_ids, attention_mask, label in tqdm(loader, desc='Evaluating', unit="batch", leave=False):
                pred = self.model.predict(input_ids, attention_mask)
                results.extend(
                    zip(pred.argmax(dim=1).tolist(), label.tolist()))

        # labels are 0 (negative), 1 (neutral) and 2 (positive)
        accuracy = sum(pred == label for pred, label in results) / len(results)
        precision = {0: 0, 1: 0, 2: 0}
        recall = {0: 0, 1: 0, 2: 0}
        f1 = {0: 0, 1: 0, 2: 0}
        for positive_label in [0, 1, 2]:
            predicted_positives = sum(
                pred == positive_label for pred, _ in results)
            precision[positive_label] = sum(pred == label == positive_label for pred, label in results) / predicted_positives\
                if predicted_positives > 0 else 0
            actual_positives = sum(
                label == positive_label for _, label in results)
            recall[positive_label] = sum(pred == label == positive_label for pred, label in results) / actual_positives\
                if actual_positives > 0 else 0
            f1[positive_label] = 2 * precision[positive_label] * recall[positive_label] / (precision[positive_label] + recall[positive_label])\
                if precision[positive_label] + recall[positive_label] > 0 else 0

        self.__store__(accuracy, precision, recall, f1)

        return accuracy, precision, recall, f1
