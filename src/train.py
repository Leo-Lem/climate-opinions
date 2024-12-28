from os import path
from torch import no_grad, Tensor, save, load
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from __params__ import BATCH_SIZE, EPOCHS, OUT_PATH, LEARNING_RATE
from src.model import Bert
from src.data import ClimateOpinions


class BertTrainer:
    def __init__(self, model: Bert):
        self.model = model

        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = CrossEntropyLoss()

        self.checkpoint_file = path.join(
            OUT_PATH, f"{model.__class__.__name__}-checkpoint.pt")
        self.best_file = path.join(
            OUT_PATH, f"{model.__class__.__name__}-best.pt")

    def __save__(self, epoch: int, loss: float):
        """ Save the model, optimizer, and loss to a file. """
        if not path.exists(self.best_file) or loss < (best_loss := load(self.best_file, weights_only=False).get("loss", float("inf"))):
            save({
                "epoch": epoch,
                "loss": loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }, self.best_file)

        save({
            "epoch": epoch,
            "loss": loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, self.checkpoint_file)

    def __load__(self):
        """ Load the model, optimizer, and loss from a file. """
        if path.exists(self.checkpoint_file) and (checkpoint := load(self.checkpoint_file, weights_only=False)).get("model") is not None:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint["epoch"], checkpoint["loss"]
        return 0, float("inf")

    def __call__(self, train: ClimateOpinions, val: ClimateOpinions):
        epoch, loss = self.__load__()

        train_loader = DataLoader(train,
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val,
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        for epoch in (epochs := trange(epoch, EPOCHS, desc="Epoch", unit="epoch")):
            self.model.train()
            for input_ids, attention_mask, label in (batches := tqdm(train_loader, desc="Training", unit="batch", leave=False)):
                self.optimizer.zero_grad()
                output = self.model(input_ids, attention_mask)

                loss: Tensor = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()

                batches.set_postfix(loss=loss.item())

            self.model.eval()
            val_loss, val_correct = 0, 0
            with no_grad():
                for input_ids, attention_mask, label in (batches := tqdm(val_loader, desc="Validation", unit="batch", leave=False)):
                    prediction = self.model.predict(input_ids, attention_mask)

                    loss: Tensor = self.loss_fn(prediction, label)
                    val_loss += loss.item()
                    val_correct += sum(prediction == label).item()

                    batches.set_postfix(loss=loss.item())

            val_loss /= len(val_loader)
            val_accuracy = val_correct / len(val)
            epochs.set_postfix(loss=val_loss, accuracy=val_accuracy)

            self.__save__(epoch, val_loss)
