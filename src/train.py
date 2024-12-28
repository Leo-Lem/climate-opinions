from os import path
from torch import no_grad, Tensor, save, load
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from __params__ import EPOCHS, BATCH_SIZE, OUT_PATH, SAMPLE
from src.model import Bert
from src.data import ClimateOpinions


class BertTrainer:
    LEARNING_RATE = 5e-5

    def __init__(self, model: Bert):
        self.model = model

        self.optimizer = AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        self.loss_fn = CrossEntropyLoss()

    def __save__(self, epoch: int, loss: float):
        """ Save the model, optimizer, and loss to a file. """
        data = {
            "epoch": epoch,
            "loss": loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        if not path.exists(self.model.BEST_FILE) or loss < load(self.model.BEST_FILE, weights_only=False).get("loss", float("inf")):
            save(data, self.model.BEST_FILE)
        save(data, self.model.CHECKPOINT_FILE)

    def __load__(self) -> tuple[int, float]:
        """ Load the model, optimizer, and loss from a file. """
        if path.exists(self.model.CHECKPOINT_FILE) and (checkpoint := load(self.model.CHECKPOINT_FILE, weights_only=False)).get("model") is not None:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint["epoch"]+1, checkpoint["loss"]
        return 0, float("inf")

    def __call__(self, train: ClimateOpinions, val: ClimateOpinions):
        epoch, loss = self.__load__()

        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val, batch_size=BATCH_SIZE)

        for epoch in (epochs := trange(epoch, EPOCHS, initial=epoch, total=EPOCHS, desc="Epoch", unit="epoch")):
            self.model.train()
            for input_ids, attention_mask, label in (batches := tqdm(train_loader, desc="Training", unit="batch", leave=False)):
                self.optimizer.zero_grad()
                output = self.model(input_ids, attention_mask)

                loss: Tensor = self.loss_fn(output, label)
                loss.backward()
                self.optimizer.step()

                batches.set_postfix(loss=loss.item())

            self.model.eval()
            val_loss = 0
            with no_grad():
                for input_ids, attention_mask, label in (batches := tqdm(val_loader, desc="Validation", unit="batch", leave=False)):
                    prediction = self.model.predict(input_ids, attention_mask)

                    loss: Tensor = self.loss_fn(prediction, label)
                    val_loss += loss.item()

                    batches.set_postfix(loss=loss.item())

            val_loss /= len(val_loader)
            epochs.set_postfix(loss=val_loss)

            epochs.set_description("Epoch (Saving…)")
            self.__save__(epoch, val_loss)
            epochs.set_description("Epoch")
