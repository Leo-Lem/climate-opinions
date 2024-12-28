from torch import no_grad, Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from __params__ import BATCH_SIZE, EPOCHS


def train(model: Module, train: Dataset, validation: Dataset):
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    for epoch in (epochs := trange(EPOCHS, desc="Epoch", unit="epoch")):
        model.train()
        for input_ids, attention_mask, label in (batches := tqdm(train_loader, desc="Training", unit="batch", leave=False)):
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)

            loss: Tensor = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            batches.set_postfix(loss=loss.item())

        model.eval()
        val_loss, val_correct = 0, 0
        with no_grad():
            for input_ids, attention_mask, label in (batches := tqdm(val_loader, desc="Validation", unit="batch", leave=False)):
                prediction = model.predict(input_ids, attention_mask)

                loss: Tensor = loss_fn(prediction, label)
                val_loss += loss.item()
                val_correct += sum(prediction == label).item()

                batches.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(validation)
        epochs.set_postfix(loss=val_loss, accuracy=val_accuracy)
