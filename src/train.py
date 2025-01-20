from datasets import Dataset
from os import path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from src.eval import compute_metrics
from __params__ import OUT_PATH, RESULTS_PATH, MODEL, BATCH_SIZE, EPOCHS

MODEL_DIR = path.join(RESULTS_PATH, MODEL)


class SaveBest(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model

        self.best_eval_loss = float('inf')

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs['metrics']['eval_loss']

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.tokenizer.save_pretrained(MODEL_DIR)
            self.model.save_pretrained(MODEL_DIR)

            print(
                f"New best model saved with eval_loss: {eval_loss:.4f}")


def train(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, train: Dataset, val: Dataset) -> Trainer:
    """ Train the model on the training dataset, validate on the validation dataset, and save the best model. """
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUT_PATH,
            eval_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS
        ),
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        callbacks=[SaveBest(model, tokenizer)]
    )
    if MODEL != "baseline":
        trainer.train(resume_from_checkpoint=path.exists(MODEL_DIR))
    return trainer
