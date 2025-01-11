import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
from datetime import datetime

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    print(f"Metrics:\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}")
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.best_eval_loss = float('inf')
        self.best_model_dir = None

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs['metrics']['eval_loss']
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_model_dir = f"./results/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.tokenizer.save_pretrained(self.best_model_dir)
            self.model.save_pretrained(self.best_model_dir)
            print(f"New best model saved with eval_loss: {eval_loss:.4f}")

if __name__ == '__main__':
    freeze_support()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print('No GPU available, using the CPU instead.')

    df = pd.read_csv("social-media-sentiment-analysis/res/twitter_sentiment_data.csv")
    df = df[df['sentiment'] != 2]
    df = df.rename(columns={'sentiment': 'label', 'message': 'text'})
    df['text'] = df['text'].str.replace(r'https://t.co/\S+', '', regex=True)
    df['text'] = df['text'].str.lower()

    pd.set_option('display.max_colwidth', None)
    print(df.head(30))

    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    def tokenize_function(example):
        return tokenizer(example['text'], padding='max_length', truncation=True)

    def transform_labels(example):
        # Umwandlung der Labels: -1 -> 0, 0 -> 1, 1 -> 2
        example['label'] = example['label'] + 1
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    transformed_dataset = tokenized_datasets.map(transform_labels, batched=False)

    # Verwenden Sie train_test_split von scikit-learn mit festem Seed
    train_val_split = transformed_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    # Split validation set into validation and test sets
    val_test_split = val_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        adam_beta1=0.75,
        adam_beta2=0.85,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[SaveBestModelCallback(model, tokenizer)]
    )

    trainer.train()
    trainer.evaluate()

    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./results/model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(test_results)

    # Save training parameters and performance metrics
    params_file = os.path.join(model_dir, "training_params_and_metrics.txt")
    with open(params_file, "w") as f:
        f.write("Training Parameters:\n")
        f.write(f"Batch size (train): {training_args.per_device_train_batch_size}\n")
        f.write(f"Batch size (eval): {training_args.per_device_eval_batch_size}\n")
        f.write(f"Number of epochs: {training_args.num_train_epochs}\n")
        f.write(f"Weight decay: {training_args.weight_decay}\n")
        f.write("\nPerformance Metrics:\n")
        f.write(f"Accuracy: {test_results['eval_accuracy']:.4f}\n")
        f.write(f"F1 Score: {test_results['eval_f1']:.4f}\n")
        f.write(f"Precision: {test_results['eval_precision']:.4f}\n")
        f.write(f"Recall: {test_results['eval_recall']:.4f}\n")