import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
from datetime import datetime

train = False
model_dir = f"./results/model_20250104_234040"

df = pd.read_csv("social-media-sentiment-analysis/res/data.csv")
df = df[df['sentiment'] != 2]
df = df.rename(columns={'sentiment': 'label', 'message': 'text'})

dataset = Dataset.from_pandas(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

# Load the saved model
loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
loaded_tokenizer = BertTokenizer.from_pretrained(model_dir)

# Predict and display individual tweets with their predictions and target labels
for i in range(len(test_dataset)):
    tweet = test_dataset[i]['text']
    target_label = test_dataset[i]['label']
    inputs = loaded_tokenizer(tweet, return_tensors='pt', padding=True, truncation=True)
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    print(f"Tweet: {tweet}\nPrediction: {prediction}\nTarget Label: {target_label}\n")