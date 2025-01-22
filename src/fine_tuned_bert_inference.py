import pandas as pd
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

train = False
model_dir = f"./results/best_model_20250111_221832"

df = pd.read_csv("social-media-sentiment-analysis/res/bluesky.csv")
#df = df.rename(columns={ 'Text': 'text'})
df['text'] = df['text'].astype(str)

dataset = Dataset.from_pandas(df)

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")



def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)


test_dataset = dataset.map(tokenize_function, batched=True)

# Load the saved model
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Collect predictions
predictions = []

# Predict and collect individual tweets with their predictions and target labels
for i in range(len(test_dataset)):
    tweet = test_dataset[i]['text']
    inputs = loaded_tokenizer(tweet, return_tensors='pt', padding=True, truncation=True)
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    predictions.append({'tweet': tweet, 'prediction': prediction})
    if i%100 == 0:
        print(f"Predicted {i} tweets")

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions)
df['prediction'] = predictions_df['prediction']
df.to_csv('bluesky_prediction.csv', index=False)