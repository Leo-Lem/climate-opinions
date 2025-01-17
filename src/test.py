import pandas as pd
from datasets import Dataset

df = pd.read_csv("social-media-sentiment-analysis/res/twitter_sentiment_data.csv")
df = df[df['sentiment'] != 2]
df = df.rename(columns={'sentiment': 'label', 'message': 'text'})
df['text'] = df['text'].str.lower()

dataset = Dataset.from_pandas(df)

def transform_labels(example):
    # Umwandlung der Labels: -1 -> 0, 0 -> 1, 1 -> 2
    example['label'] = example['label'] + 1
    return example

transformed_dataset = dataset.map(transform_labels)

# Verwenden Sie train_test_split von scikit-learn mit festem Seed
train_val_split = transformed_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

# Split validation set into validation and test sets
val_test_split = val_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = val_test_split['train']
test_dataset = val_test_split['test']

# Write the test dataset to a CSV file
test_df = test_dataset.to_pandas()
test_df.to_csv("social-media-sentiment-analysis/res/test_dataset.csv", index=False)