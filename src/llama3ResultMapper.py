import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# Load the CSV file
file_path = 'social-media-sentiment-analysis/res/test_dataset_predictionsLlama3_2.csv'
df = pd.read_csv(file_path)

# Calculate F1 score, precision, and recall
f1 = f1_score(df['label'], df['generated'], average='weighted')
precision = precision_score(df['label'], df['generated'], average='weighted')
recall = recall_score(df['label'], df['generated'], average='weighted')

print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')