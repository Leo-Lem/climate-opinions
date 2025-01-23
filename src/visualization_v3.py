import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ------------- Path Variables ---------------
main_dir = os.path.dirname(os.path.realpath(__file__))  # Get the current directory
parent_dir = os.path.dirname(main_dir)  # Get the parent directory
data_dir = os.path.join(parent_dir, "data")  # Directory to store data

# Ensure the directories exist
os.makedirs(data_dir, exist_ok=True)

# ------------- File Name ---------------
predictions_filename = os.path.join(data_dir, 'bertweet_gefilterte_daten_twitter_prediction.csv')

# Load the dataset
df = pd.read_csv(predictions_filename)

# Extract year and quarter as integers
df['Quartal'] = (df['Monat'] - 1) // 3 + 1  # Calculate quarter

# Define relevance metric
df['Relevanz'] = df['Likes'] + 1.5 * df['Replies'] + 2 * df['Retweets']

# Logarithmic scaling of relevance to reduce the impact of outliers
df['Log_Relevanz'] = np.log1p(df['Relevanz'])

# Group by year and quarter for the three types of percentages
grouped_naive = df.groupby(['Jahr', 'Quartal']).apply(
    lambda x: (x['prediction'] == 0).mean() * 100
).reset_index(name='percentage_zeros')
grouped_naive['yearQuarter'] = 'Q' + grouped_naive['Quartal'].astype(str) + ' ' + grouped_naive['Jahr'].astype(str)

grouped_weighted = df.groupby(['Jahr', 'Quartal']).apply(
    lambda x: (x.loc[x['prediction'] == 0, 'Relevanz'].sum() / x['Relevanz'].sum()) * 100
    if x['Relevanz'].sum() > 0 else 0
).reset_index(name='gewichteter_anteil')
grouped_weighted['yearQuarter'] = 'Q' + grouped_weighted['Quartal'].astype(str) + ' ' + grouped_weighted['Jahr'].astype(str)

grouped_log_weighted = df.groupby(['Jahr', 'Quartal']).apply(
    lambda x: (x.loc[x['prediction'] == 0, 'Log_Relevanz'].sum() / x['Log_Relevanz'].sum()) * 100
    if x['Log_Relevanz'].sum() > 0 else 0
).reset_index(name='log_gewichteter_anteil')
grouped_log_weighted['yearQuarter'] = 'Q' + grouped_log_weighted['Quartal'].astype(str) + ' ' + grouped_log_weighted['Jahr'].astype(str)

# Plot all three graphs in one figure
plt.figure(figsize=(14, 8))

# Plot naive percentages
plt.plot(grouped_naive['yearQuarter'], grouped_naive['percentage_zeros'], marker='o', label='Naive Percentage')

# Plot weighted percentages
plt.plot(grouped_weighted['yearQuarter'], grouped_weighted['gewichteter_anteil'], marker='o', label='Weighted Percentage')

# Plot log-weighted percentages
plt.plot(grouped_log_weighted['yearQuarter'], grouped_log_weighted['log_gewichteter_anteil'], marker='o', label='Log-Weighted Percentage')

# Customize plot
plt.xticks(rotation=90)
plt.xlabel('Time Period')
plt.ylabel('Percentage')
plt.title('Comparison of Naive, Weighted, and Log-Weighted Percentages of Climate Change Denial Tweets')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()