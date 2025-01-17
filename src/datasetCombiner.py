import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('predictions2.csv')

# Group the data by Jahr and Monat and calculate the percentage of predictions equal to 0
grouped = df.groupby(['Jahr', 'Monat']).apply(lambda x: (x['prediction'] == 0).mean() * 100).reset_index(name='percentage_zeros')

# Sort the data by Jahr and Monat
grouped = grouped.sort_values(by=['Jahr', 'Monat'])

# Plot the values
plt.figure(figsize=(10, 6))
plt.plot(grouped['Monat'], grouped['percentage_zeros'], marker='o')

plt.xlabel('Month')
plt.ylabel('Percentage of Predictions Equal to 0')
plt.title('Monthly Percentage of Predictions Equal to 0')
plt.grid(True)
plt.show()
