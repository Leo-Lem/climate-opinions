import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('youtube_prediction.csv')



try:
    # Extract the year as an integer from the date string
    df['Jahr'] = df['date'].str[0:4].astype(int)  # Extracts 'YYYY' and converts it to int
    
    # Extract the month as an integer from the date string
    df['Monat'] = df['date'].str[5:7].astype(int)  # Extracts 'MM' and converts it to int

    # Display the DataFrame with the new columns
    print(df)
except ValueError as e:
    print(f"Error converting to int: {e}")

# Group the data by Jahr and Monat and calculate the percentage of predictions equal to 0
grouped = df.groupby(['Jahr', 'Monat']).apply(lambda x: (x['prediction'] == 0).mean() * 100).reset_index(name='percentage_zeros')

# Sort the data by Jahr and Monat
grouped = grouped.sort_values(by=['Jahr', 'Monat'])
print(grouped)
grouped['yearMonth'] = grouped['Monat'].astype(str) + ' ' + grouped['Jahr'].astype(str)

# Plot the values
plt.figure(figsize=(10, 6))
plt.plot(grouped['yearMonth'], grouped['percentage_zeros'], marker='o')

# Rotate x-axis labels by 90 degrees and show every second label
plt.xticks(rotation=90)
plt.xlabel('Month')
plt.ylabel('Percentage of comments against climate change')
plt.title('Predicted Monthly Percentage of Youtube Comments that do not believe in man made climate change')
plt.grid(True)

# Set the major locator to display every second label
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='lower'))
plt.gca().set_xticks(grouped.index[::4])
plt.gca().set_xticklabels(grouped['yearMonth'][::4])

plt.show()
