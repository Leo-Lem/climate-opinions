import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
#df = pd.read_csv('bluesky_prediction.csv')
df = pd.read_csv('social-media-sentiment-analysis/res/bertweet_gefilterte_daten_twitter_prediction.csv')
df2 = pd.read_csv('social-media-sentiment-analysis/res/The Climate Change Twitter Dataset.csv')

try:
    # Extract the year as an integer from the date string
    #df['Jahr'] = df['date'].str[0:4].astype(int)  # Extracts 'YYYY' and converts it to int
    df2['Jahr'] = df2['created_at'].str[0:4].astype(int)  # Extracts 'YYYY' and converts it to int
    
    # Extract the month as an integer from the date string
    #df['Monat'] = df['date'].str[5:7].astype(int)  # Extracts 'MM' and converts it to int
    df2['Monat'] = df2['created_at'].str[5:7].astype(int)  # Extracts 'MM' and converts it to int

    # Create a quarter column
    df['Quartal'] = (df['Monat'] - 1) // 3 + 1
    df2['Quartal'] = (df2['Monat'] - 1) // 3 + 1

    df2 = df2[df2['Jahr'] > 2006 or df2['Jahr'] == 2006 and df2['Quartal'] > 3]

    # Display the DataFrame with the new columns
    print(df)
except ValueError as e:
    print(f"Error converting to int: {e}")

# Boolean to control the aggregation: True for monthly, False for quarterly
aggregate_monthly = False  # Change this to True for monthly calculations

if aggregate_monthly:
    # Group the data by Jahr and Monat and calculate the percentage of predictions equal to 0
    grouped = df.groupby(['Jahr', 'Monat']).apply(lambda x: (x['prediction'] == 0).mean() * 100).reset_index(name='percentage_zeros')

    # Sort the data by Jahr and Monat
    grouped = grouped.sort_values(by=['Jahr', 'Monat'])
    grouped['yearMonth'] = grouped['Monat'].astype(str) + ' ' + grouped['Jahr'].astype(str)
else:
    # Group the data by Jahr and Quartal and calculate the average percentage of predictions equal to 0
    grouped = df.groupby(['Jahr', 'Quartal']).apply(lambda x: (x['prediction'] == 0).mean() * 100).reset_index(name='percentage_zeros')
    grouped2 = df2.groupby(['Jahr', 'Quartal']).apply(lambda x: (x['stance'] == 'denier').mean() * 100).reset_index(name='percentage_zeros')

    # Sort the data by Jahr and Quartal
    grouped = grouped.sort_values(by=['Jahr', 'Quartal'])
    grouped['yearQuarter'] = 'Q' + grouped['Quartal'].astype(str) + ' ' + grouped['Jahr'].astype(str)

    grouped2 = grouped2.sort_values(by=['Jahr', 'Quartal'])
    grouped2['yearQuarter'] = 'Q' + grouped2['Quartal'].astype(str) + ' ' + grouped2['Jahr'].astype(str)

# Plot the values
plt.figure(figsize=(10, 6))

if aggregate_monthly:
    plt.plot(grouped['yearMonth'], grouped['percentage_zeros'], marker='o')
    plt.gca().set_xticks(grouped.index[::4])
    plt.gca().set_xticklabels(grouped['yearMonth'][::4])    
else:
    plt.plot(grouped['yearQuarter'], grouped['percentage_zeros'], marker='o')
    plt.plot(grouped2[grouped2['percentage_zeros']>0]['yearQuarter'], grouped2[grouped2['percentage_zeros']>0]['percentage_zeros'], marker='o')
    
    grouped['yearQuarter'] = 'Q' + grouped['Quartal'].astype(str) + ' ' + grouped['Jahr'].astype(str)


    plt.gca().set_xticks(grouped.index)  # For quarterly, display every label
    plt.gca().set_xticklabels(grouped['yearQuarter'])

# Rotate x-axis labels by 90 degrees
plt.xticks(rotation=90)
plt.xlabel('Time Period')
plt.ylabel('Percentage of Twitter posts against climate change')
plt.title('Predicted Percentage of posts that do not believe in man made climate change\n on our crawled Twitter posts vs. Stance of the Climate Change Twitter Dataset')
plt.grid(True)

plt.show()