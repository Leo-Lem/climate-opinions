from os import path
from pandas import read_csv
import matplotlib.pyplot as plt

from __params__ import RESULTS_PATH


def visualize(file: str):
    df = read_csv(path.join(RESULTS_PATH, f"{file}.csv"))

    # Group the data by Jahr and Monat and calculate the percentage of predictions equal to 0
    grouped = df.groupby(['Jahr', 'Monat']).apply(lambda x: (
        x['prediction'] == 0).mean() * 100).reset_index(name='percentage_zeros')

    # Sort the data by Jahr and Monat
    sorted = grouped.sort_values(by=['Jahr', 'Monat'])
    sorted['yearMonth'] = \
        sorted['Monat'].astype(str) + " " + grouped['Jahr'].astype(str)

    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.plot(sorted['yearMonth'], sorted['percentage_zeros'], marker='o')

    # Rotate x-axis labels by 90 degrees and show every second label
    plt.xticks(rotation=90)
    plt.xlabel('Month')
    plt.ylabel('Percentage of Predictions Equal to 0')
    plt.title('Monthly Percentage of Predictions Equal to 0')
    plt.grid(True)

    # Set the major locator to display every second label
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='lower'))
    plt.gca().set_xticks(sorted.index[::4])
    plt.gca().set_xticklabels(sorted['yearMonth'][::4])

    PLOT_PATH = path.join(RESULTS_PATH, f"{file}-nonbelievers_per_month.png")
    plt.savefig(PLOT_PATH)
    print(f"Saved plot to {PLOT_PATH}.")
