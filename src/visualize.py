from os import path
from pandas import read_csv
import matplotlib.pyplot as plt

from __params__ import RESULTS_PATH, SAMPLE


def visualize(file: str):
    """ Visualize the percentage of predictions equal to 0 per month. """
    df = read_csv(path.join(RESULTS_PATH, f"{SAMPLE}{file}.csv"))

    if "twitter" in file:
        nonbelievers_per_month(df)

    plot_path = path.join(RESULTS_PATH,
                          f"{SAMPLE}{file}-nonbelievers_per_month.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}.")


def nonbelievers_per_month(df):
    # TODO: use subplots when doing more visualizations
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
