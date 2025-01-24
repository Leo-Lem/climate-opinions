import pandas as pd
import matplotlib.pyplot as plt

# CSV-Dateien einlesen
csv_files = ['twitterDenialPerQuartal.csv', 'blueskyDenialPerQuartal.csv', 'youtubeDenialPerQuartal.csv']
dfs = [pd.read_csv(file) for file in csv_files]

# Wandeln Sie den 'yearQuarter'-String in ein Datum um, das sinnvoll für die X-Achse ist

# Daten zusammenführen und ein einheitliches Zeitintervall bestimmen
all_data = pd.concat(dfs)

# Wandeln Sie den 'yearQuarter'-String in ein Datum um
def convert_to_datetime(year_quarter):
    quarter, year = year_quarter.split()
    month_map = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
    return pd.to_datetime(f"{year}-{month_map[quarter]}")

all_data['date'] = all_data['yearQuarter'].apply(convert_to_datetime)

# Beschrifte die X-Achse mit allen möglichen Quartalen
combined_quarters = all_data['yearQuarter'].unique()
combined_quarters.sort()

plt.figure(figsize=(12, 6))

for idx, df in enumerate(dfs):
    df['date'] = df['yearQuarter'].apply(convert_to_datetime)
    plt.plot(df['date'], df['percentage_zeros'], marker='o', label='Twitter'if idx == 0 else 'Bluesky' if idx == 1 else 'YouTube')

# Setzen der X-Achsbeschriftungen
plt.xticks(ticks=[convert_to_datetime(q) for q in combined_quarters], labels=combined_quarters, rotation=90)

plt.xlabel('Time span')
plt.ylabel('Percentage of denying comments')
plt.title('Percentage of denying comments per quarter for different platforms')
plt.legend()

plt.tight_layout()

# Diagramm anzeigen
plt.show()