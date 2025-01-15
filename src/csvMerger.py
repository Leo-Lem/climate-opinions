import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from openpyxl.utils.exceptions import IllegalCharacterError

# Base directory paths
main_dir = os.path.dirname(os.path.realpath(__file__))  # Current directory
parent_dir = os.path.dirname(main_dir)  # Parent directory
data_dir = os.path.join(parent_dir, "data")  # Directory for data
csv_dir = os.path.join(data_dir, "csv_files_twitter")  # Directory for CSV files
log_dir = os.path.join(data_dir, "log_twitter")  # Directory for log files

# Ensure directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# File paths
BASE_DIR = csv_dir  # Base directory for CSV files
OUTPUT_FILE = os.path.join(data_dir, 'gesammelte_daten_twitter.csv')  # Output file for collected data
FILTERED_FILE = os.path.join(data_dir, 'gefilterte_daten_twitter.csv')  # Output file for filtered data
EXCLUDED_FILE = os.path.join(data_dir, 'ausgeschlossene_daten_twitter.csv')  # Output file for excluded data
ERROR_LOG = os.path.join(data_dir, 'fehlerhafte_dateien_twitter.txt')  # Error log file

# Search strings for text and hashtags
SUCHSTRINGS = ["global warming", "climate crisis", "climate emergency", "global heating", "climate change"]
HASHTAG_STRINGS = ["globalwarming", "climatecrisis", "climateemergency", "globalheating", "climatechange"]

def clean_value(value):
    """Removes invalid characters from a cell."""
    if isinstance(value, str):
        return ''.join(c for c in value if ord(c) >= 32)
    return value

def save_to_excel(df, dateiname):
    """Saves a DataFrame as a well-formatted Excel file."""
    pfad = os.path.join(data_dir, dateiname)  # File path for the Excel file
    wb = Workbook()  # Create a new workbook
    ws = wb.active  # Get the active worksheet
    ws.title = "Daten"  # Set worksheet title

    # Write DataFrame content into the Excel sheet
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=1):
        for c_idx, value in enumerate(row, start=1):
            try:
                cell = ws.cell(row=r_idx, column=c_idx, value=clean_value(value))  # Add data to the cell
                # Format header row
                if r_idx == 1:
                    cell.font = Font(bold=True)  # Bold font for headers
                    cell.alignment = Alignment(horizontal="center", vertical="center")  # Center alignment for headers
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")  # Left alignment for data
            except IllegalCharacterError as e:
                print(f"Invalid character in cell {r_idx}, {c_idx}: {e}")

    # Auto-adjust column widths
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = max_length + 2  # Adjust column width

    wb.save(pfad)  # Save the workbook
    print(f"Excel file saved: {pfad}")

def collect_csv_files(dateipraefix):
    """Collects data from all CSV files matching a specific prefix."""
    gesammelte_daten = []  # List for collected data
    gefilterte_daten = []  # List for filtered data
    ausgeschlossene_daten = []  # List for excluded data

    for file in os.listdir(BASE_DIR):
        if file.startswith(dateipraefix) and file.endswith(".csv"):
            try:
                # Extract year and month from file name
                teile = file.replace(".csv", "").split("_")
                jahr, monat = int(teile[-2]), int(teile[-1])

                # Load the CSV file
                pfad = os.path.join(BASE_DIR, file)
                daten = pd.read_csv(pfad)

                # Skip files with no columns
                if daten.empty or daten.columns.size == 0:
                    print(f"File {file} has no valid columns and will be skipped.")
                    continue

                # Add year and month columns
                daten["Jahr"] = jahr
                daten["Monat"] = monat

                # Separate filtered data
                gefilterte = daten[
                    daten["Text"].astype(str).str.contains('|'.join(SUCHSTRINGS), na=False, case=False) |
                    daten["Hashtags"].astype(str).str.contains('|'.join(HASHTAG_STRINGS), na=False, case=False)
                ]
                nicht_gefilterte = daten[~daten.index.isin(gefilterte.index)]  # Data not matching filters

                # Append data to corresponding lists
                gefilterte_daten.append(gefilterte)
                ausgeschlossene_daten.append(nicht_gefilterte)
                gesammelte_daten.append(daten)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Combine and save data
    if gesammelte_daten:
        gesamt_df = pd.concat(gesammelte_daten, ignore_index=True)  # Combine collected data
        gesamt_df.to_csv(OUTPUT_FILE, index=False)  # Save to CSV
        save_to_excel(gesamt_df, 'gesammelte_daten_twitter.xlsx')  # Save to Excel
        print(f"Collected data saved in {OUTPUT_FILE}")

    if gefilterte_daten:
        gefiltert_df = pd.concat(gefilterte_daten, ignore_index=True)  # Combine filtered data
        gefiltert_df.to_csv(FILTERED_FILE, index=False)  # Save to CSV
        save_to_excel(gefiltert_df, 'gefilterte_daten_twitter.xlsx')  # Save to Excel
        print(f"Filtered data saved in {FILTERED_FILE}")

    if ausgeschlossene_daten:
        ausgeschlossen_df = pd.concat(ausgeschlossene_daten, ignore_index=True)  # Combine excluded data
        ausgeschlossen_df.to_csv(EXCLUDED_FILE, index=False)  # Save to CSV
        save_to_excel(ausgeschlossen_df, 'ausgeschlossene_daten_twitter.xlsx')  # Save to Excel
        print(f"Excluded data saved in {EXCLUDED_FILE}")

def validate_csv_files(dateipraefix):
    """Checks if CSV files are empty or contain errors, and logs such files."""
    fehlerhafte_dateien = []  # List for faulty files

    for file in os.listdir(BASE_DIR):
        if file.startswith(dateipraefix) and file.endswith(".csv"):
            try:
                # Load the CSV file
                pfad = os.path.join(BASE_DIR, file)
                daten = pd.read_csv(pfad)

                # Check if the file is empty
                if daten.empty:
                    fehlerhafte_dateien.append(file)
            except Exception as e:
                print(f"Error checking file {file}: {e}")
                fehlerhafte_dateien.append(file)

    # Save faulty files to a log file
    with open(ERROR_LOG, "w") as log:
        for datei in fehlerhafte_dateien:
            log.write(f"{datei}\n")

    print(f"Faulty files logged in {ERROR_LOG}")

if __name__ == "__main__":
    modus = input("Select mode (collect/validate): ").strip().lower()  # Prompt for mode selection
    dateipraefix = "Twitter"  # File prefix for processing

    if modus == "collect":
        collect_csv_files(dateipraefix)  # Call data collection function
    elif modus == "validate":
        validate_csv_files(dateipraefix)  # Call data validation function
    else:
        print("Invalid mode. Please choose 'collect' or 'validate'.")

