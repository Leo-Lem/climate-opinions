# ------------------------------------------------------------------------------------
# ------------------------------- Version 3 ------------------------------------------
# - Overview:
#   - This script uses Twitter's search feature to find the top posts for a given search term over a given period (searched monthly)
#   - Functions:
#        -> Test/Production mode controlled via variables
#           -> Test mode works correctly: CSV works, search query works, logging works
#           -> Test mode with real API works: Tweets are correctly loaded
#           -> Production mode works correctly:
#        -> Cookies are correctly loaded (or created if not present).
#        -> Logging supports both file and console output, controlled via variables.
#   - Problems:
#        -> Each query only contains up to ca. 250 Posts. -> no direct solution found
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ------------- Imports ---------------
# Importing Twitter client and error handling for rate limits
from twikit import Client, TooManyRequests
import asyncio  # For handling asynchronous tasks
# For getting the current date and time
from datetime import datetime, timedelta
import pandas as pd  # For working with DataFrames and saving data to CSV
import os  # For file and directory operations
import logging  # For logging application events
import csv  # For CSV file operations
# For generating random numbers (used in mock tweets)
from random import randint
import random
from configparser import ConfigParser  # For reading configuration files
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment
from openpyxl.utils.exceptions import IllegalCharacterError

from __params__ import OUT_PATH, DATA_PATH, QUERY

# ------------- Path Variables ---------------
data_dir = OUT_PATH
csv_dir = os.path.join(data_dir, "csv_files_twitter")
log_dir = os.path.join(data_dir, "log_twitter")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ------------- File Names ---------------
config_filename = os.path.join(DATA_PATH, 'configTwitter.ini')
cookies_filename = os.path.join(data_dir, 'cookiesTwitter.json')
log_filename = os.path.join(log_dir,
                            f'crawlTwitter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
BASE_DIR = csv_dir
OUTPUT_FILE = os.path.join(data_dir, 'gesammelte_daten_twitter.csv')
FILTERED_FILE = os.path.join(data_dir, 'gefilterte_daten_twitter.csv')
EXCLUDED_FILE = os.path.join(data_dir, 'ausgeschlossene_daten_twitter.csv')
ERROR_LOG = os.path.join(data_dir, 'fehlerhafte_dateien_twitter.txt')

# ------------- Global Variables ---------------
TEST_MODE = True  # Set to False for production mode, True for test mode
USE_REAL_API_IN_TEST = True  # True for real API in test mode, False for mock tweets

ENABLE_HTTP_LOGGERS = False  # Set this to True to enable HTTP-related logging
LOG_TO_CONSOLE = True  # Set to False to disable console logging

SEARCH_QUERY = ' OR '.join(QUERY)\
    + " OR "\
    + " OR ".join(["#" + q.replace(' ', '') for q in QUERY])

# Minimum number of tweets for production mode -> doesn't work correctly, Twitter seems to find only up to 250 entries per query.
MINIMUM_TWEETS = 500
TEST_TWEETS_PER_MONTHS = 10  # Number of tweets to fetch in test mode per quarter

# Test period (if in test mode)
TEST_YEAR_START = 2019  # Start year for test mode
TEST_YEAR_END = 2020    # End year for test mode
TEST_MONTH_START = 11    # Start month for test mode
TEST_MONTH_END = 3      # End month for test mode

# Real crawling period
REAL_YEAR_START = 2006  # Start year for real data crawling
REAL_YEAR_END = 2024    # End year for real data crawling
REAL_MONTH_START = 1    # Start month for real data crawling
REAL_MONTH_END = 12     # End month for real data crawling

MONTHS = {
    "1": {"start": "01-01", "end": "01-31"},
    "2": {"start": "02-01", "end": "02-28"},
    "3": {"start": "03-01", "end": "03-31"},
    "4": {"start": "04-01", "end": "04-30"},
    "5": {"start": "05-01", "end": "05-31"},
    "6": {"start": "06-01", "end": "06-30"},
    "7": {"start": "07-01", "end": "07-31"},
    "8": {"start": "08-01", "end": "08-31"},
    "9": {"start": "09-01", "end": "09-30"},
    "10": {"start": "10-01", "end": "10-31"},
    "11": {"start": "11-01", "end": "11-30"},
    "12": {"start": "12-01", "end": "12-31"}
}

SUCHSTRINGS = ["global warming", "climate crisis",
               "climate emergency", "global heating", "climate change"]
HASHTAG_STRINGS = ["globalwarming", "climatecrisis",
                   "climateemergency", "globalheating", "climatechange"]

# ------------- Mock-Tweet-Klassen ---------------


class MockUser:
    def __init__(self, name, followers_count, location):
        self.name = name
        self.followers_count = followers_count
        self.location = location


class MockTweet:
    def __init__(self, user, text, favorite_count, retweet_count, reply_count, hashtags, tweet_id, place_data=None):
        self.user = user
        self.text = text
        self.favorite_count = favorite_count
        self.retweet_count = retweet_count
        self.reply_count = reply_count
        self.hashtags = hashtags
        self.id = tweet_id
        self._place_data = place_data
        self.created_at = datetime.now() - timedelta(days=random.randint(0, 30))


# ------------------ Functions -------------------
# (1) --- Logging ---
def configure_loggers():
    """Configures file and console logging, and disables specific loggers if needed."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for logging to console
    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.info("Console logging is enabled.")
    else:
        logger.info("Console logging is disabled.")

    # Disable HTTP-related loggers if necessary
    if ENABLE_HTTP_LOGGERS:
        logging.getLogger("httpx").disabled = False
        logging.getLogger("urllib3").disabled = False
        logging.getLogger("twikit").disabled = False
        logging.info("HTTP-related loggers enabled.")
    else:
        logging.getLogger("httpx").disabled = True
        logging.getLogger("urllib3").disabled = True
        logging.getLogger("twikit").disabled = True
        logging.info("HTTP-related loggers disabled.")

# (2) --- Others ---


def get_mock_tweets():
    """Generates mock tweets for test mode."""
    return [
        MockTweet(
            user=MockUser(f'TestUser{randint(1, 100)}',
                          randint(100, 1000), 'Mock Location'),
            text=f'Mock tweet about climate change #{randint(1, 100)}',
            favorite_count=randint(10, 100),
            retweet_count=randint(0, 50),
            reply_count=randint(0, 20),
            hashtags=['#climatechange'],
            tweet_id=str(randint(1000000000, 9999999999)),
            place_data={'full_name': 'Mock City'} if random.choice(
                [True, False]) else None
        )
        for _ in range(TEST_TWEETS_PER_MONTHS)
    ]


def generate_query(year, month, search_query):
    """Generates a search query based on the year, month, and the search query."""
    # Get the start and end dates for the month
    month_data = MONTHS.get(month)
    if not month_data:
        raise ValueError(f"Invalid month: {month}")

    month_start = f"{year}-{month_data['start']}"
    month_end = f"{year}-{month_data['end']}"

    query = f"{search_query} since:{month_start} until:{month_end} lang:en"
    return query


def save_to_csv(tweet_data_list, csv_filename):
    """Saves the collected tweet data to a CSV file."""
    df = pd.DataFrame(tweet_data_list)
    df.to_csv(csv_filename, index=False,
              encoding='utf-8', quoting=csv.QUOTE_ALL)
    logging.info(f'{len(tweet_data_list)} tweets saved to {csv_filename}')


# (3) --- Crawling ---
async def initialize_cookies(client):
    """Checks if cookies exist and loads them, or logs in to create and save cookies."""
    if os.path.exists(cookies_filename):  # Check if cookies already exist
        logging.info(f'Loading cookies from {cookies_filename}')
        client.load_cookies(cookies_filename)
    else:
        logging.info('No cookies found. Logging in to create cookies...')
        try:
            # Load configuration data (username, email, password)
            config = ConfigParser()
            config.read(config_filename)
            username = config.get('X', 'username')
            email = config.get('X', 'email')
            password = config.get('X', 'password')
            logging.info('Configuration file read successfully.')

            # Login and save cookies
            await client.login(auth_info_1=username, auth_info_2=email, password=password)
            client.save_cookies(cookies_filename)
            logging.info(f'Cookies have been saved to {cookies_filename}')
        except Exception as e:
            # Log any error that occurs during login
            logging.error(f'Error: {str(e)}')
            raise


async def get_tweets(client, query, tweets=None):
    """Fetches tweets based on the given query."""
    if TEST_MODE and not USE_REAL_API_IN_TEST:
        logging.info(f'Test mode: Returning mock tweets for query: {query}')
        return get_mock_tweets()  # Return mock tweets in test mode

    if tweets is None:
        logging.info(f'Fetching tweets for query: {query}')
        # Get tweets from real API
        return await client.search_tweet(query, product='Top')
    else:
        await asyncio.sleep(15)  # Simulate waiting time between API requests
        return await tweets.next()


async def crawl_query(client, query, tweet_limit):
    """Crawls tweets based on the provided query and returns the collected data."""
    tweet_count = 0  # Initialize tweet counter
    tweet_data_list = []  # List to store tweet data

    tweets = None  # To handle pagination of tweets

    logging.info(f'Starting query crawl: {query}')

    while tweet_count < tweet_limit:  # Continue until tweet limit is reached
        try:
            tweets = await get_tweets(client, query, tweets)  # Fetch tweets
        except TooManyRequests as e:  # Handle rate limiting
            reset_time = datetime.fromtimestamp(e.rate_limit_reset)
            logging.warning(f'Rate limit reached. Waiting until {reset_time}')
            # Wait until rate limit resets
            await asyncio.sleep((reset_time - datetime.now()).total_seconds())
            continue

        if not tweets:  # If no tweets are returned, break the loop
            logging.info(f'No more tweets for query: {query}')
            break

        # Process and collect the tweet data
        for tweet in tweets:
            tweet_count += 1  # Increment tweet counter
            tweet_data = {
                'Tweet_count': tweet_count,
                'Username': tweet.user.name,
                'Followers Count': tweet.user.followers_count,
                'text': tweet.text.replace('\n', ' ').replace('\r', ''),
                'Created at': tweet.created_at,
                'Likes': tweet.favorite_count,
                'Retweets': tweet.retweet_count,
                'Replies': tweet.reply_count,
                'Hashtags': ', '.join(tweet.hashtags) if tweet.hashtags else None,
                'Tweet ID': tweet.id,
                'Place Name': tweet._place_data['full_name'] if tweet._place_data else None,
                'User Location': tweet.user.location
            }

            tweet_data_list.append(tweet_data)  # Add tweet data to the list

            if tweet_count >= tweet_limit:  # Stop if tweet limit is reached
                break

    logging.info(f'Finished crawling query. {tweet_count} tweets collected.')
    return tweet_data_list  # Return the collected tweet data


async def crawl_time_period(client, start_year, start_month, end_year, end_month, tweet_limit, search_query):
    """Crawl tweets over the specified time period from start_year/start_month to end_year/end_month."""
    current_year = start_year
    current_month = start_month

    while current_year < end_year or (current_year == end_year and current_month <= end_month):
        # Generate query for the current month
        query = generate_query(current_year, str(current_month), search_query)

        # Define the filename for the CSV
        csv_filename = os.path.join(
            csv_dir, f'Twitter_{current_year}_{current_month}.csv')

        logging.info(f'Starting crawling for month: {
                     current_year}-{current_month}')

        # Call the crawl_query function to get the tweet data
        try:
            tweet_data_list = await crawl_query(client, query, tweet_limit)
        except Exception as e:
            logging.error(f"Error while crawling tweets for query '{
                          query}': {str(e)}")
            tweet_data_list = []  # Optional: Rückgabe einer leeren Liste bei Fehler
        # Save the collected tweet data to CSV
        save_to_csv(tweet_data_list, csv_filename)

        # Move to the next month
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1


async def crawl_missing_months(client, missing_files):
    """ Crawls only the missing months from a given list of missing files. """
    for file in missing_files:
        try:
            # Extract year and month from the file name
            parts = file.replace(".csv", "").split("_")
            year, month = int(parts[1]), int(parts[2])

            # Generate the query
            query = generate_query(year, str(month), SEARCH_QUERY)

            # Target CSV file
            csv_filename = os.path.join(csv_dir, file)

            # Crawl tweets and save them
            logging.info(f"Crawling missing month: {year}-{month}")
            tweet_data_list = await crawl_query(client, query, MINIMUM_TWEETS)
            save_to_csv(tweet_data_list, csv_filename)
        except Exception as e:
            logging.error(f"Error while crawling {file}: {e}")


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
                # Add data to the cell
                cell = ws.cell(row=r_idx, column=c_idx,
                               value=clean_value(value))
                # Format header row
                if r_idx == 1:
                    cell.font = Font(bold=True)  # Bold font for headers
                    # Center alignment for headers
                    cell.alignment = Alignment(
                        horizontal="center", vertical="center")
                else:
                    # Left alignment for data
                    cell.alignment = Alignment(
                        horizontal="left", vertical="center")
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
        # Adjust column width
        ws.column_dimensions[col_letter].width = max_length + 2

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
                    print(
                        f"File {file} has no valid columns and will be skipped.")
                    continue

                # Add year and month columns
                daten["Jahr"] = jahr
                daten["Monat"] = monat

                # Separate filtered data
                gefilterte = daten[
                    daten["Text"].astype(str).str.contains('|'.join(SUCHSTRINGS), na=False, case=False) |
                    daten["Hashtags"].astype(str).str.contains(
                        '|'.join(HASHTAG_STRINGS), na=False, case=False)
                ]
                nicht_gefilterte = daten[~daten.index.isin(
                    gefilterte.index)]  # Data not matching filters

                # Append data to corresponding lists
                gefilterte_daten.append(gefilterte)
                ausgeschlossene_daten.append(nicht_gefilterte)
                gesammelte_daten.append(daten)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    # Combine and save data
    if gesammelte_daten:
        # Combine collected data
        gesamt_df = pd.concat(gesammelte_daten, ignore_index=True)
        gesamt_df.to_csv(OUTPUT_FILE, index=False)  # Save to CSV
        # Save to Excel
        save_to_excel(gesamt_df, 'gesammelte_daten_twitter.xlsx')
        print(f"Collected data saved in {OUTPUT_FILE}")

    if gefilterte_daten:
        # Combine filtered data
        gefiltert_df = pd.concat(gefilterte_daten, ignore_index=True)
        gefiltert_df.to_csv(FILTERED_FILE, index=False)  # Save to CSV
        # Save to Excel
        save_to_excel(gefiltert_df, 'gefilterte_daten_twitter.xlsx')
        print(f"Filtered data saved in {FILTERED_FILE}")

    if ausgeschlossene_daten:
        ausgeschlossen_df = pd.concat(
            ausgeschlossene_daten, ignore_index=True)  # Combine excluded data
        ausgeschlossen_df.to_csv(EXCLUDED_FILE, index=False)  # Save to CSV
        # Save to Excel
        save_to_excel(ausgeschlossen_df, 'ausgeschlossene_daten_twitter.xlsx')
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


# ------------------ Main Program -------------------
async def crawl_twitter():
    configure_loggers()
    if TEST_MODE:
        logging.info(f'Running in test mode with API {
                     "enabled" if USE_REAL_API_IN_TEST is True else "disabeld"}')
        start_year = TEST_YEAR_START
        start_month = TEST_MONTH_START
        end_year = TEST_YEAR_END
        end_month = TEST_MONTH_END
    else:
        logging.info('Running in production mode.')
        start_year = REAL_YEAR_START
        start_month = REAL_MONTH_START
        end_year = REAL_YEAR_END
        end_month = REAL_MONTH_END

    client = Client(language='en-US')  # Initialize Twitter-Client

    # Load cookies for production mode or test mode with real API
    if (not TEST_MODE or (TEST_MODE and USE_REAL_API_IN_TEST)):
        await initialize_cookies(client)
    else:
        logging.warning(
            'No cookies loaded because we are using mock data in test mode.')

    # Set Tweet-Limit in accordance with mode.
    tweet_limit = TEST_TWEETS_PER_MONTHS if TEST_MODE else MINIMUM_TWEETS
    search_query = SEARCH_QUERY

    # Crawl the data over the defined period
    await crawl_time_period(client, start_year, start_month, end_year, end_month, tweet_limit, search_query)

    # Optional: Crawl instead for a given list of months:
    # missing_months = ["Twitter_2006_1.csv",...]
    # await crawl_missing_months(client, missing_months)

    logging.info('All crawling processes finished.')

    modus = input("Select mode (collect/validate): ").strip().lower()
    dateipraefix = "Twitter"  # File prefix for processing

    if modus == "collect":
        collect_csv_files(dateipraefix)  # Call data collection function
    elif modus == "validate":
        validate_csv_files(dateipraefix)  # Call data validation function
    else:
        print("Invalid mode. Please choose 'collect' or 'validate'.")
