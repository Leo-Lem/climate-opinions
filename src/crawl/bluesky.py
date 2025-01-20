import csv
from datetime import datetime, timedelta
from atproto import Client, client_utils

BLUESKY_USERNAME = "climatesentiment.bsky.social"
BLUESKY_PASSWORD = "Sentiment2025"

client = Client()
client.login(BLUESKY_USERNAME, BLUESKY_PASSWORD)


def generate_months(start_year, start_month, end_year, end_month):
    current_year = start_year
    current_month = start_month

    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        since_date = datetime(current_year, current_month, 1)
        until_date = (since_date + timedelta(days=31)).replace(day=1)

        yield since_date, until_date

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1


def crawl_bluesky():
    """ In the following code, we iterate through 2010-2024 to extract 100 posts (API limit) per month and write the top posts into the .csv file that will serve as input for our sentiment model. """
    query = 'Global Warming|Climate Crisis|Climate Emergency|Global Heating|Climate Change|globalwarming|climatecrisis|climateemergency|globalheating|climatechange'
    limit = 100
    output_file = 'bluesky_posts.csv'

    # CSV Header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["tweet-id", "username", "date", "text", "like-count"])

    # API daily limit was exceeded after collecting ~ 3 years of data, so we ran the following over a period of 4-5 days and combined the files manually.
    # dates can and should be modified here.
    for since_date, until_date in generate_months(2018, 4, 2024, 12):

        # Formatting dates
        since_str = since_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        until_str = until_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        # API call
        params = {'q': query, 'limit': limit, 'since': since_str,
                  'until': until_str, 'sort': 'top'}
        response = client.app.bsky.feed.search_posts(params)

        # Extract & Clean Data
        posts_data = []
        if hasattr(response, 'posts'):
            for post in response.posts:
                tweet_id = post.uri
                date = post.record.created_at.split('T')[0]
                text = post.record.text.replace("\n", " ")
                username = post.author.handle.split('@')[0]

                # Likes Count (individual API call... you can't get just the count)
                likes = client.app.bsky.feed.get_likes({'uri': tweet_id})

                # Prep print
                posts_data.append({
                    'tweet_id': tweet_id,
                    'username': username,
                    'date': date,
                    'text': text,
                    'like_count': len(likes.likes)
                })

        # console sample print
        for post in posts_data[:5]:
            print(f"{post['tweet_id']},{post['username']},{
                  post['date']},\"{post['text']}\",{post['like_count']}")

        # CSV output
        with open('bluesky_posts.csv', 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            for post in posts_data:
                csvwriter.writerow(
                    [post['tweet_id'], post['username'], post['date'], post['text'], post['like_count']])

        # Logging
        print(f"Processed data for {since_date.strftime('%B %Y')}.")
