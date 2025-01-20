from googleapiclient.discovery import build
from datetime import datetime, timedelta
import csv

# Get your own API Key :D (google cloud, https://developers.google.com/youtube/v3?hl=de)
api_key = ""
youtube = build('youtube', 'v3', developerKey=api_key)


def generate_quarters(start_year, start_month, end_year, end_month):
    current_year = start_year
    current_month = start_month

    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        since_date = datetime(current_year, current_month, 1)
        until_date = (since_date + timedelta(days=90)).replace(day=1)

        if until_date > datetime(end_year, end_month, 1):
            until_date = datetime(end_year, end_month, 1) + timedelta(days=31)

        yield since_date, until_date

        current_month += 3
        if current_month > 12:
            current_month -= 12
            current_year += 1


def crawl_youtube():
    # Query and CSV Parameters
    query = 'Global Warming|Climate Crisis|Climate Emergency|Global Heating|Climate Change|globalwarming|climatecrisis|climateemergency|globalheating|climatechange'
    output_file = 'youtube_comments.csv'

    # Write CSV header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["video-id", "comment-id", "username", "date", "text", "like-count"])

    # Loop through quarters (i.e. 2020 Q1, Q2, Q3, Q4)
    # dates can and should be modified here.
    for since_date, until_date in generate_quarters(2021, 1, 2024, 12):
        since_str = since_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        until_str = until_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        # API call
        response = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=100,
            publishedAfter=since_str,
            publishedBefore=until_str
        ).execute()

        comments_list = []

        # Retrieve Videos
        for item in response.get('items', []):

            video_id = item['id']['videoId']

            # Catch videos where comments are disabled
            try:
                comments_response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,  # Max allowed per request
                    order='relevance'  # Fetch top comments directly
                ).execute()

            except Exception as e:
                if "commentsDisabled" in str(e):
                    print(f"Comments are disabled for video ID: {
                          video_id}. Skipping...")
                    continue
                else:
                    raise e

            # Get details of comments
            for comment in comments_response.get('items', []):
                comment_snippet = comment['snippet']['topLevelComment']['snippet']
                comment_id = comment['id']
                date = comment_snippet['publishedAt'].split(
                    'T')[0]  # publishing date
                text = comment_snippet['textDisplay'].replace("\n", " ").replace(
                    "<b>", "").replace("<br>", "")  # Clean text
                username = comment_snippet['authorDisplayName']
                # Extract like count (more likes, stronger sentiment)
                like_count = comment_snippet['likeCount']

                # Append comment data to list
                comments_list.append({
                    'comment_id': comment_id,
                    'video_id': video_id,
                    'date': date,
                    'text': text,
                    'username': username,
                    'like_count': like_count
                })

        # Get top 500 comments from the previously prepared list, based on like count
        top_comments = sorted(
            comments_list, key=lambda x: x['like_count'], reverse=True)[:500]

        # Append to CSV file
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            for comment in top_comments:
                csvwriter.writerow([
                    comment['video_id'],
                    comment['comment_id'],
                    comment['username'],
                    comment['date'],
                    comment['text'],
                    comment['like_count']
                ])

        # Logging
        print(
            f"---> Processed data for {since_date.strftime('%B %Y')} - {until_date.strftime('%B %Y')}.")
