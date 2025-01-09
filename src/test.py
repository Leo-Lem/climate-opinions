import pandas as pd

def decode_string(input_string):
    # Zuerst dekodieren wir den String von ISO-8859-1.
    first_decoded = input_string.encode('utf-8').decode('iso-8859-1')
    return first_decoded

df = pd.read_csv("social-media-sentiment-analysis/res/twitter_sentiment_data.csv")

# Beispiel-Input
input_string = df.loc[7, "message"]
print(input_string)
input_string = "Ã¢â‚¬â„¢"
# Aufruf der Funktion
output_string = decode_string(input_string)
print("Bereinigter String:", output_string)