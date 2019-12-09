from TwitterAPI import TwitterAPI, TwitterError
import requests
import pandas as pd
import simplejson
import time

from config import (
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET
)

twitter_api = TwitterAPI(
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET
)


def chunker(iterable, size):
    for pos in range(0, len(iterable), size):
        yield iterable[pos:pos + size]

def lookup(tweet_ids):
    texts, dates, languages = [], [], []
    for tweet_ids_chunk in chunker(tweet_ids, 100):
        sleep = 1
        while True:
            try:
                ids = ",".join([str(ID) for ID in tweet_ids_chunk])
                r = twitter_api.request('statuses/lookup', {
                    'id': ids,
                    'tweet_mode': 'extended'
                })
                tweets = r.json()
                break
            except (
                TwitterError.TwitterConnectionError,
                requests.exceptions.ConnectionError
            ):
                sleep *= 2
            except simplejson.errors.JSONDecodeError:
                sleep *= 2
            sleep = min(sleep, 5 * 60)
            print(f'Connection error: sleeping for {sleep} seconds')
            time.sleep(sleep)
        tweets = {
            int(tweet['id_str']): tweet for tweet in tweets
        }
        for ID in tweet_ids_chunk:
            if ID in tweets:
                date = tweets[ID]['created_at']
                text = tweets[ID]['full_text']
                language = tweets[ID]['lang']
                if language == 'in':
                    language = 'id'
            else:
                date = 'NaN'
                text = 'NaN'
                language = 'NaN'
            texts.append(text)
            dates.append(date)
            languages.append(language)
    return texts, dates, languages


if __name__ == '__main__':
    df = pd.read_excel('input/labeled_tweets.xlsx')
    ids = [int(ID[2:]) for ID in df['id']]
    texts, dates, languages = lookup(ids)
    df['text'] = texts
    df['date'] = dates
    df['language'] = languages
    df.to_excel('input/labeled_tweets_hydrated.xlsx', index=False)