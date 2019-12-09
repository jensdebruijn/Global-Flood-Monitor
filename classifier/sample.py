from db.elastic import Elastic
import pandas as pd
import os
from random import shuffle, seed, choice
import elasticsearch
from datetime import datetime, timedelta, date
from db.postgresql import PostgreSQL
from psycopg2.extensions import AsIs
from methods.dates import daterange
from classes import Location
import re
import requests
import time
import urllib3


pg = PostgreSQL('gfm')
es = Elastic()
index = 'floods_detection'

START_DATE = datetime(2014, 7, 30)
MAX_DATE = datetime.utcnow()


def create_tables(refresh):
    if refresh:
        pg.cur.execute("""
            DROP TABLE IF EXISTS classification
        """)
        
    if not pg.table_exists('classification'):
        pg.cur.execute("""
            CREATE TABLE classification (
                idx SERIAL,
                id VARCHAR PRIMARY KEY,
                date TIMESTAMP,
                txt VARCHAR,
                language_code VARCHAR,
                label VARCHAR,
                who VARCHAR
            )
        """)


def tweet_is_available(ID):
    while True:
        try:
            r = requests.get(
                f"https://publish.twitter.com/oembed?url=https://twitter.com/any/status/{ID[2:]}")
            break
        except requests.exceptions.ConnectionError:
            print('just going to sleep for 30 seconds')
            time.sleep(30)
    if r.status_code != 200:
        return False
    json = r.json()
    if 'error' in json:
        return False
    else:
        return True


def find_urls(text):
    return re.findall(r"(\bhttps?://t.co/[a-zA-Z0-9]*\b)", text)

def url_is_alive(url):
    try:
        res = requests.get(url, timeout=1)
    except (
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.TooManyRedirects,
        requests.exceptions.ContentDecodingError,
        requests.exceptions.InvalidURL,
        UnicodeDecodeError,
        UnicodeError,
        urllib3.exceptions.LocationValueError,
        requests.exceptions.ChunkedEncodingError
    ):
        return False
    except AttributeError:
        print('WARNING: AttributeError at', url)
        return False
    if res.status_code == 200:
        return True
    else:
        return False

def urls_are_alive(text):
    urls = find_urls(text)
    for url in urls:
        if not url_is_alive(url):
            return False
        return True

def get_tweet(start_date, end_date, adm, level, language):
    query = {
        "size": 100,
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            f"locations.{level}_region": adm
                        },
                    }, {
                        "term": {
                            "source.retweet": False
                        }
                    }, {
                        'term': {
                            "source.lang": language
                        }
                    }, {
                        'range': {
                            "date": {
                                'gte': start_date,
                                'lt': end_date
                            }
                        }
                    }, {
                        "function_score": {
                            "functions": [
                                {
                                    "random_score": {
                                        "seed": int(time.time())
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }
    tweets = es.search(index=index, body=query)['hits']['hits']
    for tweet in tweets:
        tweet = tweet['_source']
        if not tweet_is_available(tweet['id']):
            print('not available')
            continue

        if not urls_are_alive(tweet['text']):
            print('url not alive')
            continue
        
        return tweet['id'], tweet['text'], tweet['date']
    return None




def sample_per_day_per_adm_languages(languages, max_count=1):

    start_query = datetime(2014, 7, 29)
    end_query = datetime(2018, 11, 20)
    days = list(daterange(start_query, end_query, timedelta(days=1), include_last=False))

    for language in languages:
        print(language)

        query = {
            'query': {
                'bool': {
                    'must': [
                        {
                            'term': {
                                'source.lang': language
                            }
                        }, {
                            'exists': {
                                'field': 'text'
                            }
                        }, {
                            'term': {
                                'source.retweet': False
                            }
                        }
                    ]
                }
            }
        }
        print(es.n_hits(index=index, body=query))

        while True:
            pg.cur.execute("""
                SELECT COUNT(*) FROM classification WHERE language_code = %s
            """, (language, ))
            count = pg.cur.fetchone()[0]
            print(count, '\r')
            if count >= max_count:
                break
            # pick random day
            day = choice(days)
            all_adm = []
            print(day)
            for level in ('level_0', 'level_1'):
                query = {
                    "size": 0,
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "source.lang": language
                                    }
                                },
                                {
                                    "range": {
                                        "date": {
                                            "gte": day.isoformat(), 
                                            "lt": (day + timedelta(days=1)).isoformat()   
                                        }
                                    }
                                },
                                {
                                    "term": {
                                        "source.retweet": False
                                    }
                                }
                            ]
                        }
                    },
                    "aggs" : {
                        'adm' : {
                            "terms": {
                                "field": f"locations.{level}_region",
                                "size": 500_000
                            }
                        }
                    }
                }

                res = es.search(index=index, body=query)['aggregations']['adm']
                assert res['doc_count_error_upper_bound'] == 0
                all_adm.extend([(bucket['key'], level) for bucket in res['buckets']])

            print(all_adm)

            # check if list not empty
            if all_adm:
                adm, level = choice(all_adm)

                tweet = get_tweet(day, day + timedelta(days=1), adm, level, language)
                print(day, all_adm)
                if tweet:
                    tweet_id, text, date = tweet
                
                    pg.cur.execute("""
                        INSERT INTO classification (id, txt, date, language_code)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (tweet_id, text, date, language))

            pg.conn.commit()
        pg.conn.commit()


if __name__ == '__main__':
    create_tables(refresh=False)
    df = pd.read_excel('input/tables/twitter_supported_languages.xlsx')
    languages = df[df['implemented'] == True]['language_code'].tolist()
    languages = ['en', 'id', 'es']
    sample_per_day_per_adm_languages(languages=languages, max_count=500)
