import json
import os
from collections import Counter
import statistics
import datetime

from db.elastic import Elastic
from db.postgresql import PostgreSQL

from config import LEVEL_2_COUNTRIES, DOCUMENT_INDEX

from methods.tweets import LastTweetsDeque

es = Elastic()
pg = PostgreSQL('gfm')


def get_time_query(body, max_tweets=1000):
    last_tweets = LastTweetsDeque()
    tweets = es.scroll_through(index=DOCUMENT_INDEX, body=body, source=False)
    n = 0
    for tweet in tweets:
        if not last_tweets.is_similar_to(text=tweet['text']):
            date = tweet['date'] + datetime.timedelta(minutes=30)
            yield date.hour
            n += 1
            if n == max_tweets:
                break


def get_hours_country_mentions(level_0_region):
    body = {
        "_source": ["date", "text"],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term":
                                    {"locations.level_0_region": level_0_region}
                            },
                            {
                                "term":
                                    {"locations.location_ID": level_0_region}
                            }
                        ]
                    }
                },
                "functions": [{
                    "random_score": {}
                }]
            }
        }
    }
    yield from get_time_query(body)


def get_hours_country(level_0_region):
    body = {
        "_source": ["date", "text"],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term":
                                    {"locations.level_0_region": level_0_region}
                            }
                        ]
                    },
                },
                "functions": [{
                    "random_score": {}
                }]
            }
        }
    }

    yield from get_time_query(body)


def get_hours_admin(level_1_region):
    body = {
        "_source": ["date", "text"],
        "query": {
            "function_score": {
                "query": {
                    "term":
                        {"locations.level_1_region": level_1_region}
                },
                "functions": [{
                    "random_score": {}
                }]
            }
        }
    }

    yield from get_time_query(body)



def get_function(counts, smooth=5, plot=False):
    assert smooth % 2 == 1
    extra = int((smooth - 1) / 2)
    for h in range(0, 24):
        if h not in counts:
            # Setting to at least one, so we don't get overflow problems
            counts[h] = 1
    counts = [v for h, v in sorted(counts.items())]
    fill_array = counts[-extra:] + counts + counts[:extra]
    values = []
    for h in range(extra, 24 + extra):
        values.append(statistics.mean(fill_array[h-extra:h+extra]))
    total = sum(values)
    stats = {
        hour: value / total
        for hour, value
        in zip(range(24), values)
    }

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(list(range(24)), values, color='green')
        ax.scatter(list(range(24)), counts, color='blue')
        plt.show()

    return stats


def get_time_correction(min_tweet_count=100):
    folder = os.path.join('input')
    try:
        os.makedirs(folder)
    except OSError:
        pass
    f = os.path.join(folder, "time_correction.json")
    counts = {}
    pg.cur.execute("""
        SELECT
            location_ID,
            country_location_ID
        FROM locations
        WHERE location_type = 'country'
    """)
    res = pg.cur.fetchall()
    n_res = len(res)
    for i, (adm1_location_ID, country_location_ID) in enumerate(res, start=1):
        if i % 10 == 0:
            print(f"Getting time corrections level 0: {i}/{n_res}")
        counter = Counter(get_hours_admin(adm1_location_ID))
        if sum(counter.values()) < min_tweet_count:
            if country_location_ID is not None:
                counter = Counter(get_hours_country(country_location_ID))
                if sum(counter.values()) < min_tweet_count:
                    counts[adm1_location_ID] = False
                else:
                    counts[adm1_location_ID] = get_function(counter)
        else:
            counts[adm1_location_ID] = get_function(counter)

    pg.cur.execute("""
        SELECT location_ID
        FROM locations
        WHERE (
            location_type = 'adm1'
            AND
            country_location_ID NOT IN %s
        )
        OR (
            location_type = 'adm2'
            AND
            country_location_ID IN %s
        )
    """, (LEVEL_2_COUNTRIES, LEVEL_2_COUNTRIES))
    res = pg.cur.fetchall()
    n_res = len(res)
    for i, (location_ID, ) in enumerate(res, start=1):
        if i % 100 == 0:
            print(f"Getting time corrections level 1: {i}/{n_res}")
        counter = Counter(get_hours_admin(location_ID))
        if sum(counter.values()) < min_tweet_count:
            counts[location_ID] = False
        else:
            counts[location_ID] = get_function(counter)
    print("Dumping time correction to json-file")
    with open(f, 'w') as f:
        json.dump(counts, f)


if __name__ == '__main__':
    get_time_correction()
