from db.elastic import Elastic
from methods.sanitize import clean_text
from db.remove_elasticsearch_fields import remove_field_from_index
from classifier.predict import Predictor
from datetime import datetime
import gzip
from itertools import chain, islice

from config import DOCUMENT_INDEX

refresh = False

def chunker(iterable, chunk_size):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, chunk_size - 1))

def classify():
    es = Elastic()

    classify_per = 10_000

    if refresh:
        remove_field_from_index(DOCUMENT_INDEX, 'event_related')

    predictor = Predictor()

    query = {
        'query': {
            "bool": {
                "must": [
                    {
                        'exists': {
                            'field': 'locations'
                        }
                    }
                ],
                "must_not": {
                    'exists': {
                        'field': 'event_related'
                    }
                }
            }
        }
    }
    n = es.n_hits(index=DOCUMENT_INDEX, body=query)
    tweets = es.scroll_through(index=DOCUMENT_INDEX, body=query)
    tweet_subset = []
    for i, tweet in enumerate(tweets):
        if not i % classify_per:
            print(f"{i}/{n} ({int(i/n*100)}%) - {datetime.now()}")
        tweet_subset.append(tweet)

        if len(tweet_subset) == classify_per:
            IDs = []
            examples = []
            for tweet in tweet_subset:
                tweet = tweet['_source']
                IDs.append(tweet['id'])
                example = {
                    "id": tweet['id'],
                    "sentence1": clean_text(tweet['text'], lower=False),
                    "label": 0
                }
                examples.append(example)

            labels = predictor(examples)
            es_update = []
            for ID, label in zip(IDs, labels):
                es_update.append({
                    'doc': {
                        'event_related': True if label == 'yes' else False
                    },
                    '_index': DOCUMENT_INDEX,
                    '_id': ID,
                    '_op_type': 'update',
                })

            es.bulk_operation(es_update)

            tweet_subset = []


def export():
    es = Elastic()

    query = {}
    tweets = es.scroll_through(index=DOCUMENT_INDEX, body=query)
    n = 1
    with gzip.open('tweets.gz', 'wt', encoding='utf-8') as f:
        for tweet in tweets:
            if not n % 1000:
                print(f"{n} - {datetime.now()}")
            tweet = tweet['_source']
            if 'locations' in tweet:
                n += 1
                ID = tweet['id']
                text = clean_text(tweet['text'], lower=False)
                f.write(f'{ID}\t{text}\n')


def classify_gzip(classify_per=10000):
    def get_tweets():
        with gzip.open('tweets.gz', 'rt', encoding='utf-8') as f:
            for line in f.readlines():
                ID, text = line.strip().split('\t')
                yield ID, text

    def classify(tweet_subset):
        IDs = []
        texts = []
        examples = []
        for ID, text in tweet_subset:
            IDs.append(ID)
            example = {
                "id": ID,
                "sentence1": text,
                "label": 0
            }
            examples.append(example)

        labels = predictor(examples)
        for ID, label in zip(IDs, labels):
            f.write(f'{ID}\t{label}\n')

    with gzip.open('tweets_labelled.gz', 'wt', encoding='utf-8') as f:
        predictor = Predictor()
        for i, tweet_subset in enumerate(chunker(get_tweets(), classify_per)):
            print(i)
            classify(tweet_subset)
        

def gzip_to_es(move_per=10000):
    es = Elastic()

    def get_labels():
        with gzip.open('tweets_labelled.gz', 'rt', encoding='utf-8') as f:
            for line in f.readlines():
                ID, label = line.strip().split('\t')
                yield ID, label

    def move_to_db(labels):
        es_update = []
        for ID, label in labels:
            es_update.append({
                'doc': {
                    'event_related': True if label == 'yes' else False
                },
                '_index': DOCUMENT_INDEX,
                '_id': ID,
                '_op_type': 'update',
            })

        es.bulk_operation(es_update)

    for i, labels in enumerate(chunker(get_labels(), move_per)):
        print(i)
        move_to_db(labels)

if __name__ == '__main__':
    export()
    classify_gzip()
    gzip_to_es()