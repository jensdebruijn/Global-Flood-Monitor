import os
import json
import pandas as pd
from methods import sanitize
from datetime import datetime

from config import DOCUMENT_INDEX
from methods.tweets import tweet_parser, tweet_to_namedtuple
from db.elastic import Elastic


class Fill:
    def __init__(self):
        self.keywords = self.set_keywords()
        self.es = Elastic()

    def set_keywords(self):
        df = pd.read_excel('input/twitter_supported_languages.xlsx')
        df = df[df['implemented'] == True].set_index('language_code')['floods_filtered']
        keywords = {}
        for language, words in df.iteritems():
            keywords[language] = set([word.strip().lower() for word in words.split(',')])
        return keywords

    def generate_tweets(self, fp, start=datetime(1970, 1, 1)):
        with open(fp, 'rb') as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                tweet = line.strip()
                try:
                    tweet = json.loads(tweet)
                except json.decoder.JSONDecodeError:
                    continue
                try:
                    language = tweet['lang']
                except KeyError:
                    continue
                clean_text = sanitize.clean_text(tweet['text'], lower=False)
                clean_text_lower = clean_text.lower()
                try:
                    if not any(keyword in clean_text_lower for keyword in self.keywords[language]):
                        continue
                except KeyError:
                    continue
                yield tweet

    def prepare_doc(self, json_doc):
        doc2es = tweet_parser(json_doc)
        doc2es['_index'] = DOCUMENT_INDEX
        doc2es['_id'] = doc2es['id']
        doc2es['_type'] = '_doc'
        doc2es['source']['type'] = 'tweet'
        return doc2es

    def prepare_docs(self, docs):
        for doc in docs:
            doc2es = self.prepare_doc(doc)
            if doc2es:
                yield doc2es

    def commit_docs(self, docs):
        self.es.bulk_operation(docs)

    def __call__(self, fp):
        tweets = self.generate_tweets(fp)
        tweets = self.prepare_docs(tweets)
        self.commit_docs(tweets)


if __name__ == '__main__':
    filler = Fill()
    filler('input/example.jsonl')
