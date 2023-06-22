import os
import json
import pandas as pd
import gzip
from methods import sanitize
from datetime import datetime

from config import DOCUMENT_INDEX
from methods.tweets import tweet_parser
from db.elastic import Elastic


class Fill:
    def __init__(self):
        self.keywords = self.set_keywords()
        self.es = Elastic()

    def set_keywords(self):
        df = pd.read_excel('input/twitter_supported_languages.xlsx')
        df = df[df['implemented'] == True].set_index('language_code')['floods_filtered']
        keywords = {}
        for language, words in df.items():
            keywords[language] = set([word.strip().lower() for word in words.split(',')])
        return keywords

    def open(self, fp):
        if fp.endswith('.gzip') or fp.endswith('.gz'):
            with gzip.open(fp, 'r') as gz:
                try:
                    for tweet in gz:
                        yield tweet.decode('utf-8')
                except EOFError:
                    print('EOFError')
                    pass
        elif fp.endswith('.jsonl'):
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if line.startswith('#'):
                        continue
                    yield line.strip()
        else:
            raise NotImplementedError(f'reader for extension {fp.split(".")[-1]} not implemented')

    def generate_tweets(self, fp, start=datetime(1970, 1, 1)):
        for tweet in self.open(fp):
            try:
                tweet = json.loads(tweet)
            except json.decoder.JSONDecodeError:
                continue
            yield tweet

    def prepare_doc(self, json_doc):
        if 'limit' in json_doc:
            return None
        doc2es = tweet_parser(json_doc)
        if not doc2es:
            return None
        language = doc2es['source']['lang']
        clean_text = sanitize.clean_text(doc2es['text'], lower=False)
        clean_text_lower = clean_text.lower()
        try:
            if not any(keyword in clean_text_lower for keyword in self.keywords[language]):
                return None
        except KeyError:
            return None
        doc2es['_index'] = DOCUMENT_INDEX
        doc2es['_id'] = doc2es['id']
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
