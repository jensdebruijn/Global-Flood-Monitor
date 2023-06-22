from db.elastic import Elastic
import gzip
import json
from tqdm import tqdm

es = Elastic()

# export all data from the database to a json file
query = {}

with gzip.open('gfm.jsonl.gz', 'wt') as f:
    n = es.count(index='gfm', body=query)['count']
    # query['sort'] = [{'date': {'order': 'desc'}}]
    for tweet in tqdm(es.scroll_through(index='gfm', body=query), total=n):
        tweet['_source']['date'] = tweet['_source']['date'].isoformat()
        tweet = json.dumps(tweet['_source'])
        f.write(tweet + '\n')