import json
from datetime import datetime
import gzip
from tqdm import tqdm

# get name of file

export_name = __file__.replace('.py', '.jsonl')
country_geonameid = "g-2802361"

with open(export_name, 'w') as json_file:
    with gzip.open("gfm.jsonl.gz", 'rb') as f:
        for line in tqdm(f):
            line = line.strip()
            tweet = json.loads(line)
            if 'locations' not in tweet:
                continue
            locations = tweet['locations']
            for location in locations:
                if location['level_0_region'] == country_geonameid:
                    tweet = json.dumps(tweet, default=str)
                    json_file.write(tweet + '\n')
                    break

# query = es.build_date_query(
#     start=datetime(2014, 1, 1),
#     end=datetime.utcnow(),
#     filter_within_countries=country_geonameid
# )
# print(query)

# with open(export_name, 'w') as f:
#     for tweet in es.scroll_through(index='gfm', body=query):
#         tweet = tweet['_source']
#         tweets = json.dumps(tweet, default=str)
#         f.write(tweets + '\n')