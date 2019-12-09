from datetime import datetime, timedelta
from collections import deque, OrderedDict
from methods import sanitize
from classes import Tweet


class LastAuthorDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

    def delete_old_authors(self, dt):
        keys_to_delete = []
        for key, value in self.items():
            if value < dt - timedelta(days=14):
                keys_to_delete.append(key)
            else:
                break
        for key in keys_to_delete:
            del self[key]

    def move_to_front(self, key, new_value):
        del self[key]
        OrderedDict.__setitem__(self, key, new_value)

    def is_old_author(self, author_id, dt):
        self.delete_old_authors(dt)
        if author_id in self:
            self.move_to_front(author_id, dt)
            return True
        else:
            self.__setitem__(author_id, dt)
            return False



class LastTweetsDeque(deque):
    def __init__(self, *args, **kwargs):
        deque.__init__(self, *args, **kwargs)

    def clean_text(self, text, clean_text=sanitize.clean_text):
        return clean_text(text, lower=True)

    def ngramify(
        self,
        cleaned_text,
        tokenize=sanitize.tokenize,
        gramify=sanitize.gramify
    ):
        tokens = tokenize(cleaned_text, remove_punctuation=True)
        if len(tokens) < 5:
            return (" ".join(tokens),)
        else:
            return tuple(gramify(tokens, 5, 5, remove_tokens_with_punctuation=False))

    def text_to_ngrams(self, text):
        return self.ngramify(self.clean_text(text))

    def is_similar_to(self, *, text=None, clean_text=None, ngrams=None):
        if not ngrams:
            if not clean_text:
                clean_text = self.clean_text(text)
            ngrams = self.ngramify(clean_text)
        else:
            ngrams = tuple(ngrams)
        for ngrams_in_deque in self:
            if any(ngram in ngrams_in_deque for ngram in ngrams):
                self.remove(ngrams_in_deque)
                deque.appendleft(self, ngrams_in_deque)
                return True
        else:
            self.appendleft(ngrams)
            return False

    def appendleft(self, value):
        deque.appendleft(self, value)
        if len(self) > 100:
            self.pop()


def tweet_parser(tweet):
    post = {
        'id': 't-' + tweet['id_str'],
        # 'id_old': 't-' + str(tweet['id']),
        'date': datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'),
        'source': {
            'author': {
                'id': tweet['user']['id'],
                'location': tweet['user']['location'],
                'timezone': tweet['user']['time_zone']
            }
        }
    }
    if 'media' in tweet['entities']:
        post['media'] = [
            media['media_url']
            for media in tweet['entities']['media']
        ]
    urls = tweet['entities']['urls']
    if urls:
        urls = [url['expanded_url'] for url in urls]
        post['source']['url'] = urls

    if 'retweeted_status' in tweet:
        try:
            post['text'] = tweet['retweeted_status']['text']
            post['source']['retweet'] = True
        except KeyError:
            return None
    else:
        post['text'] = tweet['text']
        post['source']['retweet'] = False

    try:
        lang = tweet['lang']
        if lang == 'in':
            lang = 'id'
        post['source']['lang'] = lang
    except KeyError:
        return None

    try:
        # Longitude first
        location = tweet['coordinates']['coordinates']
        if location != [0, 0]:
            post['source']['coordinates'] = location
    except TypeError:
        pass

    try:
        bbox = tweet['place']['bounding_box']['coordinates'][0]
        post['source']['bbox'] = float(bbox[0][0]), float(bbox[0][1]), float(bbox[2][0]), float(bbox[2][1])
    except TypeError:
        pass

    return post


def filter_docs_by_score(docs, location_ID, kind, score):
    docs_ = []
    for doc in docs:
        if kind == 'additional_relation':
            for loc in doc['locations']:
                if loc['additional_relations'] and location_ID in loc['additional_relations'] and loc['score'] >= score:
                    docs_.append(doc)
        elif kind == 'level_0_region':
            for loc in doc['locations']:
                if loc['level_0_region'] and location_ID == loc['level_0_region'] and loc['score'] >= score:
                    docs_.append(doc)
        elif kind == 'level_1_region':
            for loc in doc['locations']:
                if loc['level_1_region'] and location_ID == loc['level_1_region'] and loc['score'] >= score:
                    docs_.append(doc)
        else:
            print('kind is', kind)
            raise NotImplementedError
    return docs_


def tweet_to_namedtuple(doc):
    if 'bbox' in doc['source']:
        bbox_center = (doc['source']['bbox'][0] + doc['source']['bbox'][2]) / 2, (doc['source']['bbox'][1] + doc['source']['bbox'][3]) / 2
    else:
        bbox_center = None
    return Tweet(
        id=doc['id'],
        text=doc['text'],
        clean_text=sanitize.clean_text(doc['text'], lower=False),
        language=doc['source']['lang'],
        date=doc['date'],
        author_id=doc['source']['author']['id'],
        author_location=doc['source']['author']['location'],
        author_timezone=doc['source']['author']['timezone'],
        coordinates=tuple(doc['source']['coordinates']) if 'coordinates' in doc['source'] else None,
        bbox_center=bbox_center,
        media=doc['source']['media'] if 'media' in doc['source'] else None,
        urls=doc['source']['urls'] if 'urls' in doc['source'] else None,
        repost=doc['source']['retweet']
    )


if __name__ == '__main__':
    last_tweets = LastTweetsDeque()
    print(len(last_tweets))
    text1 = "Niger urges people in flood-prone areas to relocate https://t.co/ukafyVg0Sp"
    print(text1)
    print(last_tweets.is_similar_to(text=text1))
    for i in range(510):
        text = f"test{i}"
        last_tweets.is_similar_to(text=text)
    text2 = "Niger urges people in flood-prone areas to relocate https://t.co/7CkEHrEXGo #Africa #Nigeria"
    print(text2)
    print(last_tweets)
    print(last_tweets.is_similar_to(text=text2))
    print(last_tweets)
    print(last_tweets.is_similar_to(text=text2))
