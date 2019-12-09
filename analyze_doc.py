import json
import pandas as pd
from datetime import datetime
from math import e
from os import path, makedirs, listdir
from re import compile
from pprint import pprint
from itertools import zip_longest
from itertools import combinations, permutations
from operator import itemgetter
from collections import OrderedDict, defaultdict, namedtuple

import elasticsearch.exceptions

from methods import sanitize
from methods.tweets import tweet_to_namedtuple
from classes import Location, Article, AnalyzedDoc, BaseDoc

from config import DOC_SCORE_TYPES, DOCUMENT_INDEX


LocScores = namedtuple("LocScores", DOC_SCORE_TYPES.keys())


class LastUserLocationDict(OrderedDict):
    def __init__(self, size, *args, **kwargs):
        self.size = 10000
        self.pop = False
        OrderedDict.__init__(self, *args, **kwargs)

    def getandmove(self, key):
        OrderedDict.move_to_end(self, key)
        return OrderedDict.__getitem__(self, key)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if not self.pop and len(self) > self.size:
            self.pop = True
        if self.pop:
            self.popitem(last=False)

class DocAnalyzer:
    def __init__(
        self,
        es,
        pg,
        doc_score_types,
        n_words,
        minimum_gram_length,
    ):
        """Set some initial values and call the __init__ of the its parent classes"""
        self.es = es
        self.pg = pg
        self.doc_score_types = doc_score_types
        self.minimum_gram_length = minimum_gram_length

        # The size order of the administrative levels. Can be used for sorting
        # locations by its size.
        self.order_of_preference = {
            "continent": 0,
            "country": 1,
            "adm1": 2,
            "town": 3,
            "adm2": 4,
            "adm3": 5,
            "adm4": 6,
            "adm5": 7,
            "landmark": 8,
        }

        self._set_language_info()
        self._set_country_alternative_names()
        self._set_discard_patterns()

        # Dictionary to get the continent(s) a country is in
        # self.country_2_continent = self._load_country_2_continent()

        # Dictonary with most common words for each language (max 10000)
        # Alternative names for each country
        self._set_region_names()

        self.most_common_words = self._get_most_common_words(n_words)
        self.language_matches = self._load_language_matches()

        self.lastuserlocationdict = LastUserLocationDict(10000)

        self.time_zones_per_region = self.get_tzs_per_region()

    def _set_country_alternative_names(self):
        self.alternative_names = {
            language: set()
            for language in self.language_keywords
        }
        self.pg.cur.execute("""
            SELECT alternate_name, isolanguage
            FROM locations
            JOIN alternate_names
            ON locations.location_ID = alternate_names.location_ID
            WHERE locations.location_type = 'country'
            AND isolanguage IN %s
        """, (tuple(self.language_keywords.keys()), ))
        for alternate_name, isolanguage in self.pg.cur.fetchall():
            self.alternative_names[isolanguage].add(alternate_name)

    def _set_discard_patterns(self):
        self.discard_patterns = (
            compile(r"(\b[0-9][0-9]{0,2}\s*-\s*[0-9][0-9]{0,2}\b)"),
            compile(r"(\b[0-9][0-9]{0,2}\s(?:[a-zA-Z]+\s){1,2}[0-9][0-9]{0,2}\b)"),
            compile(r"(\?)""")
        )

    def get_tzs_per_region(self):
        with open('input/timezone_scores_per_region.json', 'r') as f:
            return json.load(f)

    def _load_language_matches(self):
        def conv(x):
            if isinstance(x, float):
                return set()
            if isinstance(x, int):
                return set((x, ))
            else:
                return set(int(i) for i in x.split(';'))
        df = pd.read_excel(
            'input/language_matches.xlsx'
        ).set_index('language')
        df['match'] = df['match'].apply(conv)
        return df['match'].to_dict()

    def get_tokens_space_separable(self, clean_text, lang):
        tokens = sanitize.tokenize(clean_text, remove_punctuation=False)
        return tokens

    def get_ngrams_space_separable(self, tokens):
        return sanitize.gramify(tokens, 1, 3)

    def _set_language_info(self):
        f = path.join(
            'input',
            'twitter_supported_languages.xlsx'
        )
        df = pd.read_excel(f).query('implemented == True')

        self.language_keywords, self.ngram_mapper, self.toponym_capitalization = {}, {}, {}
        for _, row in df.iterrows():
            language_code = row['language_code']
            keywords = row['floods_filtered']
            language_whitelist = set()
            if isinstance(keywords, str):
                for keyword in keywords.split(','):
                    language_whitelist.add(keyword.strip().lower())
            self.language_keywords[language_code] = language_whitelist
            self.toponym_capitalization[language_code] = row['toponym_capitalization']

    def get_ngrams(self, clean_text, lang):
        return self.ngram_mapper[lang](clean_text, lang)

    def _load_country_2_continent(self):
        """Return a dictionary with the continent(s) a country is on"""
        self.pg.cur.execute("SELECT location_ID, continents FROM countries")
        country_2_continent = {
            country: continents
            for country, continents in self.pg.cur.fetchall()
        }
        return country_2_continent

    def _set_region_names(self):
        self.pg.cur.execute("""
            SELECT location_ID, name, location_type
            FROM locations
            WHERE location_type IN %s
        """, (('country', 'adm1'), )
        )
        region_names = defaultdict(set)
        country_alternative_names_set = set()
        for location_ID, name, location_type in self.pg.cur.fetchall():
            if name:
                name = name.lower()
                if location_type == 'country':
                    country_alternative_names_set.add(name)
                region_names[name].add(location_ID)
        self.region_names = dict(region_names)

    def _get_most_common_words(self, n):
        """Read n (max = 10000) most common words from the database"""
        if n > 10000:
            print("Can only download 1000 most common using this website \
             - setting n to 1000")
            n = 10000

        d = {}
        self.pg.cur.execute("""
            SELECT DISTINCT language
            FROM most_common_words
        """)
        for language, in self.pg.cur.fetchall():
            d[language] = set()
            self.pg.cur.execute("""
                SELECT word
                FROM most_common_words
                WHERE language = %s
                ORDER BY n ASC LIMIT %s
            """, (language, n))
            for word, in self.pg.cur.fetchall():
                word = word.lower()
                d[language].add(word)
        return d

    def extract_user_locations_child(
        self,
        child,
        original_name,
        parent_name,
        parent_location,
    ):
        try:
            locations = [Location({**loc, **{"toponym": child}}) for loc in self.es.get(
                index='locations',
                id=child
            )['_source']['locations']]
        except (elasticsearch.exceptions.NotFoundError, ValueError):
            return parent_location
        else:
            locations = sorted(locations, key=lambda loc: loc.translations, reverse=True)
            for loc in locations:
                if ('abbr' not in loc.languages or original_name in loc.abbreviations and loc.is_child_of(parent_location)):
                    return loc
            else:
                return parent_location

    def find_user_location_town(
        self,
        name,
        original_name,
    ):
        try:
            locations = self.es.get(index='locations', id=name)['_source']['locations']
        except (elasticsearch.exceptions.NotFoundError, ValueError):
            return []
        else:
            locations = sorted(locations, key=itemgetter('translations'), reverse=True)
            for loc in locations:
                if (
                    'abbr' not in loc['languages']
                    or original_name in loc['abbreviations']
                ):
                    loc.update({"toponym": name})
                    return [Location(loc)]
            else:
                return []

    def find_user_location(self, u_location):
        """Parses the location field of the user. The user field is split at a comma if present. If a comma is present,
        it is assumed that the part before the comma is the city and the second part the country. If no comma is present
        we assume that the user field specifies the country. The function returns False if not location is found, and a tuple
        otherwise."""
        if not u_location:
            return []

        for ch in ('/', ' and ', '&', '|', ' - ', ';'):
            if ch in u_location:
                return [
                    loc for split in u_location.split(ch) for loc in self.find_user_location(split)
                ]

        u_location = u_location.strip().replace('.', '')
        u_location_lower = u_location.lower()

        if ' - ' in u_location_lower:
            u_location_splitted = u_location_lower.split(' - ')
        else:
            u_location_splitted = u_location_lower.split(',')
        if len(u_location_splitted) == 1:
            u_location_lower_splitted_space = u_location_lower.split(' ')
            for i in range(1, len(u_location_lower_splitted_space) + 1):
                name = ' '.join(u_location_lower_splitted_space[-i:])
                if name in self.region_names:
                    parent_location_IDs = self.region_names[name]
                    parent_locations = self.es.get(index='locations', id=name)['_source']['locations']
                    parent_locations = [Location(loc) for loc in parent_locations if loc['location_ID'] in parent_location_IDs]

                    original_name = ' '.join(u_location.split(' ')[-len(name.split(' ')):])
                    parent_locations = [
                        parent_location
                        for parent_location
                        in parent_locations
                        if not parent_location.abbreviations or original_name in parent_location.abbreviations
                    ]
                    if parent_locations:
                        break
                else:
                    continue
            else:
                return self.find_user_location_town(u_location_lower, u_location)
            child = u_location_lower[:-len(name)].strip()
            if child:
                original_name_i = u_location_lower.index(child)
                original_name = u_location[original_name_i:original_name_i+len(child)]
                locations = []
                for parent_location in parent_locations:
                    locations.append(self.extract_user_locations_child(child, name, original_name, parent_location))
                return locations
            else:
                return parent_locations
        elif len(u_location_splitted) == 2:
            child, parent = u_location_splitted
            child, parent = child.strip(), parent.strip()
            if parent not in self.region_names:
                # Parent is not found, so might be neighborhood, town rather than town, country.
                return self.find_user_location_town(parent, u_location.split(',')[-1].strip())
            else:
                parent_location_IDs = self.region_names[parent]
                parent_locations = self.es.get(index='locations', id=parent.lower())['_source']['locations']
                parent_locations = [Location(loc) for loc in parent_locations if loc['location_ID'] in parent_location_IDs]
                original_parent_name = u_location.split(',')[-1].strip()
                parent_locations = [
                    parent_location
                    for parent_location
                    in parent_locations
                    if not parent_location.abbreviations or original_parent_name in parent_location.abbreviations
                ]
                if not parent_locations:
                    return self.find_user_location_town(parent, original_parent_name)
                else:
                    return [
                        self.extract_user_locations_child(child, parent, original_parent_name, parent_location)
                        for parent_location in parent_locations
                    ]
        elif len(u_location_splitted) == 3:
            if ' - ' in u_location_lower:
                u_location_original_splitted = u_location.split(' - ')
            else:
                u_location_original_splitted = u_location.split(',')
            return self.find_user_location(','.join([u_location_original_splitted[0] + u_location_original_splitted[-2]]))
        else:
            return []

    def is_language_match(
        self,
        location,
        doc
    ):
        return location.country_location_ID in self.language_matches[doc.language] if doc.language is not None else False

    def match_user_locations(self, loc, user_locations):
        """Returns true if a user location mathes given location"""
        user_location = sorted(
            user_locations,
            key=lambda loc: loc.translations,
            reverse=True
        )[0]
        return self.get_family_score_ratio(loc, user_location)

    def get_score_ratio_loc_coordinates(self, loc, coordinates):
        if loc.location_ID in (loc.level_0_region, loc.level_1_region):
            res = loc.contains(coordinates, self.pg)
            if res is not None:
                return res
            else:
                return 0
        else:
            distance = loc.distance_to_coordinates(coordinates, self.pg)
            if distance is not None and distance < 500_000:
                return e ** (-distance / 100_000)
            else:
                return 0

    def get_family_score_ratio(self, loc1, loc2):
        if loc1.location_ID == loc2.location_ID:
            return 1
        is_sibblings = loc1.is_sibbling_with(loc2)
        if is_sibblings is not False:
            if is_sibblings == 0:
                return 0
            else:
                score_ratio = .5
                distance_between_sibblings = loc1.distance_between_sibblings(
                    loc2,
                    pg=self.pg
                )
                if distance_between_sibblings is not None and distance_between_sibblings < 500_000:
                    score_ratio = max(
                        e ** (-distance_between_sibblings / 100_000),
                        score_ratio
                    )
                else:
                    score_ratio = 0
                return score_ratio
        else:
            parental_relation = loc1.is_parental_relation(loc2)
            if parental_relation is not False:
                if (
                    (loc1.type == 'adm1' and loc2.type == 'country') or
                    (loc1.type == 'country' and loc2.type == 'adm1')
                ):
                    score_ratio = 1
                elif parental_relation[1] > 0:
                    score_ratio = .5
                else:
                    score_ratio = 0  # could set this to 0.25 for less sure analysis
            else:
                score_ratio = 0
            return score_ratio

    def strip_tags(self, ngrams, tags):
        new_ngrams = []
        subsetted_ngrams = set()
        for ngram in ngrams:
            subsetted = False
            for tag in tags:
                while True:
                    try:
                        i = ngram.lower().index(tag)
                    except ValueError:
                        break
                    else:
                        new_ngram = (
                            ngram[:i] + ngram[i + len(tag):]
                        ).strip().replace('  ', ' ')
                        new_ngrams.append(new_ngram)
                        subsetted_ngrams.add(new_ngram)
                        subsetted = True
                        break
                if subsetted:
                    break
            else:
                new_ngrams.append(ngram)
        return new_ngrams, subsetted_ngrams

    def find_first_letter_original_ngram(self, text, ngram):
        return text[text.lower().index(ngram)]

    def get_digits(self, ngrams):
        for ngram in ngrams:
            try:
                yield int(ngram)
            except ValueError:
                pass

    def contains_recent_year(self, ngrams):
        digit_tokens = self.get_digits(ngrams)
        if any(1800 < int(token) < datetime.utcnow().year for token in digit_tokens):
            return True
        else:
            return False

    def get_lowercase_ngrams(
        self,
        ngrams,
        lang,
    ):
        all_lower_case_ngrams = [None] * len(ngrams)
        ngrams_orgininal = {}
        for i, ngram in enumerate(ngrams):
            ngram_lower = ngram.lower()
            if ngram_lower not in ngrams_orgininal:
                ngrams_orgininal[ngram_lower] = ngram
            elif not ngrams_orgininal[ngram_lower].istitle():
                ngrams_orgininal[ngram_lower] = ngram
            all_lower_case_ngrams[i] = ngram_lower

        # Create set from all ngrams that are longer than the
        # MINIMUM_GRAM_LENGTH unless part of the list of alternative names
        # for countries
        lower_case_ngrams = list(set(
            ngram
            for ngram
            in all_lower_case_ngrams
            if (
                len(ngram) >= self.minimum_gram_length or
                ngram in self.alternative_names[lang]
            )
        ))
        return all_lower_case_ngrams, lower_case_ngrams, ngrams_orgininal

    def find_toponyms_in_toponym(self, documents):
        # Build set of toponyms that are part of other toponyms
        # (e.g. Remove York if New York is also in the set)
        topynyms_in_toponym = set()
        found_ngrams = set(doc['_id'] for doc in documents)
        for ngram1, ngram2 in permutations(found_ngrams, 2):
            if ' ' + ngram1 + ' ' in ' ' + ngram2 + ' ':
                topynyms_in_toponym.add(ngram1)
        return topynyms_in_toponym

    def discard_similar_tokens(self, tokens, tokens_other_tweets_by_user, language):
        tokens_other_tweets_by_user = zip_longest(*tokens_other_tweets_by_user, fillvalue=None)
        for i, (this_token, that_tokens) in enumerate(zip(tokens, tokens_other_tweets_by_user)):
            n_matches = sum([1 for that_token in that_tokens if that_token == this_token])
            if n_matches >= 2 and this_token.lower() not in self.language_keywords[language]:
                continue
            else:
                break

        try:
            if 4 > i > 1:
                tokens = tokens[i:]
        except UnboundLocalError:
            return tokens

        return tokens

    def validate_and_analyze_tweet_text(self, doc, check_with_previous_tweets=False):
        if any(discard_pattern.search(doc.text) is not None for discard_pattern in self.discard_patterns):
            return None

        tokens = self.get_tokens_space_separable(doc.clean_text, doc.language)

        if check_with_previous_tweets:

            document_date = doc.date.isoformat()

            query = {
                'size': 20,
                'query': {
                    'bool': {
                        'must': [
                            {
                                'term': {
                                    'source.author.id': doc.author_id
                                }
                            },
                            {
                                'range': {
                                    'date': {
                                        'lt': document_date
                                    }
                                }
                            }
                        ]
                    }
                },
                'sort': {
                    'date': 'desc'
                }           
            }

            hits = self.es.search(index=DOCUMENT_INDEX, body=query)['hits']['hits']
            n_hits = len(hits)

            if n_hits < 20:
                query = {
                    'size': 20 - n_hits,
                    'query': {
                        'bool': {
                            'must': [
                                {
                                    'term': {
                                        'source.author.id': doc.author_id
                                    }
                                },
                                {
                                    'range': {
                                        'date': {
                                            'gt': document_date
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    'sort': {
                        'date': 'asc'
                    }           
                }
                res = self.es.search(index=DOCUMENT_INDEX, body=query)
                hits.extend(res['hits']['hits'])
                    
            if hits:
                other_tweets = [
                    tweet_to_namedtuple(hit['_source'])
                    for hit in hits
                ]

                other_tweets = [
                    other_tweet
                    for other_tweet
                    in other_tweets
                    if other_tweet.clean_text != doc.clean_text
                ]

                tokens_other_tweets = [
                    self.get_tokens_space_separable(other_tweet.text, other_tweet.language)
                    for other_tweet in other_tweets
                ]

                tokens = self.discard_similar_tokens(tokens, tokens_other_tweets, doc.language)

                

        # if not self.is_event_related(doc.text, doc.language):
        #     print('not-related', doc.text)
        #     return None
        # print('related', doc.text)

        ngrams = self.get_ngrams_space_separable(tokens)
        if self.contains_recent_year(ngrams):
            return None

        ngrams = sanitize.discard_ngrams_with_digits(ngrams)

        # Get all the tags for analysis in a language. Sort them by lenght.
        # This is important because all keywords a doc is found by are
        # removed before futher analysis.
        try:
            tags = sorted(self.language_keywords[doc.language], reverse=True)
        except KeyError:
            return None
        # Remove all tags from the tokens
        ngrams, subsetted_ngrams = self.strip_tags(ngrams, tags)
        ngrams = tuple(ngram for ngram in ngrams if ngram)

        _, lower_case_ngrams, ngrams_orgininal = self.get_lowercase_ngrams(
            ngrams,
            doc.language
        )

        if not lower_case_ngrams:
            return None

        return doc, tags, lower_case_ngrams, ngrams_orgininal, subsetted_ngrams

    def extract_candidates_per_toponym(self, locations_dict, doc, filter=None, toponym_languages=None, filter_most_common_words=False):
        toponym = locations_dict['_id']
        if not toponym_languages:
            toponym_languages = [doc.language]

        assert isinstance(toponym_languages, list)

        # Do not consider if toponym is part of other toponym
        if filter and toponym in filter:
            return None

        locations = [
            Location({**loc, **{'toponym': toponym}}, scores=True) for loc in locations_dict['_source']['locations'] if (
                'general' in loc['languages'] or
                'partial' in loc['languages'] or
                'abbr' in loc['languages'] or
                any(language in loc['languages'] for language in toponym_languages)
            )
        ]

        if filter_most_common_words and toponym in self.most_common_words[doc.language]:
            locations_ = []
            for location in locations:
                self.pg.cur.execute("""
                    SELECT population FROM locations WHERE location_ID = %s
                """, (location.location_ID, ))
                population, = self.pg.cur.fetchone()
                if population is not None and population > 100_000:
                    locations_.append(location)
            locations = locations_

        # If multiple locations bear the same name and are family, only keep
        # the one with the highest number of translations in the locations
        # database. This is a proxy for the importance of the locations
        if len(locations) >= 2:
            to_discard = set()
            for loc1, loc2 in combinations(locations, 2):
                if loc1.is_child_of(loc2) or loc2.is_child_of(loc1):
                    sorted_locs = sorted(
                        sorted(
                            [loc1, loc2],
                            key=lambda loc: self.order_of_preference[loc.type]
                        ),
                        key=lambda loc: loc.translations,
                        reverse=True
                    )
                    to_discard.add(sorted_locs[1].location_ID)

            if to_discard:
                locations = [
                    loc for loc in locations
                    if loc.location_ID not in to_discard
                ]
        if locations:
            return toponym, locations
        else:
            return None

    def get_locations_per_toponyms(self, ids, language):
        if ids:
            locations_per_toponym = self.es.mget(
                index='locations',
                body={'ids': ids}
            )['docs']

            # select only found documents
            return tuple(locations for locations in locations_per_toponym if locations['found'] is True)
        else:
            return None

    def parse_tweet(
        self,
        doc,
        first_word_regex=compile(r'(?:^|(?:[.!?:]\s))(\w+)')
    ):
        doc = tweet_to_namedtuple(doc)

        if not doc:
            return None
        res = self.validate_and_analyze_tweet_text(doc, check_with_previous_tweets=False)
        if res:
            doc, tags, lower_case_ngrams, ngrams_orgininal, _ = res
        else:
            return None

        if self.toponym_capitalization[doc.language] is True:
            first_word_sentences = first_word_regex.findall(doc.clean_text)
            capital_ngrams = [
                ngram for ngram in lower_case_ngrams if ngrams_orgininal[ngram][0].isupper()
            ]
            capital_ngrams_excluding_first_words = [
                ngram for ngram in capital_ngrams
                if not any(ngrams_orgininal[ngram].startswith(first_word) for first_word in first_word_sentences)
            ]
            if capital_ngrams_excluding_first_words:
                lookup_ngrams = capital_ngrams
            else:
                lookup_ngrams = lower_case_ngrams
        else:
            lookup_ngrams = lower_case_ngrams

        locations_per_toponym = self.get_locations_per_toponyms(lookup_ngrams, doc.language)
        if not locations_per_toponym:
            return None

        toponyms_in_toponym = self.find_toponyms_in_toponym(locations_per_toponym)
        filter = toponyms_in_toponym | set(tags)
        candidate_locations = [
            self.extract_candidates_per_toponym(
                locations,
                doc,
                filter=filter,
                filter_most_common_words=True
            ) for locations in locations_per_toponym
        ]
        candidate_locations = [loc for loc in candidate_locations if loc is not None]
        return doc, candidate_locations

    def get_doc_scores(self, doc, candidate_locations):
        user_locations = None
        doc_locations = {}

        if len(candidate_locations) > 1:
            total_candidate_location_score = sum(
                1 / len(locations)
                for locations in candidate_locations.values()
            )

        # Loop through all documents
        for toponym, locations in candidate_locations.items():
            for loc in locations:
                if self.is_language_match(loc, doc):
                    loc.scores['language_match'] = self.doc_score_types['language_match']
                else:
                    loc.scores['language_match'] = 0

            # match doc coordinates
            if hasattr(doc, 'coordinates') and doc.coordinates:
                for loc in locations:
                    loc.scores['coordinates_match'] = self.doc_score_types['coordinates_match'] * self.get_score_ratio_loc_coordinates(loc, doc.coordinates)
            else:
                for loc in locations:
                    loc.scores['coordinates_match'] = 0

            # match doc time zone
            if hasattr(doc, 'author_timezone') and doc.author_timezone:
                for loc in locations:
                    time_zone_score = loc.matches_time_zone(doc.author_timezone, self.time_zones_per_region)
                    if time_zone_score:
                        loc.scores['time_zone'] = self.doc_score_types['time_zone'] * time_zone_score
                    else:
                        loc.scores['time_zone'] = 0
            else:
                for loc in locations:
                    loc.scores['time_zone'] = 0

            # Match doc user location
            if hasattr(doc, 'author_location') and doc.author_location:
                if user_locations is None:
                    if doc.author_location:
                        try:
                            user_locations = self.lastuserlocationdict.getandmove(
                                doc.author_location
                            )
                        except KeyError:
                            user_locations = self.find_user_location(doc.author_location)
                            self.lastuserlocationdict[doc.author_location] = user_locations
                    else:
                        user_locations = False

                if user_locations:
                    for loc in locations:
                        user_locations_score_ratio = self.match_user_locations(
                            loc,
                            user_locations
                        )
                        loc.scores['user_home'] = user_locations_score_ratio * self.doc_score_types['user_home']
                else:
                    for loc in locations:
                        loc.scores['user_home'] = 0
            else:
                for loc in locations:
                    loc.scores['user_home'] = 0

            # Match doc bounding box
            # Do not consider a bounding box if a coordinate is already present.
            if hasattr(doc, 'bbox_center') and doc.bbox_center:
                for loc in locations:
                    loc.scores['bbox'] = self.doc_score_types['bbox'] * self.get_score_ratio_loc_coordinates(loc, doc.bbox_center)
            else:
                for loc in locations:
                    loc.scores['bbox'] = 0

            for loc in locations:
                loc.scores['family'] = 0

            # If other locaions are already added to the doc_locations we can
            # check for family. If family is true, set both to True
            if doc_locations:
                for location_IDs in doc_locations.values():
                    additional_family_scores = defaultdict(int)
                    for loc1 in location_IDs:
                        for loc2 in locations:
                            family_score_ratio = self.get_family_score_ratio(loc1, loc2)

                            additional_family_score_loc1 = family_score_ratio * (1 / len(locations)) / total_candidate_location_score

                            additional_family_score_loc2 = family_score_ratio * (1 / len(location_IDs)) / total_candidate_location_score

                            additional_family_scores[loc1.location_ID] = max(additional_family_scores[loc1.location_ID], additional_family_score_loc1)

                            additional_family_scores[loc2.location_ID] = max(additional_family_scores[loc2.location_ID], additional_family_score_loc2)

                    for location_IDs in doc_locations.values():
                        for loc in location_IDs:
                            loc.scores['family'] += additional_family_scores[loc.location_ID] * self.doc_score_types['family']
                    for loc in locations:
                        loc.scores['family'] += additional_family_scores[loc.location_ID] * self.doc_score_types['family']

            doc_locations[toponym] = locations

        for locations in doc_locations.values():
            for loc in locations:
                loc.scores = LocScores(
                    **loc.scores
                )

        return doc_locations

    def analyze_doc(
        self,
        doc
    ):
        if not doc:
            return None
        res = self.parse_tweet(doc)
        if not res:
            return None
        else:
            analyzed_doc, candidate_locations = res
        clean_text = analyzed_doc.clean_text

        candidate_locations = dict([
            loc for loc in candidate_locations if loc is not None
        ])
        if not candidate_locations:
            return None

        doc_locations = self.get_doc_scores(analyzed_doc, candidate_locations)
        if not doc_locations:
            return None

        return analyzed_doc.id, AnalyzedDoc(
            locations=doc_locations,
            resolved_locations=None,
            author_id=analyzed_doc.author_id,
            date=analyzed_doc.date,
            text=analyzed_doc.text,
            clean_text=clean_text,
            language=analyzed_doc.language,
            repost=analyzed_doc.repost
        )