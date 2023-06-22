import time
import elasticsearch
from elasticsearch.helpers import scan, streaming_bulk
import logging
import json
from methods.dates import isoformat_2_date

from config import (
    ELASTIC_USER,
    ELASTIC_PASSWORD,
    ELASTIC_HOST,
    ELASTIC_PORT,
)


class Elastic(elasticsearch.Elasticsearch):
    """Extend the Elasticsearch base class with custom queries."""

    def __init__(
        self,
        host=None,
        port=None,
        *args,
        **kwargs
    ):
        """Initialize the connection and retrieve negative keywords."""
        if ELASTIC_USER:
            super().__init__(
                [{
                    'host': host or ELASTIC_HOST,
                    'port': port or ELASTIC_PORT,
                    'scheme': 'http'
                }],
                http_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
                timeout=30, max_retries=10, retry_on_timeout=True, *args, **kwargs
            )
        else:
            super().__init__(
                [{
                    'host': host or ELASTIC_HOST,
                    'port': port or ELASTIC_PORT,
                    'scheme': 'http'
                }],
                timeout=30, max_retries=10, retry_on_timeout=True, *args, **kwargs
            )
        tracer = logging.getLogger('elasticsearch')
        tracer.setLevel(logging.CRITICAL)

    def has_docs_with_locations(self, index):
        return self.n_hits(index=index, body={
            "query": {
                "exists": {
                    "field": "locations"
                }
            }
        }) > 0

    def maybe_create_document_index(self, index, score_types):
        if not self.indices.exists(index=index):
            with open('input/es_documents_index_settings.json', 'r') as f:
                es_documents_index_settings = json.load(f)
            self.indices.create(
                index=index,
                settings=es_documents_index_settings['settings'],
                mappings=es_documents_index_settings['mappings']
            )
            print(f"Created index {index}")

        if not self.has_docs_with_locations(index):
            body = {
                "properties": {
                    "locations": {
                        "properties": {
                            "scores": {
                                "properties": {
                                    score_type: {
                                      "type": "half_float"
                                    } for score_type in score_types
                                }
                            }
                        }
                    }
                }
            }
            self.indices.put_mapping(index=index, body=body)

    def bulk_operation(
        self,
        docs,
        size=1000,
        streaming_bulk=streaming_bulk,
        raise_on_error=True
    ):
        """Send docs in iterator to db."""
        if docs:
            bulk = streaming_bulk(
                self,
                docs,
                chunk_size=size,
                request_timeout=60,
                raise_on_error=raise_on_error
            )
            while True:
                try:
                    next(bulk)
                except StopIteration:
                    break

    def scroll_through(
        self,
        body,
        index=None,
        size=1000,
        scroll='10m',
        source=True,
        scan=scan,
        raise_on_error=True,
    ):
        """Return all tweets in given query as iterator."""
        if index is None:
            index = self.index
        last_id = None
        sleep = 3
        while True:
            try:
                for tweet in scan(
                    self,
                    query=body,
                    index=index,
                    scroll=scroll,
                    size=size,
                    clear_scroll=True,
                    preserve_order=True,
                    raise_on_error=raise_on_error
                ):
                    if tweet['_id'] == last_id:
                        last_id = None
                    elif not last_id:
                        last_id = None
                        if 'date' in tweet['_source']:
                            tweet['_source']['date'] = isoformat_2_date(tweet['_source']['date'])
                        if source:
                            yield tweet
                        else:
                            yield tweet['_source']
                else:
                    break
            except (
                elasticsearch.exceptions.ConnectionError,
                elasticsearch.helpers.ScanError,
                elasticsearch.exceptions.NotFoundError
            ) as e:
                try:
                    last_id = tweet['_id']
                except NameError:
                    pass
                print(f'{index}:')
                print(e)
                time.sleep(sleep)
                sleep *= 2

    def get(self, id, index=None):
        if index is None:
            index = self.index
        res = elasticsearch.Elasticsearch.get(self, index=index, id=id)
        if 'date' in res['_source']:
            res['_source']['date'] = isoformat_2_date(res['_source']['date'])
        return res

    def update(self, id, body, index=None):
        if index is None:
            index = self.index
        return elasticsearch.Elasticsearch.update(self, index=index, body=body, id=id)

    def n_hits(self, body=None, index=None):
        """Retrieve number of documents for query."""
        if index is None:
            index = self.index
        return elasticsearch.Elasticsearch.search(
            self,
            index=index,
            body=body,
            size=0
        )['hits']['total']['value']

    def search(self, body=None, index=None, **kwargs):
        """Retrieve number of documents for query."""
        if index is None:
            index = self.index
        return elasticsearch.Elasticsearch.search(
            self,
            index=index,
            body=body,
            **kwargs
        )

    def build_date_query(
        self,
        start,
        end,
        locations=False,
        filter_countries=False,
        filter_within_adm1=False,
        filter_additional_relations=False,
        filter_within_countries=False,
        filter_subbasins=False,
        source=False,
        filter_classes=False,
    ):
        assert filter_subbasins is False or isinstance(filter_subbasins, tuple)
        """
        Build query with dates from set input parameters.

        Args:
            start (datetime or isoformatted date): start date
            end (datetime or isoformatted date): end date
            locations=False (Boolean): filter by location
            filter_countries=False (int or list): list of location_ID(s) of
            countries to filter for filter_within_adm1=False (int or list): list of
            location_ID(s) of adm1 areas to filter for source=False (Boolean):
            include source of document.

        Returns:
            dict: The elasticsearch query

        """
        if not isinstance(start, str):
            start = start.isoformat()
        if not isinstance(end, str):
            end = end.isoformat()

        if not locations and not (filter_countries or filter_within_adm1 or filter_additional_relations or filter_within_countries or filter_subbasins):
            bool_query = {
                "must": [{
                    "range": {
                        "date": {
                            "gte": start,
                            "lt": end,
                        }
                    }
                }]
            }
        else:
            if filter_countries or filter_within_adm1 or filter_additional_relations or filter_within_countries or filter_subbasins:
                bool_query = {
                    "must": [{
                        "range": {
                            "date": {
                                "gte": start,
                                "lt": end,
                            }
                        }
                    }],
                    "should": [],
                    "minimum_should_match": 1
                }
                if filter_additional_relations:
                    term = "terms" if isinstance(filter_additional_relations, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.additional_relations": filter_additional_relations
                        }
                    })
                if filter_within_adm1:
                    term = "terms" if isinstance(filter_within_adm1, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.level_1_region": filter_within_adm1
                        }
                    })
                if filter_within_countries:
                    term = "terms" if isinstance(filter_within_countries, list) else "term"
                    bool_query["should"].append({
                        "bool": {
                            "should": [{
                                    term: {
                                        "locations.level_0_region": filter_within_countries
                                    }
                                }, {
                                    term: {
                                        "locations.location_ID": filter_within_countries
                                    },
                            }]
                        }
                    })
                if filter_countries:
                    term = "terms" if isinstance(filter_countries, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.location_ID": filter_countries
                        }
                    })
                if filter_subbasins:
                    term = "terms" if isinstance(filter_subbasins[1], list) else "term"
                    bool_query["should"].append({
                        term: {
                            f"locations.subbasin_ids_{filter_subbasins[0]}": filter_subbasins[1]
                        }
                    })
            else:
                bool_query = {
                    "must": [
                        {
                            "exists": {
                                "field": "locations"
                            },
                        },
                        {
                            "range": {
                                "date": {
                                    "gte": start,
                                    "lt": end,
                                }
                            }
                        }
                    ]
                }
        if filter_classes:
            term = "terms" if isinstance(filter_classes, list) else "term"
            bool_query["must"].append({
                term: {
                    "relevancer.class": filter_classes
                }
            })
        query = {
            "query": {
                "bool": bool_query
            },
            "sort": [
                {"date": {"order": "asc"}}
            ]
        }
        if source:
            query.update({
                "_source": source
            })
        return query

    def build_query(
        self,
        locations=False,
        filter_countries=False,
        filter_within_adm1=False,
        filter_additional_relations=False,
        filter_within_countries=False,
        filter_subbasins_6=False,
        filter_subbasins_8=False,
        source=False,
        filter_classes=False,
        sort='asc'
    ):
        """Create a query without date paramters with or without locations."""
        if not locations and not (filter_countries or filter_within_adm1 or filter_additional_relations or filter_within_countries or filter_subbasins_6 or filter_subbasins_8):
            bool_query = {
                "must": []
            }
        else:
            if filter_countries or filter_within_adm1 or filter_additional_relations or filter_within_countries or filter_subbasins_6 or filter_subbasins_8:
                bool_query = {
                    "must": [],
                    "should": [],
                    "minimum_should_match": 1
                }
                if filter_additional_relations:
                    term = "terms" if isinstance(filter_additional_relations, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.additional_relations": filter_additional_relations
                        }
                    })
                if filter_within_adm1:
                    term = "terms" if isinstance(filter_within_adm1, list) else "term"
                    bool_query["should"].append({
                        "bool": {
                            "must": {
                                term: {
                                    "locations.level_1_region": filter_within_adm1
                                }
                            }
                        }
                    })
                if filter_countries:
                    term = "terms" if isinstance(filter_countries, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.location_ID": filter_countries
                        }
                    })
                if filter_within_countries:
                    term = "terms" if isinstance(filter_within_countries, list) else "term"
                    bool_query["should"].append({
                        "bool": {
                            "should": [{
                                    term: {
                                        "locations.level_0_region": filter_within_countries
                                    }
                                }, {
                                    term: {
                                        "locations.location_ID": filter_within_countries
                                    },
                            }]
                        }
                    })
                if filter_subbasins_6:
                    term = "terms" if isinstance(filter_subbasins_6, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.subbasin_ids_6": filter_subbasins_6
                        }
                    })
                if filter_subbasins_8:
                    term = "terms" if isinstance(filter_subbasins_8, list) else "term"
                    bool_query["should"].append({
                        term: {
                            "locations.subbasin_ids_8": filter_subbasins_8
                        }
                    })
            else:
                bool_query = {
                    "must": [{
                        "exists": {
                            "field": "locations"
                        }
                    }]
                }
        if filter_classes:
            term = "terms" if isinstance(filter_classes, list) else "term"
            bool_query["must"].append({
                term: {
                    "relevancer.class": filter_classes
                }
            })
        query = {
            "query": {
                "bool": bool_query
            }
        }
        assert sort in ['asc', 'desc'], "Sort must be either 'asc' or 'desc'"
        query.update({
            "sort": [
                {"date": {"order": sort}}
            ]})
        if source:
            query.update({
                "_source": source
            })
        return query

    def build_query_cities_in_country(
        self,
        location_ID,
        pg
    ):
        pg.cur.execute(f"""
            SELECT
                adm1.location_ID
            FROM adm1
            LEFT JOIN locations
            ON locations.location_ID = adm1.location_ID
            WHERE locations.country_location_ID = {location_ID}
        """)
        adm1_location_IDs = [location_ID for location_ID, in pg.cur.fetchall()]
        return {
            "query": {
                "bool": {
                    "must_not": [{
                            "term": {
                                "locations.location_ID": location_ID
                            }
                        },
                        {
                            "terms": {
                                "locations.location_ID": adm1_location_IDs
                            }
                        }
                    ],
                    "must": {
                        "term": {
                            "locations.country_location_ID": location_ID
                        }
                    }
                }
            }
        }


if __name__ == '__main__':
    from pprint import pprint
    from datetime import datetime
    es = Elastic()
    # query = es.build_date_query(start=datetime(2013, 7, 1), end=datetime(2018, 7, 1), filter_countries=4155751, filter_classes=['flood'])
    query = es.build_date_query(start=datetime(2013, 7, 1), end=datetime(2018, 7, 1), filter_classes=['irrelevant'])
    # query = es.build_query()

    # query = es.build_date_query(start=datetime(2016, 3, 19), end=datetime(2016, 3, 21), filter_within_adm1=6769512)
    # for tweet in es.scroll_through(index='floods', body=query):
    #     pprint(tweet['_source'])
    #     pass
    # pprint(query)
    print(es.n_hits(index='floods', body=query))
