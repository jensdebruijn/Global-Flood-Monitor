from http.client import IncompleteRead
from urllib3.exceptions import ReadTimeoutError
from datetime import datetime, timedelta
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import os
import json

from methods.tweets import tweet_parser
from analyze_doc import DocAnalyzer

from db.elastic import Elastic
from db.postgresql import PostgreSQL

from config import DOCUMENT_INDEX, ELASTIC_HOST


class DocLoader:
    def __init__(
        self,
        doc_score_types,
        n_words,
        minimum_gram_length
    ):
        self.doc_score_types = doc_score_types
        self.n_words = n_words
        self.minimum_gram_length = minimum_gram_length

    def encode_dt(self, dt):
        return str.encode(dt.isoformat())


class DocLoaderES(DocLoader):
    def __init__(self, *args):
        DocLoader.__init__(self, *args)

    def load_timestep_es(
        self,
        es,
        doc_analyzer,
        docs_queue,
        n_docs_to_unload,
        query_start,
        query_end
    ):
        print("Loading timestep", query_start, query_end)
        query = es.build_date_query(query_start, query_end)
        for doc in es.scroll_through(index=DOCUMENT_INDEX, body=query, size=1000, source=False):
            doc = doc_analyzer.analyze_doc(doc)
            if doc:
                n_docs_to_unload.increment()
                docs_queue.put(doc)

    def load_docs(
        self,
        docs_queue,
        n_docs_to_unload,
        start,
        analysis_length,
        timestep_length,
        event_1,
        event_2,
        timestep_end_str,
        is_real_time,
        datetime=datetime
    ):
        try:
            es = Elastic(host=ELASTIC_HOST)
            pg = PostgreSQL('gfm')
            doc_analyzer = DocAnalyzer(
                es,
                pg,
                self.doc_score_types,
                self.n_words,
                self.minimum_gram_length
            )
            spinup_start = start - analysis_length + timestep_length
            self.load_timestep_es(
                es,
                doc_analyzer,
                docs_queue,
                n_docs_to_unload,
                spinup_start,
                start
            )

            timestep = 1
            timestep_end = start + timestep * timestep_length

            while timestep_end < datetime.utcnow():
                query_start = timestep_end - timestep_length

                self.load_timestep_es(
                    es,
                    doc_analyzer,
                    docs_queue,
                    n_docs_to_unload,
                    query_start,
                    timestep_end
                )

                timestep_end_str.value = self.encode_dt(timestep_end)
                timestep += 1
                timestep_end = start + timestep * timestep_length

                event_2.clear()
                event_1.set()
                event_2.wait()

            last_timestep_end = timestep_end - timestep_length
            is_real_time.value = True

            while True:
                timestep_end = datetime.utcnow()

                sleep = (
                    timedelta(minutes=3) - (timestep_end - last_timestep_end)
                ).total_seconds()
                if sleep > 0:
                    time.sleep(sleep)
                    timestep_end = datetime.utcnow()

                self.load_timestep_es(
                    es,
                    doc_analyzer,
                    docs_queue,
                    n_docs_to_unload,
                    last_timestep_end,
                    timestep_end
                )
                last_timestep_end = timestep_end
                timestep_end_str.value = self.encode_dt(timestep_end)

                event_2.clear()
                event_1.set()
                event_2.wait()
        except Exception as e:
            raise