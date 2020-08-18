import argparse
import sys
import multiprocessing as mp
import os
import sys
import traceback

from datetime import datetime, timedelta
from ctypes import c_bool
from datetime import timedelta
from methods.dates import daterange, isoformat_2_date

from geoparser import Geoparser
from event_detector import EventDetector
from classifier.predict import Predictor as TextClassifier

from db import remove_elasticsearch_fields
from db.elastic import Elastic
from db.postgresql import PostgreSQL

from config import DOCUMENT_INDEX, DOC_SCORE_TYPES, DETECTION_PARAMETERS, ELASTIC_HOST, START_DATE

n_words = 10_000
minimum_gram_length = 4


if sys.version_info < (3, 6):
    print("This application requires python 3.6+")
    sys.exit(1)


def parse_bool(s):
    if s.lower() in ('yes', 'y', 'true'):
        return True
    elif s.lower() in ('no', 'n', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Not a valid boolean '{s}'")


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        self.events = [arg for arg in kwargs['args'] if isinstance(arg, mp.synchronize.Event)]
        self.event_1 = kwargs['args'][7]
        self.event_2 = kwargs['args'][8]
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            for event in self.events:
                event.set()

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class Counter(object):
    def __init__(self, initval=0):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()

    def increment(self, n=1):
        with self.lock:
            self.val.value += n

    def decrease(self, n=1):
        with self.lock:
            self.val.value -= n

    def value(self):
        with self.lock:
            return self.val.value


class Detection(Geoparser):
    def __init__(
        self,
        doc_loader,
        n_words,
        classify_tweets,
        minimum_gram_length,
        max_distance_entities_doc,
        doc_score_types,
    ):
        """Get out doc_analyzer, save the minimum score neccesary for docs
        and if the event detection module is turned on, initalize the class
        for that (spinup)"""
        self.n_words = n_words
        self.classify_tweets = classify_tweets
        self.es = Elastic(host=ELASTIC_HOST)
        self.check_toponym_index()
        self.pg = PostgreSQL('gfm')
        super().__init__(self.pg, self.es, doc_score_types, max_distance_entities_doc)
        if self.classify_tweets == 'bert':
            self.text_classifier = TextClassifier()
        self.docs = {}
        doc_loader_args = (
            doc_score_types,
            n_words,
            minimum_gram_length
        )
        from doc_loader import DocLoaderES
        self.doc_loader = DocLoaderES(*doc_loader_args)

    def check_toponym_index(self):
        if not self.es.indices.exists("locations"):
            print("Toponym index does not exist")
            sys.exit()

    def maybe_set_table_name(self, regions, detection_parameters):
        for i, setting in enumerate(detection_parameters):
            if setting.run_name is None:
                run_name = (
                    "floods"
                    f"_{regions}"
                    f"_{int(setting.location_threshold*100)}"
                    f"_{setting.factor}_{int(setting.fraction*10)}"
                    f"_{setting.base}"
                )
                detection_parameters[i] = setting._replace(
                    run_name=run_name
                )

    def initial_detection(
        self,
        start,
        end,
    ):
        print("Initial detection")
        for query_start, query_end in daterange(
            start,
            end,
            timedelta(days=1),
            ranges=True
        ):
            query_end = min(query_end, end)
            print("Initial detection:", query_start, "-", query_end)
            query = self.es.build_date_query(
                query_start,
                query_end,
                locations=True,
            )
            query['query']['bool']['must'].append({
                'term': {
                    'event_related': True
                }
            })
            documents = self.es.scroll_through(
                index=DOCUMENT_INDEX,
                body=query,
                source=False
            )
            self.event_detector.detect_events_l(documents, is_real_time=mp.Value(c_bool, False), convert_to_named_tuple=True)
        print("Finished initial detection")

    def run(
        self,
        start,
        spinup_time,
        timestep_length,
        analysis_length,
        detection_parameters,
        regions,
        real_time,
        max_n_docs_in_memory=None,
        check_previous_docs=True,
        geoparsing_start=False,
        update_locations=True,
        end=False,
        load_detectors=False,
        detection=True,
    ):

        """This program uses 2 processes. The main process (this one) that
        analyzes groups of docs and detects based on this. In addition a
        child process is spawned that reads the docs from the database or
        receives them from a stream. This process is the doc_loader.
        Two events, event_1 and event_2, regulate the execution of both
        processes. First the doc_loader loads the docs used for the spinup
        from the database, then the docs for the first timestep, which are
        all put in a queue (docs_queue). Then this one of the events is
        released, while the doc_loader is paused. The execution of the main
        process is restarted. First it unloads the docs from the docs_queue
        and releases the doc_loader again. This process then iterates."""
        if not update_locations:
            print("WARNING: Not updating locations")

        # Check if timestep not bigger than analysis length
        if timestep_length > analysis_length:
            print("Timestep too big")
            sys.exit(0)

        # Set parameters for sharing between processes
        n_docs_to_unload = Counter(0)
        timestep_end_str = mp.Array('c', 26)
        docs_queue = mp.Queue()
        event_1 = mp.Event()
        event_2 = mp.Event()
        is_real_time = mp.Value(c_bool, False)

        end_date_spinup = start + spinup_time
        if geoparsing_start:
            if geoparsing_start < start:
                print("ERROR: Geoparsing start is smaller than start date")
                sys.exit()
            geoparsing_start = int((geoparsing_start - start) / timestep_length) * timestep_length + start
            print("Geoparsing start:", geoparsing_start)
            doc_loader_start = geoparsing_start
        else:
            doc_loader_start = start

        doc_loader_mp = Process(
            target=self.doc_loader.load_docs,
            args=(
                docs_queue,
                n_docs_to_unload,
                doc_loader_start,
                analysis_length,
                timestep_length,
                event_1,
                event_2,
                timestep_end_str,
                is_real_time
            )
        )
        doc_loader_mp.daemon = True
        doc_loader_mp.start()

        if detection and geoparsing_start and geoparsing_start > end_date_spinup:
            self.event_detector = EventDetector(
                self.pg,
                self.es,
                start,
                spinup_time,
                detection_parameters=detection_parameters,
                regions=regions,
                load_detectors=load_detectors,
            )
            self.initial_detection(
                start,
                geoparsing_start
            )
            end_date_spinup = None

        while True and (real_time or not is_real_time.value):
            event_1.wait()
            if doc_loader_mp.exception is not None:
                _, traceback = doc_loader_mp.exception
                print(traceback)
                sys.exit()

            unloaded_docs = []
            for i in range(n_docs_to_unload.value()):
                unloaded_docs.append(docs_queue.get())
                n_docs_to_unload.decrease()

            if self.classify_tweets == 'bert':
                about_ongoing_event_docs = []
                about_ongoing_event_doc_ids = set()
                classified_docs = set()
                
                # Check whether documents are already classified in ES. If so, load classification from ES.
                if unloaded_docs:
                    documents = self.es.mget(index=DOCUMENT_INDEX, body={'ids': [ID for ID, _ in unloaded_docs]})['docs']
                for doc in documents:
                    doc = doc['_source']
                    if 'event_related' in doc:
                        classified_docs.add(doc['id'])
                        if doc['event_related'] is True:
                            about_ongoing_event_doc_ids.add(doc['id'])

                for doc in unloaded_docs:
                    if doc[0] in about_ongoing_event_doc_ids:
                        about_ongoing_event_docs.append(doc)

                docs_to_classify = []
                examples_to_classify = []
                for doc in unloaded_docs:
                    ID, doc_info = doc
                    if ID not in classified_docs:
                        example = {
                            'id': ID,
                            'sentence1': doc_info.clean_text,
                            'label': 0
                        }
                        examples_to_classify.append(example)
                        docs_to_classify.append(doc)

                classes = self.text_classifier(examples_to_classify)

                assert len(classes) == len(docs_to_classify)
                es_update = []
                for doc_class, doc in zip(classes, docs_to_classify):
                    doc_class = True if doc_class == 'yes' else False
                    if doc_class is True:
                        about_ongoing_event_docs.append(doc)
                    es_update.append({
                        'doc': {
                            'event_related': doc_class
                        },
                        '_index': DOCUMENT_INDEX,
                        '_id': doc[0],
                        '_op_type': 'update',
                    })

                self.es.bulk_operation(es_update)

                about_ongoing_event_docs = sorted(
                    about_ongoing_event_docs,
                    key=lambda x: x[1].date,
                    reverse=False
                )
                
                self.docs.update(dict(about_ongoing_event_docs))
            elif self.classify_tweets == 'db':
                # Check whether documents are already classified in ES. If so, load classification from ES.
                about_ongoing_event_docs = []
                if unloaded_docs:
                    documents = self.es.mget(index=DOCUMENT_INDEX, body={'ids': [ID for ID, _ in unloaded_docs]})['docs']
                    for doc in documents:
                        doc = doc['_source']
                        if doc['event_related'] is True:
                            about_ongoing_event_doc_ids.add(doc['id'])

                    for doc in unloaded_docs:
                        if doc[0] in about_ongoing_event_doc_ids:
                            about_ongoing_event_docs.append(doc)
                self.docs.update(dict(about_ongoing_event_docs))
            else:
                self.docs.update(dict(unloaded_docs))
            
            if max_n_docs_in_memory is not None and len(self.docs) > max_n_docs_in_memory:
                n_docs_to_delete = len(self.docs) - max_n_docs_in_memory
                IDs_to_remove = list(self.docs.keys())[:n_docs_to_delete]
                for ID in IDs_to_remove:
                    del self.docs[ID]

            event_1.clear()
            event_2.set()
            near_end_date_spinup = False
            if self.docs:
                timestep_end = str(timestep_end_str.value, 'utf-8')
                timestep_end = isoformat_2_date(timestep_end)
                l_docs = []

                if detection and end_date_spinup and timestep_end >= end_date_spinup:
                    self.event_detector = EventDetector(
                        self.pg,
                        self.es,
                        start,
                        spinup_time,
                        detection_parameters=detection_parameters,
                        load_detectors=load_detectors,
                        regions=regions
                    )
                    self.initial_detection(
                        start,
                        timestep_end - analysis_length
                    )
                    near_end_date_spinup = True

                for ID, doc in self.docs.items():
                    if doc.date > timestep_end - analysis_length:
                        break
                    else:
                        l_docs.append(ID)

                for i, ID in enumerate(l_docs):
                    l_docs[i] = self.docs[ID]
                    del self.docs[ID]

                self.geoparse_timestep(timestep_end, update_locations=update_locations)
                if detection and not end_date_spinup and (
                    not geoparsing_start or
                    timestep_end > geoparsing_start + analysis_length
                ):
                    self.event_detector.detect_events_l(l_docs, is_real_time=is_real_time)
                    self.event_detector.detect_events_s(self.docs.values(), is_real_time=is_real_time)
                if near_end_date_spinup:
                    end_date_spinup = None

                if end and timestep_end > end:
                    return None


def valid_date(s):
    if s == 'now':
        return datetime.utcnow()
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H")
        except ValueError:
            try:
                return datetime.strptime(s, "%Y-%m-%dT%H:%M")
            except ValueError:
                msg = f"Not a valid date: '{s}'."
                raise argparse.ArgumentTypeError(msg)


def main():
    event_detector = Detection(
        doc_loader=args.doc_loader,
        n_words=n_words,
        minimum_gram_length=minimum_gram_length,
        max_distance_entities_doc=args.max_distance_entities_doc,
        classify_tweets=args.classify_tweets,
        doc_score_types=DOC_SCORE_TYPES,
    )

    event_detector.maybe_set_table_name(args.regions, DETECTION_PARAMETERS)

    event_detector.run(
        detection=args.detection,
        start=START_DATE,
        spinup_time=SPINUP_TIME,
        timestep_length=TIMESTEP_LENGTH,
        analysis_length=ANALYSIS_LENGTH,
        geoparsing_start=args.geoparsing_start,
        detection_parameters=DETECTION_PARAMETERS,
        update_locations=args.update_locations,
        load_detectors=args.load_detectors,
        regions=args.regions,
        real_time=args.real_time,
        max_n_docs_in_memory=args.max_n_docs_in_memory
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run-type', default='l')
    parser.add_argument('-g', '--geoparsing-start', type=valid_date, default=False)
    parser.add_argument('-ld', '--load-detectors', type=parse_bool, default=False)
    parser.add_argument('-ul', '--update-locations', type=parse_bool, default=True)
    parser.add_argument('-ct', '--classify-tweets', type=str, default='bert')
    parser.add_argument('-re', '--regions', default='admin')
    parser.add_argument('-de', '--detection', type=parse_bool, default=True)
    parser.add_argument('-rt', '--real-time', type=parse_bool, default=True)
    parser.add_argument('-dl', '--doc_loader', type=str, default=None)
    parser.add_argument('-cp', '--check-previous-documents', type=parse_bool, default=True)
    parser.add_argument('-mde', '--max-distance-entities-doc', type=int, default=200_000)
    parser.add_argument('-mdm', '--max-n-docs-in-memory', type=int, default=None)

    args = parser.parse_args()

    if args.run_type == 'l':
        ANALYSIS_LENGTH = timedelta(hours=12)
        SPINUP_TIME = timedelta(days=365)
        TIMESTEP_LENGTH = timedelta(hours=3)
    elif args.run_type == 's':
        ANALYSIS_LENGTH = timedelta(hours=.3)
        SPINUP_TIME = timedelta(hours=.4)
        TIMESTEP_LENGTH = timedelta(hours=.2)
    else:
        print("Option unknown: choose either l or s for run type")
        sys.exit()

    main()
