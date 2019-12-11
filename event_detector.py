import urllib
import os
import pandas as pd
import multiprocessing as mp
import sys
import json
import dill

from ctypes import c_bool
from collections import namedtuple, defaultdict
from datetime import timedelta
from psycopg2.extensions import AsIs
from classes import Location
from methods.tweets import LastTweetsDeque
from methods.dates import daterange
from detector import Detector

from config import DOCUMENT_INDEX, LEVEL_2_COUNTRIES

TweetScores = namedtuple('doc_scores', 'event_related')
DataPoint = namedtuple('DataPoint', 'date language text clean_text ngrams scores locations author_id')
AnalyzedDocShort = namedtuple("AnalyzedDocShort", "resolved_locations date language text clean_text scores repost author_id")


class EventDetector:
    def __init__(
        self,
        pg,
        es,
        initialize_start,
        spinup_time,
        detection_parameters,
        regions,
        load_detectors=False,
    ):
        self.pg = pg
        self.es = es
        self.region_type = regions

        # (sensitivity, threshold)
        self.detection_parameters = detection_parameters
        self.flood_tracker = {
            setting: {}
            for setting in self.detection_parameters
        }

        self._create_table()
        self.flood_delay = timedelta(hours=24)
        self.detectors = self._initialize_detectors(
            initialize_start,
            spinup_time,
            load_detectors,
        )

    def _create_table(self):
        for setting in self.detection_parameters:
            self.pg.cur.execute("""
                DROP TABLE
                IF EXISTS
                %s
            """, (AsIs(setting.run_name), ))
            self.pg.cur.execute("""
                CREATE TABLE %s (
                    event_id SERIAL PRIMARY KEY,
                    location_ID VARCHAR,
                    detection_time TIMESTAMP,
                    first_doc TIMESTAMP,
                    latest_doc TIMESTAMP,
                    childs jsonb
                )
            """, (AsIs(setting.run_name), ))
            self.pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS %s_location_ID_idx
                ON %s (location_ID)
            """, (AsIs(setting.run_name), AsIs(setting.run_name)))
            self.pg.conn.commit()
        print("Created events tables")

    def update_db(self, is_real_time):
        for setting, setting_data in self.flood_tracker.items():
            values = [
                (
                    ID,
                    flood_data['first_doc'],
                    flood_data['latest_doc']
                ) for ID, flood_data in setting_data.items()
            ]
            if values:
                # Query to update events. Returns events that are not updated.
                # These location_IDs can be used to insert new events in the
                # next query.
                query = f"""
                    WITH upd AS
                    (
                        SELECT DISTINCT ON
                            (
                                {setting.run_name}.location_ID
                            )
                            {setting.run_name}.event_id,
                            amend.location_ID,
                            amend.latest_doc,
                            amend.first_doc
                        FROM {setting.run_name}
                        JOIN
                        (
                            VALUES {{}}
                        ) AS amend(
                            location_ID,
                            first_doc,
                            latest_doc
                        )
                        ON amend.location_ID = {setting.run_name}.location_ID
                        ORDER BY
                            {setting.run_name}.location_ID, {setting.run_name}.latest_doc
                            DESC
                    )
                    UPDATE {setting.run_name}
                    SET latest_doc = GREATEST(
                            upd.latest_doc, {setting.run_name}.latest_doc
                        ),
                        first_doc = LEAST(
                            upd.first_doc, {setting.run_name}.first_doc
                        )
                    FROM upd
                    WHERE upd.location_ID={setting.run_name}.location_ID
                        AND upd.first_doc
                            < {setting.run_name}.latest_doc + interval '3 day'
                        AND upd.event_id = {setting.run_name}.event_id
                    RETURNING upd.location_ID, {setting.run_name}.event_id
                """

                self.pg.do_query(query, "(%s, %s, %s)", values)
                res = self.pg.cur.fetchall()
                updated_events = set(
                    (location_ID, event_id) for location_ID, event_id in res
                )
                if updated_events:
                    self.pg.cur.execute(f"""
                        SELECT
                            event_id,
                            location_ID,
                            childs
                        FROM {setting.run_name}
                        WHERE event_id IN (
                            {','.join(
                                str(event_id) for location_ID, event_id in updated_events
                            )}
                        )
                    """)
                    values = []
                    for event_id, location_ID, childs in self.pg.cur.fetchall():
                        city_location_IDs = set(city['location_ID'] for city in childs)
                        for city in setting_data[location_ID]['childs']:
                            city = {
                                'location_ID': city.location_ID,
                                'centroid': city.coordinates
                            }
                            if city['location_ID'] not in city_location_IDs:
                                childs.append(city)
                                city_location_IDs.add(city['location_ID'])
                        values.append((event_id, json.dumps(childs)))

                    query = f"""
                        UPDATE {setting.run_name}
                        SET childs = c.childs::json
                        FROM (VALUES
                            {{}}
                        ) AS c(event_id, childs)
                        WHERE c.event_id = {setting.run_name}.event_id
                    """
                    self.pg.do_query(query, "(%s, %s)", values)

                new_events = []
                location_IDs_updated_events = set(location_ID for location_ID, event_id in updated_events)
                for location_ID, flood_data in setting_data.items():
                    if location_ID not in location_IDs_updated_events:
                        childs, city_location_IDs = [], set()
                        for city in flood_data['childs']:
                            city = {
                                'location_ID': city.location_ID,
                                'centroid': city.coordinates
                            }
                            if city['location_ID'] not in city_location_IDs:
                                childs.append(city)
                                city_location_IDs.add(city['location_ID'])
                        new_events.append((
                            location_ID,
                            flood_data['latest_doc'],
                            flood_data['first_doc'],
                            flood_data['latest_doc'],
                            json.dumps(childs)
                        ))

                if new_events:
                    query = f"""
                        INSERT INTO {setting.run_name} (
                            location_ID,
                            detection_time,
                            first_doc,
                            latest_doc,
                            childs
                        )
                        VALUES {{}}
                        RETURNING location_ID, {setting.run_name}.event_id
                        """
                    self.pg.do_query(query, "(%s, %s, %s, %s, %s)", new_events)
                    res = self.pg.cur.fetchall()

                self.pg.conn.commit()

            setting_data.clear()

    def _add_detectors(self, detectors, ids, start, spinup_time, time_correction, triggers=None):
        for ID in ids:
            try:
                detector_time_correction = time_correction[ID]
            except KeyError:
                print(f"WARNING: {ID} not found in time_correction")
                detector_time_correction = False
            for setting in self.detection_parameters:
                detectors[setting][ID] = Detector(
                    ID,
                    start=start,
                    setting=setting,
                    time_correction=detector_time_correction,
                    spinup_time=spinup_time,
                    triggers=triggers,
                )

    def _initialize_detectors(self, start, spinup_time, load_detectors):
        detector_folder = os.path.join('save')
        try:
            os.makedirs(detector_folder)
        except OSError:
            pass
        detector_path = os.path.join(detector_folder, "detectors.dill")
        if load_detectors and os.path.exists(detector_path):
            print("Reading detectors from dill")
            with open(detector_path, 'rb') as f:
                return dill.load(f)
        else:
            detectors = {
                setting: {}
                for setting in self.detection_parameters
            }
            with open(os.path.join('input', 'time_correction.json'), 'r') as f:
                time_correction = json.load(f)
                for location_ID, stats in time_correction.items():
                    if stats:
                        time_correction[location_ID] = {
                            int(hour): v
                            for hour, v
                            in stats.items()
                        }

            self.pg.cur.execute("""
                SELECT
                    location_ID
                FROM
                    locations
                WHERE
                    (
                        location_type = 'adm1'
                        AND
                        country_location_ID NOT IN %s
                    )
                OR
                    (
                        location_type = 'adm2'
                        AND
                        country_location_ID IN %s
                    )
                OR
                    location_type = 'country'
            """, (LEVEL_2_COUNTRIES, LEVEL_2_COUNTRIES))
            ids = tuple(location_ID for location_ID, in self.pg.cur.fetchall())
            self._add_detectors(detectors, ids, start, spinup_time, time_correction, triggers=None)

            detectors = self._spinup_detectors(detectors, start, spinup_time)
            print("Dumping detectors to dill")
            with open(detector_path, 'wb') as f:
                dill.dump(detectors, f)
            return detectors

    def update_flood_tracker(self, setting, location_ID, flood_data):
        if location_ID not in self.flood_tracker[setting]:
            self.flood_tracker[setting][location_ID] = {
                'first_doc': flood_data.first_doc,
                'latest_doc': flood_data.latest_doc,
                'detection_time': flood_data.latest_doc,
                'childs': flood_data.childs
            }
        else:
            self.flood_tracker[setting][location_ID]['latest_doc'] = \
                flood_data.latest_doc
            self.flood_tracker[setting][location_ID]['childs'].extend(flood_data.childs)

    def find_datapoints(
        self,
        doc,
        ngramify=LastTweetsDeque().ngramify
    ):
        data_points = defaultdict(list)
        doc_locations = doc.resolved_locations

        # Do not use document if more than one country is mentioned
        n_countries = sum(
            [
                1 for loc
                in doc_locations
                if loc.level_0_region == loc.location_ID
            ]
        )
        if n_countries >= 2:
            return None

        for loc1 in doc_locations:
            if loc1.type in ('landmark', 'adm5', 'adm4', 'adm3', 'adm2', 'town') and loc1.location_ID != loc1.level_1_region and any(
                loc2.level_0_region != loc1.level_0_region
                and loc1.location_ID != loc2.location_ID
                for loc2 in doc_locations
            ):
                return None

        for loc in doc_locations:
            # Do not use landmarks for event detection
            if loc.type == 'landmark':
                continue

            # Check if location has a level 1 region, otherwise, do not use
            elif loc.level_1_region and loc.level_1_region != 99:
                data_points[loc.level_1_region].append(loc)

            # is country
            elif loc.level_0_region == loc.location_ID:
                data_points[loc.location_ID].append(loc)


        return {
            admin_area: DataPoint(
                date=doc.date,
                language=doc.language,
                text=doc.text,
                clean_text=doc.clean_text.lower(),
                ngrams=ngramify(doc.clean_text),
                scores=None,
                locations=locations,
                author_id=doc.author_id
            )
            for admin_area, locations in data_points.items()
        }

    def send_doc_to_detector(self, doc, detectors, run_type):
        """Send doc to detector(s).

        in this function, the names of the detectors relevant for the
        specific doc are collected. If relevant detectors are found the
        doc a data point is made for each setting and detector name
        and the doc is send to the detector. If a non-False or non-None
        value returns, this data is send to the function responsible for
        updating the flood tracker.
        """
        data_points = self.find_datapoints(doc)
        if not data_points:
            return None

        used_detectors = set()
        for setting in self.detection_parameters:
            detectors_setting = detectors[setting]
            for detector_location_ID, data_point in data_points.items():
                try:
                    detector = detectors_setting[detector_location_ID]
                except KeyError:
                    print(f"{detector_location_ID} detector not found", setting)
                else:
                    if run_type == 'spinup':
                        detector.add_to_startup(data_point)
                        used_detectors.add(detector_location_ID)
                    else:
                        # None = duplicate
                        # False = not flooded
                        # FloodData (namedtuple) = flooded
                        if run_type == 'l':
                            flood_data = detector.send_l(data_point)
                        elif run_type == 's':
                            flood_data = detector.send_s(data_point)
                        else:
                            raise ValueError
                        if flood_data:
                            self.update_flood_tracker(
                                setting,
                                detector_location_ID,
                                flood_data
                            )
                        used_detectors.add(detector_location_ID)
        return used_detectors

    def doc_to_namedtuple(self, doc, clean_text=LastTweetsDeque().clean_text):
        try:
            return AnalyzedDocShort(
                resolved_locations=[Location(loc) for loc in doc['locations']],
                date=doc['date'],
                language=doc['source']['lang'],
                text=doc['text'],
                clean_text=clean_text(doc['text']),
                scores=doc['scores'] if 'scores' in doc else None,
                repost=doc['source']['retweet'],
                author_id=doc['source']['author']['id']
            )
        except KeyError:
            print(doc)
            raise

    def _spinup_detectors(self, detectors, start, spinup_time):
        print("spinup detectors")
        for query_start, query_end in daterange(
            start,
            start + spinup_time,
            timedelta(hours=24),
            ranges=True
        ):
            query = self.es.build_date_query(
                query_start,
                min(query_end, start + spinup_time),
                locations=True,
            )
            query['query']['bool']['must'].append({
                    'term': {
                        'event_related': True
                    }
                })
            print(f'{query_start}:', self.es.n_hits(index=DOCUMENT_INDEX, body=query), 'docs')
            docs = self.es.scroll_through(
                index=DOCUMENT_INDEX,
                body=query,
                source=False
            )
            for doc in docs:
                doc = self.doc_to_namedtuple(doc)
                self.maybe_send_doc_to_detector(doc, detectors, 'spinup')

        for detectors_per_setting in detectors.values():
            for detector in detectors_per_setting.values():
                detector.initialize()
        return detectors

    def maybe_send_doc_to_detector(self, doc, detectors, run_type):
        """Send doc to detector if necessary.

        This function unpacks the doc if neccesary and checks if a
        location is available. If so, the doc is passed on to the
        function that actually sends the docs to the detecor
        """
        if doc.resolved_locations and doc.repost is False:
            return self.send_doc_to_detector(
                doc,
                detectors,
                run_type
            )

    def detect_events_l(
        self,
        docs,
        is_real_time=mp.Value(c_bool, False),
        convert_to_named_tuple=False,
    ):
        for doc in docs:
            if convert_to_named_tuple:
                doc = self.doc_to_namedtuple(doc)
            self.maybe_send_doc_to_detector(
                doc,
                self.detectors,
                run_type='l'
            )
        self.update_db(is_real_time=is_real_time)

    def detect_events_s(self, docs, is_real_time=mp.Value(c_bool, False)):
        used_detectors = set()
        for doc in docs:
            data_points = self.maybe_send_doc_to_detector(
                doc,
                self.detectors,
                run_type='s'
            )
            if data_points:
                for location_ID in data_points:
                    used_detectors.add(location_ID)

        for location_ID in used_detectors:
            for setting in self.detection_parameters:
                self.detectors[setting][location_ID].reset_s()
        self.update_db(is_real_time=is_real_time)
