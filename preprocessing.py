import urllib
from zipfile import ZipFile
import geopandas as gpd
import subprocess
from datetime import datetime
import json
from methods.sampler import MapToPixel
from osgeo import gdal
import numpy as np
import geopandas as gpd
import time
import requests
import os
import csv
import psycopg2
from psycopg2.extensions import AsIs
import pandas as pd
from db.postgresql import PostgreSQL
from db.elastic import Elastic
from config import LEVEL_2_COUNTRIES, PG_DB, DOCUMENT_INDEX, POSTGRESQL_USER


pd.options.mode.chained_assignment = None

TOWN_CODES = set([
    'PPL',
    'PPLA',
    'PPLA2',
    'PPLA3',
    'PPLA4',
    'PPLC',
    'PPLG',
    'PPLR',
    'PPLS',
    'PPLX',
    'STLMT'
])

pg = PostgreSQL('gfm')
es = Elastic()


class Preprocess():
    def __init__(self):
        self.level_0_codes = self._load_level_0_codes()

    def _load_level_0_codes(self):
        gdf = gpd.GeoDataFrame.from_file(os.path.join('input', 'maps', 'level0.json'))
        return set(['g-' + geonameid for geonameid in gdf['geoNameId']])

    def get_location_type(
        self,
        country_code,
        feature_code,
        location_ID,
        town_codes=TOWN_CODES,
        landmark_codes=set(['SCH', 'PIER'])
    ):
        if isinstance(feature_code, float):
            return None
        assert isinstance(feature_code, str)
        assert isinstance(location_ID, str)

        if location_ID in self.level_0_codes:
            return 'country'
        if (feature_code.startswith('ADM') and len(feature_code) == 4 and feature_code[3].isdigit()):
            return feature_code[:4].lower()
        if feature_code in town_codes:
            return 'town'
        if feature_code == 'CONT':
            return 'continent'
        if feature_code in landmark_codes:
            return 'landmark'
        return None

    def find_hierarchy(self):
        if not pg.column_exists('locations', 'adm1_location_id'):
            print("Finding hierarchy")
            pg.cur.execute("""
                ALTER TABLE locations
                ADD COLUMN country_location_ID VARCHAR,
                ADD COLUMN adm1_location_ID VARCHAR,
                ADD COLUMN adm2_location_ID VARCHAR,
                ADD COLUMN adm3_location_ID VARCHAR,
                ADD COLUMN adm4_location_ID VARCHAR
            """)

            pg.cur.execute("""
                SELECT location_ID, _country_code, _adm1_code, _adm2_code, _adm3_code, _adm4_code
                FROM locations
            """)
            for location_ID, _country_code, _adm1_code, _adm2_code, _adm3_code, _adm4_code in pg.cur.fetchall():
                sql = """
                    UPDATE locations
                    SET
                        country_location_ID = (
                            SELECT
                                location_ID
                            FROM locations
                            WHERE _feature_code IN ('PCL', 'PCLD', 'PCLF', 'PCLS', 'PCLI')
                            AND _country_code = %s
                        ),
                        adm1_location_ID = (
                            SELECT
                                location_ID
                            FROM locations
                            WHERE _feature_code = 'ADM1'
                            AND _country_code = %s
                            AND _adm1_code = %s
                        ),
                        adm2_location_ID = (
                            SELECT
                                location_ID
                            FROM locations
                            WHERE _feature_code = 'ADM2'
                            AND _country_code = %s
                            AND _adm1_code = %s
                            AND _adm2_code = %s
                        ),
                        adm3_location_ID = (
                            SELECT
                                location_ID
                            FROM locations
                            WHERE _feature_code = 'ADM3'
                            AND _country_code = %s
                            AND _adm1_code = %s
                            AND _adm2_code = %s
                            AND _adm3_code = %s
                        ),
                        adm4_location_ID = (
                            SELECT
                                location_ID
                            FROM locations
                            WHERE _feature_code = 'ADM4'
                            AND _country_code = %s
                            AND _adm1_code = %s
                            AND _adm2_code = %s
                            AND _adm3_code = %s
                            AND _adm4_code = %s
                        )
                    WHERE location_ID = %s
                """
                try:
                    pg.cur.execute(sql, (
                        _country_code,
                        _country_code,
                        _adm1_code,
                        _country_code,
                        _adm1_code,
                        _adm2_code,
                        _country_code,
                        _adm1_code,
                        _adm2_code,
                        _adm3_code,
                        _country_code,
                        _adm1_code,
                        _adm2_code,
                        _adm3_code,
                        _adm4_code,
                        location_ID))
                except psycopg2.ProgrammingError:
                    print(location_ID)
                    raise
            pg.cur.execute("""
                UPDATE locations
                SET adm1_location_ID = %s
                WHERE adm1_location_ID = %s
            """, ("g-389469", "g-11777514"))
            pg.conn.commit()

    def index_toponyms(self):
        """This function gets all toponyms from the locations and alternative names table, collects
        all some data from these databases and indexes all data to elasticsearch ready for querying"""

        def get_toponyms(toponyms):
            # Check if the index exists. If it does not exist, the database is emtpy and we need to
            # index all toponyms. Otherwise we first query the database to find all toponyms
            # already indexed and only index the toponyms not yet indexed.

            n_toponyms = len(toponyms)
            for i, name in enumerate(toponyms, start=1):

                if i % 100 == 0:
                    print("Indexing toponyms ({}/{})".format(i, n_toponyms), end="\r")

                pg.cur.execute("""
                    SELECT location_ID
                    FROM locations
                    WHERE name = %s
                """, (name, ))
                location_IDs = [location_ID for location_ID, in pg.cur.fetchall()]

                languages = {
                    location_ID: ['general']
                    for location_ID in location_IDs
                }

                pg.cur.execute("""
                    SELECT alternate_names.location_ID, location_type, isolanguage, alternate_names.full_name
                    FROM alternate_names
                    JOIN locations
                    ON alternate_names.location_ID = locations.location_ID
                    WHERE alternate_name = %s
                """, (name, ))
                abbreviations = {}
                for location_ID, location_type, isolanguage, alternate_name in pg.cur.fetchall():
                    if not isolanguage:
                        if location_type == 'landmark':
                            continue
                        else:
                            pg.cur.execute("""
                                SELECT name FROM locations WHERE location_ID = %s
                            """, (location_ID, ))
                            original_name = pg.cur.fetchone()[0]
                            if " " + alternate_name.lower() + " " not in " " + original_name + " ":
                                continue
                            else:
                                isolanguage = 'partial'
                    if location_ID not in languages:
                        languages[location_ID] = [isolanguage]
                    else:
                        languages[location_ID].append(isolanguage)
                    if isolanguage == 'abbr':
                        if location_ID not in abbreviations:
                            abbreviations[location_ID] = []
                        abbreviations[location_ID].append(alternate_name)

                if languages:
                    pg.cur.execute("""
                        SELECT
                            location_ID,
                            ST_X(ST_Centroid(geom)),
                            ST_Y(ST_Centroid(geom)),
                            location_type,
                            country_location_ID,
                            adm1_location_ID,
                            adm2_location_ID,
                            adm3_location_ID,
                            adm4_location_ID,
                            (
                                SELECT COUNT(*)
                                FROM alternate_names
                                WHERE locations.location_ID = alternate_names.location_ID
                            ) AS translations
                        FROM locations
                        WHERE location_ID IN %s
                    """, (tuple(location_ID for location_ID in languages), ))

                    locations = []
                    for (
                        location_ID,
                        longitude,
                        latitude,
                        location_type,
                        country_location_ID,
                        adm1_location_ID,
                        adm2_location_ID,
                        adm3_location_ID,
                        adm4_location_ID,
                        translations
                    ) in pg.cur.fetchall():
                        if languages[location_ID] is not None:

                            if country_location_ID in LEVEL_2_COUNTRIES:
                                level_1_region = adm2_location_ID
                            else:
                                level_1_region = adm1_location_ID
                            location = {
                                'location_ID': location_ID,
                                'languages': languages[location_ID],
                                'coordinates': (longitude, latitude),
                                'country_location_ID': country_location_ID if country_location_ID else None,
                                'adm1_location_ID': adm1_location_ID if adm1_location_ID else None,
                                'adm2_location_ID': adm2_location_ID if adm2_location_ID else None,
                                'adm3_location_ID': adm3_location_ID if adm3_location_ID else None,
                                'adm4_location_ID': adm4_location_ID if adm4_location_ID else None,
                                'level_0_region': country_location_ID if country_location_ID else None,
                                'level_1_region': level_1_region if level_1_region else None,
                                'translations': translations,
                                'abbreviations': abbreviations[location_ID] if location_ID in abbreviations else None,
                                'type': location_type,
                            }
                            locations.append(location)
                    if locations:
                        body = {
                            'locations': locations,
                            '_index': "locations",
                            '_id': name,
                            '_op_type': 'index',
                        }
                        yield body

            # Print the final number ones more without \r to that it is not overwritten by the next print.
            try:
                print(f"Indexing toponyms ({n_toponyms}/{n_toponyms})")
            except UnboundLocalError:
                pass

        # # Retrieve all distinct names from the locations and alternative names table
        print("indexing toponyms", end='\r')
        pg.cur.execute("""
            SELECT DISTINCT name
            FROM
            (
                SELECT name FROM locations
                UNION ALL
                SELECT alternate_name FROM alternate_names
            ) AS x
            WHERE name IS NOT NULL
        """)
        toponyms = tuple(name for name, in pg.cur.fetchall())
        es.bulk_operation(get_toponyms(toponyms))

    def get_geonames(self, fn, ext, remove=False):
        """This function downloads data from the geonames website and unzips if
        neccesary. For more info see: http://download.geonames.org/export/dump/readme.txt"""
        folder = os.path.join('input', 'geonames')
        try:
            os.makedirs(folder)
        except OSError:
            pass
        fp_wo_ext = os.path.join(folder, fn)
        if not os.path.exists(fp_wo_ext + '.txt'):
            print('Downloading {}'.format(fn))
            url = 'http://download.geonames.org/export/dump/{}'.format(fn + '.' + ext)
            print(f"\t{url}")
            file_path = fp_wo_ext + '.' + ext
            urllib.request.urlretrieve(url, file_path)
            print("Finished downloading")
            if ext == 'zip':
                with ZipFile(file_path, 'r') as zf:
                    zf.extractall(folder)

    def parse_geonames_table(self, file_path, column_names, columns_out, dtypes=None, nrows=None, skiprows=0):
        """This function parses a geonames table and retuns it as a pandas dataframe"""
        file_path = os.path.join('input', 'geonames', file_path)
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=column_names,
            dtype=dtypes,
            engine='c',
            keep_default_na=False,
            na_values=["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "N/A", "NULL", "NaN", "nan"],
            quoting=csv.QUOTE_NONE,
            usecols=columns_out,
            nrows=nrows,
            skiprows=skiprows
        )
        return df[columns_out]

    def create_continent_table(self):
        """This function reads the continents shapefile and commits it to PostgreSQL"""
        if not pg.table_exists('continents'):
            print("Creating continents table")
            pg.cur.execute(f"""CREATE TABLE continents (
                location_ID VARCHAR PRIMARY KEY,
                geom GEOMETRY(Geometry, 4326)
            )""")

            local_folder = os.path.join('input', 'maps', 'continents')
            try:
                os.makedirs(local_folder)
            except OSError:
                pass
            local_zip_file = os.path.join(local_folder, 'continents.zip')
            if not os.path.exists(local_zip_file):
                remote_file = 'https://pubs.usgs.gov/of/2006/1187/basemaps/continents/continents.zip'
                urllib.request.urlretrieve(remote_file, local_zip_file)
            
            shapefile = os.path.join(local_folder, 'continent.shp')
            if not os.path.exists(shapefile):
                with ZipFile(local_zip_file, 'r') as z:
                    z.extractall(local_folder)      

            continents = gpd.GeoDataFrame.from_file(shapefile)
            
            continents['ID'] = continents['CONTINENT'].map({
                'Asia': 'g-6255147',
                'North America': 'g-6255149',
                'Europe': 'g-6255148',
                'Africa': 'g-6255146',
                'South America': 'g-6255150',
                'Oceania': 'g-6255151',
                'Australia': 'g-6255151',
                'Antarctica': 'g-6255152'
            })
            continents = continents.drop('CONTINENT', axis=1)
            continents = continents.dissolve(by='ID')
            for ID, row in continents.iterrows():
                pg.cur.execute("""
                    INSERT INTO continents (location_ID, geom)
                    VALUES (%s, ST_GeometryFromText(%s, 4326))
                """, (ID, row['geometry'].wkt))
            pg.conn.commit()

    def create_country_table(self):
        """This function creates the country table, parsing most data from the locations table. In additon
        it finds the country outlines and income groups"""
        if not pg.table_exists('countries'):
            print("Creating country table")
            pg.cur.execute(f"""
                CREATE TABLE countries (
                    location_ID VARCHAR PRIMARY KEY,
                    ISO2 VARCHAR(2),
                    ISO3 VARCHAR(3),
                    continents VARCHAR[],
                    languages VARCHAR(300),
                    name VARCHAR(200)
                )
            """)

            features = self.get_geoname_table('countryInfo', 'txt', ['ISO2', 'ISO3', 'ISO-Numeric', 'FIPS', 'Country', 'Capital', 'Area', 'Population', 'Continent', 'tld', 'pg.currencyCode', 'pg.currencyName', 'Phone', 'Postal Code Format', 'Postal Code Regex', 'Languages', 'location_ID', 'neighbours', 'EquivalentFipsCode'], ['location_ID', 'ISO2', 'ISO3', 'Continent', 'Languages', 'Country'], skiprows=51)
            features['Country'] = features.Country.str.lower()
            features['location_ID'] = 'g-' + features['location_ID'].astype('str')

            continent_code2location_ID = {
                'EU': 'g-6255148',
                'AS': 'g-6255147',
                'AF': 'g-6255146',
                'NA': 'g-6255149',
                'OC': 'g-6255151',
                'SA': 'g-6255150',
                'AN': 'g-6255152'
            }

            special_continent_cases = {
                'g-2017370': ['g-6255148', 'g-6255147'],
                'g-1643084': ['g-6255147', 'g-6255151'],
                'g-1522867': ['g-6255147', 'g-6255148'],
                'g-587116': ['g-6255147', 'g-6255148'],
                'g-614540': ['g-6255147', 'g-6255148'],
                'g-298795': ['g-6255147', 'g-6255148'],
                'g-357994': ['g-6255147', 'g-6255146'],
                'g-3703430': ['g-6255149', 'g-6255150'],
            }

            def convert_to_location_ID(row):
                if row['location_ID'] in special_continent_cases:
                    return special_continent_cases[row['location_ID']]
                else:
                    return [continent_code2location_ID[row['Continent']]]

            features['Continent'] = features.apply(lambda row: convert_to_location_ID(row), axis=1)

            pg.execute_values(
                "INSERT INTO countries (location_ID, ISO2, ISO3, continents, languages, name) VALUES %s",
                [value for _, value in features.iterrows()],
                template="(%s, %s, %s, %s, %s, %s)"
            )

            pg.conn.commit()

    def get_geoname_table(self, file_path, ext, columns_in, columns_out, skiprows=0):
        self.get_geonames(file_path, ext)
        dtypes = {
            'location_ID': object,
            'name': object,
            'asciiname': object,
            'alternatenames': object,
            'latitude': object,
            'longitude': object,
            'feature_class': object,
            'feature_code': object,
            'country code': object,
            'cc2': object,
            'admin1_code': object,
            'admin2_code': object,
            'admin3_code': object,
            'admin4_code': object,
            'population': float,
            'elevation': float,
            'dem': float,
            'time_zone': object,
        }
        return self.parse_geonames_table(f'{file_path}.txt', column_names=columns_in, columns_out=columns_out, dtypes=dtypes, skiprows=skiprows)

    def create_locations_table(self):
        # pg.cur.execute("""DROP TABLE IF EXISTS locations""")
        if not pg.table_exists('locations'):
            print("Creating locations table")
            pg.cur.execute(f"""
                CREATE TABLE locations (
                    location_ID VARCHAR PRIMARY KEY NOT NULL,
                    name VARCHAR(200),
                    full_name VARCHAR(200),
                    location_type VARCHAR(10) NOT NULL,
                    population BIGINT,
                    geom GEOMETRY(Geometry, 4326),
                    _country_code VARCHAR(2),
                    _adm1_code VARCHAR(20),
                    _adm2_code VARCHAR(80),
                    _adm3_code VARCHAR(20),
                    _adm4_code VARCHAR(20),
                    _feature_code VARCHAR(10)
                )
            """)

            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS locations_names_idx
                ON locations (name)
            """)

            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS locations_country_code
                ON locations (_country_code)
            """)
            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS feature_code_idx
                ON locations (_feature_code)
            """)
            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS locations_adm1_idx
                ON locations (_adm1_code)
            """)
            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS locations_adm2_idx
                ON locations (_adm2_code)
            """)
            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS locations_adm3_idx
                ON locations (_adm3_code)
            """)
            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS locations_adm4_idx
                ON locations (_adm4_code)
            """)

            pg.cur.execute("""
                CREATE INDEX
                IF NOT EXISTS geom_idx
                ON locations
                USING GIST (geom)
            """)

            pg.conn.commit()

    def fill_locations_table_geonames(self):
        pg.cur.execute("""
            SELECT COUNT(*) FROM locations WHERE substring(location_ID, 0, 3) = 'g-'
        """)
        if pg.cur.fetchone()[0] == 0:
            print('filling locations table geonames')

            columns_in = [
                'location_ID',
                'name',
                'asciiname',
                'alternatenames',
                'latitude',
                'longitude',
                'feature_class',
                'feature_code',
                'country_code',
                'cc2',
                'admin1_code',
                'admin2_code',
                'admin3_code',
                'admin4_code',
                'population',
                'elevation',
                'dem',
                'time_zone',
                'modification_date'
            ]
            columns_out = [
                'location_ID',
                'name',
                'feature_code',
                'admin1_code',
                'admin2_code',
                'admin3_code',
                'admin4_code',
                'country_code',
                'population',
                'longitude',
                'latitude'
            ]

            self.get_geonames('allCountries', 'zip')
            print("parsing")
            dtypes = {
                'location_ID': str,
                'name': object,
                'asciiname': object,
                'alternatenames': object,
                'latitude': object,
                'longitude': object,
                'feature_class': object,
                'feature_code': object,
                'country_code': object,
                'cc2': object,
                'admin1_code': object,
                'admin2_code': object,
                'admin3_code': object,
                'admin4_code': object,
                'population': float,
                'elevation': float,
                'dem': float,
                'time_zone': object,
            }

            geonames_features = self.parse_geonames_table(
                'allCountries.txt',
                column_names=columns_in,
                columns_out=columns_out,
                dtypes=dtypes,
            )

            geonames_features['location_ID'] = 'g-' + geonames_features['location_ID']

            habitated_islands = {}
            for _, row in geonames_features[(geonames_features['feature_code'] == 'ISL') & (geonames_features['population'] > 0)].iterrows():
                habitated_islands[row['name'].lower()] = row['country_code']

            ds = gdal.Open(os.path.join('input', 'maps', 'population.tif'))
            band = ds.GetRasterBand(1)
            population_map = band.ReadAsArray()
            gt = ds.GetGeoTransform()

            def is_valid(row):
                if row['name'].lower() in habitated_islands and row['country_code'] == habitated_islands[row['name'].lower()] and row['location_type'] not in ('country', 'adm1'):
                    return False
                if row['location_type'] == 'landmark':
                    return True
                if row['is_valid'] is True:
                    return True
                if row['location_type'] is None:
                    return False
                if row['location_type'] != 'town':
                    return True
                else:
                    x, y = MapToPixel(row['longitude'], row['latitude'], gt)
                    x, y = int(x + .5), int(y + .5)
                    try:
                        population = np.sum(population_map[y-1:y+2, x-1:x+2])
                    except IndexError:
                        population = 0
                    return population > 5000

            geonames_features['location_type'] = geonames_features.apply(lambda row: self.get_location_type(row['country_code'], row['feature_code'], row['location_ID']), axis=1)
            geonames_features = geonames_features[~geonames_features['location_type'].isnull()]

            geonames_features['is_valid'] = geonames_features['population'] > 5000

            geonames_features['longitude'] = pd.to_numeric(geonames_features['longitude'])
            geonames_features['latitude'] = pd.to_numeric(geonames_features['latitude'])

            geonames_features['is_valid'] = geonames_features.apply(is_valid, axis=1)

            geonames_features = geonames_features[geonames_features['is_valid'] == True]

            geonames_features['full_name'] = geonames_features['name']
            geonames_features['name'] = geonames_features.name.str.lower()

            def load_region_geoms():
                region_geoms = {}
                for level in (0, 1):
                    gdf = gpd.GeoDataFrame.from_file(os.path.join('input', 'maps', f'level{level}.json')).set_index("geoNameId")
                    for ID, row in gdf.iterrows():
                        region_geoms['g-' + ID] = row['geometry'].wkt
                return region_geoms

            region_geoms = load_region_geoms()

            def create_geom(row, region_geoms):
                try:
                    return region_geoms[row['location_ID']]
                except KeyError:
                    return f"POINT ( {row['longitude']} {row['latitude']} )"

            geonames_features['geom'] = geonames_features.apply(lambda row: create_geom(row, region_geoms), axis=1)
            geonames_features = geonames_features.drop(['is_valid', 'longitude', 'latitude'], axis=1)

            print("moving locations to DB")
            pg.execute_values(
                """
                INSERT INTO locations (location_ID, name, _feature_code, _adm1_code, _adm2_code, _adm3_code, _adm4_code, _country_code, population, location_type, full_name, geom)
                VALUES %s
                """,
                [value for i, value in geonames_features.iterrows()],
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))",
                page_size=1000
            )
            
            pg.conn.commit()

    def create_alternate_names_table(self):
        # Only execute if table does not exist
        if not pg.table_exists('alternate_names'):
            print("Creating alternate_names table")
            # Create table
            pg.cur.execute("""
                CREATE TABLE alternate_names (
                    location_ID VARCHAR,
                    isolanguage VARCHAR,
                    alternate_name VARCHAR(400),
                    full_name VARCHAR(400)
                )
            """)

            # Create index for faster query. We need that later
            pg.cur.execute("""CREATE INDEX
                IF NOT EXISTS alternate_names_names
                ON alternate_names (alternate_name)""")
            pg.cur.execute("""CREATE INDEX
                IF NOT EXISTS alternative_names_location_IDs
                ON alternate_names (location_ID)""")
            pg.conn.commit()

    def fill_alternate_names_table_geonames(self):
            pg.cur.execute("""
                SELECT COUNT(*) FROM alternate_names WHERE substring(location_ID, 0, 3) = 'g-'
            """)
            if pg.cur.fetchone()[0] == 0:
                print('filling alternative names table geonames')
                self.get_geonames('alternateNames', 'zip')
                columns_in = ['alternateNameId', 'location_ID', 'isolanguage', 'alternate_name', 'isPreferredName', 'isShortName', 'isColloquial', 'isHistoric']
                columns_out = ['location_ID', 'isolanguage', 'alternate_name']
                dtypes = {
                    'alternateNames': object,
                    'location_ID': int,
                    'isolanguage': object,
                    'alternate_name': object,
                    'isPreferredName': object,
                    'isShortName': object,
                    'isColloquial': object,
                    'isHistoric': object,
                }
                alternate_names = self.parse_geonames_table('alternateNames.txt', column_names=columns_in, columns_out=columns_out, dtypes=dtypes)
                # Use only rows with location_IDs in locations table. Others we don't need
                alternate_names['location_ID'] = 'g-' + alternate_names['location_ID'].astype('str')
                pg.cur.execute("""
                    SELECT location_ID
                    FROM locations
                """)
                location_IDs = set(id for id, in pg.cur.fetchall())

                alternate_names = alternate_names[~alternate_names['isolanguage'].isin(['link', 'post'])]
                alternate_names = alternate_names[alternate_names['location_ID'].isin(location_IDs)]

                # Set name to lowercase
                alternate_names['full_name'] = alternate_names['alternate_name']
                alternate_names['alternate_name'] = alternate_names['alternate_name'].str.lower()

                # Commit to database
                pg.execute_values("""
                    INSERT INTO alternate_names (
                        location_ID,
                        isolanguage,
                        alternate_name,
                        full_name
                    ) VALUES %s""",
                    [value for i, value in alternate_names.iterrows()],
                )
                pg.conn.commit()

    def get_most_common_words(self):
        if not pg.table_exists('most_common_words'):
            print("Creating most common words table")
            pg.cur.execute("""
                CREATE TABLE most_common_words (
                n SMALLINT,
                language VARCHAR(2),
                word VARCHAR(30)
            )""")

            languages = ['en', 'id', 'tl', 'fr', 'de', 'it', 'nl', 'pl', 'sr', 'pt', 'es', 'tr', 'sw']

            words = []
            for language in languages:
                with open(os.path.join('input', 'word_frequencies', f'words_{language}.txt'), 'rb') as f:
                    for i, line in enumerate(f.readlines(), start=1):
                        word = line.decode().split(' ')[0].strip()
                        words.append((i, language, word))
                        if i == 10000:
                            break

            pg.execute_values(
                "INSERT INTO most_common_words (n, language, word) VALUES %s",
                words,
                page_size=10_000
            )
            pg.conn.commit()

    def assign_region_levels(self):
        if not pg.column_exists('locations', 'region_level'):
            print("Assigning region levels")
            pg.cur.execute("""
                ALTER TABLE locations
                ADD COLUMN region_level INT
            """)
            pg.cur.execute("""
                UPDATE locations
                SET region_level = 0
                WHERE location_type = 'country'
            """)
            pg.cur.execute("""
                UPDATE locations
                SET region_level = 1
                WHERE (
                    location_type = 'adm1'
                    AND
                    country_location_ID NOT IN %s
                ) OR (
                    location_type = 'adm2'
                    AND
                    country_location_ID IN %s
                )
            """, (LEVEL_2_COUNTRIES, LEVEL_2_COUNTRIES))
            pg.conn.commit()

    def create_simple_geoms(self):
        if not pg.table_exists('simple_geoms'):
            print("Creating simple_geoms table")
            pg.cur.execute(f"""
                CREATE TABLE simple_geoms (
                    location_ID VARCHAR PRIMARY KEY,
                    geom VARCHAR,
                    level INT
                )
            """)

            pg.cur.execute("""
                INSERT INTO simple_geoms (location_ID, geom, level)
                SELECT location_ID, ST_AsGeoJSON(geom), region_level
                FROM locations
                WHERE region_level IN (0, 1)
            """)

            pg.conn.commit()


def run():
    p = Preprocess()
    p.create_continent_table()
    p.create_country_table()
    p.get_most_common_words()
    p.create_locations_table()
    p.create_alternate_names_table()
    p.fill_locations_table_geonames()
    p.fill_alternate_names_table_geonames()
    p.find_hierarchy()
    p.assign_region_levels()
    p.create_simple_geoms()
    p.index_toponyms()


if __name__ == '__main__':
    run()
