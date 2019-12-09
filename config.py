from collections import namedtuple
from datetime import datetime

START_DATE = datetime(2014, 7, 30)


PG_DB = 'gfm'
DOCUMENT_INDEX = 'gfm'
POSTGRESQL_HOST = '127.0.0.1'
# POSTGRESQL_HOST = '145.108.190.209'
if POSTGRESQL_HOST != '127.0.0.1':
    print(f"Using {POSTGRESQL_HOST} as PostgreSQL host")
POSTGRESQL_PORT = 5432
POSTGRESQL_USER = 'postgres'
POSTGRESQL_PASSWORD = None

ELASTIC_HOST = '127.0.0.1'
ELASTIC_PORT = 9200
ELASTIC_USER = False
ELASTIC_PASSWORD = False

TWITTER_CONSUMER_KEY = ''
TWITTER_CONSUMER_SECRET = ''
TWITTER_ACCESS_TOKEN = ''
TWITTER_ACCESS_TOKEN_SECRET = ''

CLASSIFIER_OUTPUT_MODE = 'regression'
CLASSIFIER_MAX_LENGTH = 64
CLASSIFIER_BATCH_SIZE = 16
CLASSIFIER_LABELS = ['yes', 'no']

LEVEL_2_COUNTRIES = (
    'g-2635167',  # UK
)

DOC_SCORE_TYPES = {
    'coordinates_match': 2,
    'user_home': 2,
    'bbox': 2,
    'time_zone': .3,
    'family': 3,
    'language_match': .15
}

DetectorSetting = namedtuple('setting', [
    'run_name',
    'location_threshold',
    'factor',
    'fraction',
    'base',
    'doc_scores',
    'location_scores',
])


DETECTION_PARAMETERS = [
    DetectorSetting(
        run_name="floods_sensitive",
        location_threshold=.2,
        factor=7,
        fraction=.3,
        base=1,
        doc_scores=False,
        location_scores=False,
    ),
    DetectorSetting(
        run_name="floods_balanced",
        location_threshold=.2,
        factor=7,
        fraction=.3,
        base=2,
        doc_scores=False,
        location_scores=False,
    ),
    DetectorSetting(
        run_name="floods_strict",
        location_threshold=.2,
        factor=7,
        fraction=.3,
        base=3,
        doc_scores=False,
        location_scores=False,
    ),
]
