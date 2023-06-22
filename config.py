from collections import namedtuple
from datetime import datetime

START_DATE = datetime(2014, 7, 30)


PG_DB = 'gfm'
POSTGRESQL_HOST = '127.0.0.1'
POSTGRESQL_PORT = 5432
POSTGRESQL_USER = 'postgres'
POSTGRESQL_PASSWORD = 'ikbenhet'

DOCUMENT_INDEX = 'gfm'
ELASTIC_HOST = '127.0.0.1'
ELASTIC_PORT = 9200
ELASTIC_USER = 'elastic'
ELASTIC_PASSWORD = 'GgG2Mk25DPrvzkWrQxZq'

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
    'base'
])


DETECTION_PARAMETERS = [
    DetectorSetting(
        run_name="floods_sensitive",
        location_threshold=.2,
        factor=7,
        fraction=0.05,
        base=1
    ),
    DetectorSetting(
        run_name="floods_balanced",
        location_threshold=.2,
        factor=7,
        fraction=0.05,
        base=2,
    ),
    DetectorSetting(
        run_name="floods_strict",
        location_threshold=.2,
        factor=7,
        fraction=0.05,
        base=3,
    ),
]
