from collections import namedtuple, defaultdict
from datetime import timedelta, datetime, date
import math
import numpy as np

from methods.dates import daterange, round_up_to_full_day
from methods.tweets import LastTweetsDeque, LastAuthorDict

EventDetectionData = namedtuple(
    'EventDetectionData',
    [
        'first_doc',
        'latest_doc',
        'childs'
    ]
)

DetectionData = namedtuple(
    'DetectionData',
    [
        'data_point',
        'score',
        'diff',
        'corrected_diff'
    ]
)


class Detector:
    def __init__(
        self,
        name,
        start,
        time_correction,
        setting,
        spinup_time,
        triggers,
        threshold_type="dynamic"
    ):
        """Set class parameters, set initial receiver to startup()."""
        self.name = name
        self.time_correction = time_correction
        self.location_threshold = setting.location_threshold
        self.factor = setting.factor
        self.fraction = .05
        self.base = setting.base + 1
        self.max_limit = 3 * 24 * 3600
        self.start_date_full_day = round_up_to_full_day(start).date()
        self.spinup_time = spinup_time
        self.end_spinup = start + self.spinup_time
        self.threshold_type = threshold_type
        self.triggers = triggers
        self.initialize_spinup()

    def add_to_startup(self, data_point):
        if (
            data_point and
            self.is_above_location_threshold(data_point.locations) and not
            self.last_tweets.is_similar_to(ngrams=data_point.ngrams) and not
            self.last_authors.is_old_author(data_point.author_id, data_point.date)
        ):
            self.n_docs_per_day[data_point.date.date()] += 1

    def initialize_spinup(self):
        self.n_docs_per_day = defaultdict(int)
        self.last_tweets = LastTweetsDeque()
        self.last_authors = LastAuthorDict()

    def calculate_lim(self, from_dt, to_dt, x=30, inf=np.inf, array=np.array, percentile=np.percentile):
        """Calculate the limit."""
        assert isinstance(from_dt, date)
        assert isinstance(to_dt, date)
        if self.n_docs_per_day:
            values = [
                self.n_docs_per_day[day]
                for day
                in daterange(from_dt, to_dt, timedelta(days=1))
            ]
            if values:
                values = array(values)
                self.norm_n_docs_per_day = (percentile(values, x) + percentile(values, 100 - x)) / 2
                if self.norm_n_docs_per_day == 0:
                    limit = inf
                else:
                    normal_gap_between_docs = 24 * 3600 / self.norm_n_docs_per_day
                    limit = normal_gap_between_docs * self.fraction
            else:
                self.norm_n_docs_per_day = 0
                limit = inf
        else:
            self.norm_n_docs_per_day = 0
            limit = inf
        assert limit >= 0
        return limit

    def initialize(self):
        self.last_tweet_date_l = datetime(1970, 1, 1)
        self.date_last_recalc_limit = self.end_spinup.date()

        self.limit = self.calculate_lim(
            self.start_date_full_day,
            self.end_spinup.date()
        )

        self.last_tweets = LastTweetsDeque()
        self.last_authors = LastAuthorDict()
        self.last_data_point_is_event_l = datetime(1970, 1, 1)

        self.local_array_l = []
        self.n_check_dates = self.calculate_flood_score()

    def get_time_corrected_diff(self, diff, doc_date):
        if self.norm_n_docs_per_day > self.n_check_dates:
            corrected_diff = diff \
                * self.get_time_correction_factor(doc_date)
        else:
            corrected_diff = diff
        assert corrected_diff >= 0
        return corrected_diff

    def add_l(self, corrected_diff, date):
        self.n_docs_per_day[date.date()] += 1

    def maybe_recalculate_limit(self):
        last_tweet_date_l_date = self.last_tweet_date_l.date()
        if last_tweet_date_l_date > self.date_last_recalc_limit:
            if self.spinup_time.seconds > 0 or self.spinup_time.microseconds > 0:
                td = timedelta(days=self.spinup_time.days + 1)
            else:
                td = timedelta(days=self.spinup_time.days)
            self.calculate_lim(
                last_tweet_date_l_date - td,
                last_tweet_date_l_date
            )
            self.date_last_recalc_limit = last_tweet_date_l_date

    def _diffs_from_data(self, data, zip=zip):
        return [
            (j[0] - i[0]).total_seconds()
            for i, j
            in zip(data[:-1], data[1:])
        ]

    def _time_correction_list(self, data, diffs):
        return [
            diff * self.get_time_correction_factor(date)
            for (date, _), diff
            in zip(data, diffs)
        ]

    def _get_dummy_diffs(self, data, percentile=np.percentile):
        diffs = self._diffs_from_data(data)
        data = data[1:]
        if self.time_correction:
            '''Need to check if this works properly'''
            diffs = self._time_correction_list(data, diffs)

        if diffs:
            percentile_low = percentile(diffs, 30)
            percentile_high = percentile(diffs, 100)

            return [
                diff
                for diff in diffs
                if percentile_low < diff < percentile_high
            ]
        else:
            return []

    def get_time_correction_factor(self, date, timedelta=timedelta):
        """Read time correction factor for specific hour."""
        if self.time_correction:
            hour = (date + timedelta(minutes=30)).hour
            return self.time_correction[hour] * 24
        else:
            return 1

    def remove_docs_after_or_on(self, date):
        for i, doc_date in enumerate(self.dates[::-1]):
            if doc_date <= date:
                break
        if i > 0:
            try:
                self.corrected_diffs = self.corrected_diffs[:-i]
                self.dates = self.dates[:-i]
            except UnboundLocalError:
                pass

    def remove_docs_before_or_on(self, date):
        for i, doc_date in enumerate(self.dates):
            if doc_date > date:
                break
        try:
            self.corrected_diffs = self.corrected_diffs[i:]
            self.dates = self.dates[i:]
        except UnboundLocalError:
            pass

    def local_array_size_formula(self, limit, e=math.e):
        return self.factor * e ** (
            -.5 / 86400 * limit / self.fraction
        ) + self.base

    def calculate_flood_score(
        self,
        floor=math.floor,
        isinf=np.isinf,
        max_lim=182.5*24*3600
    ):
        """self.limit / self.fraction = the average time between tweets in
            this area. The limit goes to 1
        """
        if isinf(self.limit) or self.limit > max_lim:
            return self.base
        else:
            return floor(self.local_array_size_formula(self.limit))

    def is_above_location_threshold(self, locations):
        for loc in locations:
            # In case of country, always True
            if loc.level_0_region == loc.location_ID:
                return True
            # In case of level 1 region, always True
            elif loc.level_1_region == loc.location_ID:
                return True
            # Otherwise check with location threshold
            else:
                return any(
                    location.score > self.location_threshold for location in locations
                )

    def get_diff_l(self, data_point, append=True):
        if not self.is_above_location_threshold(data_point.locations):
            return None

        if self.last_tweets.is_similar_to(ngrams=data_point.ngrams):
            return None

        if self.last_authors.is_old_author(data_point.author_id, data_point.date):
            return None

        diff = (data_point.date - self.last_tweet_date_l).total_seconds()
        corrected_diff = self.get_time_corrected_diff(diff, data_point.date)

        if (
            append
            and corrected_diff < self.limit * 10
            and self.last_tweet_date_l != datetime(1970, 1, 1)
        ):
            self.last_tweet_date_l = data_point.date
            self.add_l(corrected_diff, data_point.date)
            self.maybe_recalculate_limit()
        else:
            self.last_tweet_date_l = data_point.date

        return diff, corrected_diff

    def get_diff_s(self, data_point):
        if not self.is_above_location_threshold(data_point.locations):
            return None
        if not hasattr(self, 'last_data_point_is_event_s'):
            self.last_data_point_is_event_s = self.last_data_point_is_event_l
        if not hasattr(self, 'last_tweet_date_s'):
            self.last_tweet_date_s = self.last_tweet_date_l
            self.last_tweet_date_s = datetime(1980, 1, 1)
        if not hasattr(self, 'last_tweets_s'):
            self.last_tweets_s = LastTweetsDeque(self.last_tweets)
        if not hasattr(self, 'last_authors_s'):
            self.last_authors_s = LastAuthorDict(self.last_authors)
        if (
            not self.last_tweets_s.is_similar_to(ngrams=data_point.ngrams) and not
            self.last_authors_s.is_old_author(data_point.author_id, data_point.date)
        ):
            diff = (data_point.date - self.last_tweet_date_s).total_seconds()
            corrected_diff = diff * self.get_time_correction_factor(data_point.date)
            self.last_tweet_date_s = data_point.date
            assert corrected_diff >= 0
            return diff, corrected_diff

    def score_data_point(self, data_point):
        return {
            'total': 1,
            'location': set()
        }

    def flag(self, dt, diffs, corrected_diffs, last_data_point_is_event, mean=np.mean, array=np.array, three_days=24*3600):
        if any(diff > three_days for diff in diffs):
            return False
        else:
            limit = min(self.limit, self.max_limit)
            if last_data_point_is_event + timedelta(days=3) > dt:
                return mean(array(corrected_diffs)) < limit * 6
            else:
                return mean(array(corrected_diffs)) < limit


    def is_flooded(self, dt, detection_data_array, last_data_point_is_event, array=np.array, children_types=('adm5', 'adm4', 'adm3', 'adm2', 'town', 'landmark')):
        self.n_check_dates = math.ceil(self.calculate_flood_score())
        detection_data_array = detection_data_array[-self.n_check_dates-1:]

        n = self.n_check_dates
        diffs, corrected_diffs = [], []
        for i, detection_data in enumerate(detection_data_array[::-1]):
            diffs.append(detection_data.diff)
            corrected_diffs.append(detection_data.corrected_diff)
            n -= detection_data.score['total']
            if n <= 0 and self.flag(dt, diffs, corrected_diffs, last_data_point_is_event):
                break
        else:
            return detection_data_array, False

        childs = []
        for detection_data in detection_data_array[:-i-2:-1]:
            for location in detection_data.data_point.locations:
                if location.type in children_types and location.location_ID != location.level_1_region:
                    childs.append(location)

        first_doc_date = detection_data.data_point.date
        for detection_data in detection_data_array[i::-1]:
            if self.flag(dt, [detection_data.diff], [detection_data.corrected_diff], last_data_point_is_event):
                first_doc_date = detection_data.data_point.date
            else:
                break

        return detection_data_array, EventDetectionData(
            first_doc=first_doc_date,
            latest_doc=detection_data_array[-1].data_point.date,
            childs=childs
        )

    def send_l(self, data_point):
        data_point_diffs = self.get_diff_l(data_point)
        if not data_point_diffs:
            return None
        data = DetectionData(
            data_point=data_point,
            score=self.score_data_point(data_point),
            diff=data_point_diffs[0],
            corrected_diff=data_point_diffs[1]
        )
        self.local_array_l.append(data)
        self.local_array_l, flood_data = self.is_flooded(data_point.date, self.local_array_l, self.last_data_point_is_event_l)
        if flood_data:
            self.last_data_point_is_event_l = data_point.date
        return flood_data

    def send_s(self, data_point):
        data_point_diffs = self.get_diff_s(data_point)
        if not data_point_diffs:
            return None
        data = DetectionData(
            data_point=data_point,
            score=self.score_data_point(data_point),
            diff=data_point_diffs[0],
            corrected_diff=data_point_diffs[1]
        )
        if not hasattr(self, 'local_array_s'):
            self.local_array_s = list(self.local_array_l)  # Copy list rather than use =
        self.local_array_s.append(data)
        self.local_array_s, flood_data = self.is_flooded(data_point.date, self.local_array_s, self.last_data_point_is_event_s)
        if flood_data:
            self.last_data_point_is_event_s = data_point.date
        return flood_data

    def reset_s(self):
        if hasattr(self, 'last_data_point_is_event_s'):
            del self.last_data_point_is_event_s
        if hasattr(self, 'last_tweets_s'):
            del self.last_tweets_s
        if hasattr(self, 'last_tweet_date_s'):
            del self.last_tweet_date_s
        if hasattr(self, 'last_authors_s'):
            del self.last_authors_s
        if hasattr(self, 'local_array_s'):
            del self.local_array_s
