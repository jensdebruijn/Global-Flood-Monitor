import pytz
import numpy as np
from datetime import datetime, timedelta, date
from operator import le, lt

months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


def date_2_datestr(date):
    return date.strftime('%Y_%m_%d')


def datestr_2_date(datestr, datetime=datetime):
    return datetime.strptime(datestr, '%Y_%m_%d')


def isoformat_2_date(datestr, datetime=datetime):
    try:
        return datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        return datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S.%f')


def numpy64_todatetime(np64):
    assert isinstance(np64, np.datetime64)
    return datetime.utcfromtimestamp(np64.astype('O') / 1e9)


def date2datetime(date, datetime=datetime):
    return datetime.combine(date, datetime.min.time())


def round_down_to_full_day(date):
    return date2datetime(date.date())


def round_up_to_full_day(date, timedelta=timedelta):
    return date2datetime((date - timedelta(seconds=1) + timedelta(days=1)).date())


def parse_date_4_weird_datetime(date, datetime=datetime):
    date = date.split(' ')
    month, day, time, year = date[0], int(date[1]), date[2], int(date[4])
    hour, minute, second = [int(x) for x in time.split(':')]
    return datetime(year, months[month], day, hour, minute, second)


def daterange(start_date, end_date, delta, ranges=False, include_last=False, UTC=False, timedelta=timedelta):
    if UTC:
        start_date = start_date.replace(tzinfo=pytz.UTC)
        end_date = end_date.replace(tzinfo=pytz.UTC)
    if not isinstance(delta, timedelta):
        delta = timedelta(seconds=int(delta))
    if include_last:
        sign = le
    else:
        sign = lt
    while sign(start_date, end_date):
        if ranges:
            yield start_date, start_date + delta
        else:
            yield start_date
        start_date += delta


def read_date(date, datetime=datetime):
    return datetime.strptime(date, '%d/%m/%Y %H:%M:%S')


def format(date, date_only=False):
    if not date_only:
        return date.strftime('%d/%m/%Y %H:%M:%S')
    else:
        return date.strftime('%d/%m/%Y')


def file_format(date, date_only=False):
    if not date_only:
        return date.strftime('%d_%m_%Y %H:%M:%S')
    else:
        return date.strftime('%d_%m_%Y')


def date_handler(obj, datetime=datetime, date=date):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    print(obj, type(obj))
    raise TypeError("Type not serializable")
