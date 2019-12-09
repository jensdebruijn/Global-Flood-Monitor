import re
import numpy as np
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.extras import execute_values

from config import (
    POSTGRESQL_USER,
    POSTGRESQL_PASSWORD,
    POSTGRESQL_HOST,
    POSTGRESQL_PORT
)


class PostgreSQL:
    def __init__(self, db, host=None, user=POSTGRESQL_USER, password=POSTGRESQL_PASSWORD):
        self.host = host or POSTGRESQL_HOST
        self._connect(POSTGRESQL_PORT, user, password, db)
        self.set_pg_init()
        self.initialize_postgis()
        self.initialize_intarray()
        self.create_aggregates()

    def set_pg_init(self):
        self.cur.execute("""SET work_mem TO '256MB';""")
        self.cur.execute("""SET max_parallel_workers_per_gather TO 4;""")
        # self.cur.execute("""SET max_worker_processes TO 4;""")

    def table_exists(self, table):
        self.cur.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            )""", (table, ))
        return self.cur.fetchone()[0]

    def column_exists(self, table, column):
        self.cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = %s
                AND column_name = %s
            )
        """, (table, column))
        return self.cur.fetchone()[0]

    def commit_chunk(self, query, mogr, items, size=1000, commit=True):
        print('rewrite this using execute values')
        exit()

    def execute_values(self, *args, **kwargs):
        execute_values(self.cur, *args, **kwargs)

    def do_query(self, query, mogr, items, commit=True):
        arg_str = ','.join(self.cur.mogrify(mogr, value).decode() for value in items)
        query_str = query.format(arg_str)
        try:
            self.cur.execute(query_str)
        except psycopg2.InternalError:
            print(query_str)
            raise
        if commit:
            self.conn.commit()

    def _connect(self, port, user, password, db):
        try:
            if password:
                connect_str = f"user='{user}' host='{self.host}' port='{port}' password='{password}'"
            else:
                connect_str = f"user='{user}' host='{self.host}' port='{port}'"
            try:
                conn = psycopg2.connect(f"dbname='{db}' " + connect_str)
            except psycopg2.OperationalError:
                conn = psycopg2.connect("dbname='postgres' " + connect_str)
                conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
                cur = conn.cursor()
                cur.execute("CREATE DATABASE {}".format(db))
                conn.close()
                conn = psycopg2.connect("dbname='{}' ".format(db) + connect_str)
            cur = conn.cursor()
            register_adapter(float, self._nan_to_null)
            self._register_numpy_types()
            self.conn = conn
            self.cur = cur

        except Exception as e:
            raise

    def _register_numpy_types(self):
        """Register the AsIs adapter for following types from numpy:
          - numpy.int8
          - numpy.int16
          - numpy.int32
          - numpy.int64
          - numpy.float16
          - numpy.float32
          - numpy.float64
        """
        for typ in ['int8', 'int16', 'int32', 'int64',
                    'float16', 'float32', 'float64']:
            register_adapter(np.__getattribute__(typ), AsIs)

    def _nan_to_null(self, f,
                     _NULL=psycopg2.extensions.AsIs('NULL'),
                     _Float=psycopg2.extensions.Float):
        if not np.isnan(f):
            return _Float(f)
        return _NULL

    def initialize_postgis(self):
        self.cur.execute('CREATE EXTENSION IF NOT EXISTS postgis')
        self.conn.commit()
        # try:
        #     self.cur.execute('CREATE EXTENSION IF NOT EXISTS postgis_sfcgal')
        # except psycopg2.OperationalError:
        #     print('Could not load sfcgal')
        # self.conn.commit()

    def initialize_intarray(self):
        self.cur.execute('CREATE EXTENSION IF NOT EXISTS intarray')
        self.conn.commit()

    def create_aggregates(self):
        self.cur.execute("""DROP AGGREGATE IF EXISTS array_accum(anyelement)""")
        self.cur.execute("""
            CREATE AGGREGATE array_accum(anyelement)
            (
                sfunc = array_append,
                stype = anyarray,
                initcond = '{}'
            )
        """)
        self.conn.commit()

    def get_columns(self, table):
        self.cur.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = '{table}'""")
        return tuple(c for c, in self.cur.fetchall())

    def coordinates_from_Box2D(self, bbox):
        bbox = re.findall(
            "BOX\((-{0,1}\d+.\d+) (-{0,1}\d+.\d+),(-{0,1}\d+.\d+) (-{0,1}\d+.\d+)\)",
            bbox
        )[0]
        return float(bbox[0]), float(bbox[3]), float(bbox[2]), float(bbox[1])  # ulLon, ulLat, lrLon, lrLat


if __name__ == '__main__':
    conn, cur = connect()
    cur.execute("SELECT * FROM timezones")
    print(cur.fetchone())
    cur.execute("SELECT * FROM geonames WHERE geonameid = '3579932'")
    print(cur.fetchone())
