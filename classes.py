from methods.docs import create_url
import psycopg2
from collections import namedtuple

from methods.spatial import distance_coords


class Location:
    __slots__ = [
        "location_ID",
        "score",
        "coordinates",
        "country_location_ID",
        "adm1_location_ID",
        "adm2_location_ID",
        "adm3_location_ID",
        "adm4_location_ID",
        "ext_scores",
        "type",
        "translations",
        "level_0_region",
        "level_1_region",
        "languages",
        "abbreviations",
        "toponym",
        "scores",
        "levels"
    ]

    def __init__(self, loc, scores=False):
        self.location_ID = loc['location_ID']
        try:
            self.score = loc['score']
        except KeyError:
            pass
        try:
            self.coordinates = tuple(loc['coordinates'])
        except KeyError:
            self.coordinates = None
        try:
            self.country_location_ID = loc['country_location_ID']
        except KeyError:
            pass
        try:
            self.adm1_location_ID = loc['adm1_location_ID']
        except KeyError:
            pass
        try:
            self.adm2_location_ID = loc['adm2_location_ID']
        except KeyError:
            pass
        try:
            self.adm3_location_ID = loc['adm3_location_ID']
        except KeyError:
            pass
        try:
            self.adm4_location_ID = loc['adm4_location_ID']
        except KeyError:
            pass
        try:
            self.ext_scores = loc['ext_scores']
        except KeyError:
            pass
        self.type = loc['type']
        self.translations = loc['translations']
        self.level_0_region = loc['level_0_region']
        self.level_1_region = loc['level_1_region']
        try:
            self.languages = tuple(loc['languages'])
        except KeyError:
            pass
        try:
            self.abbreviations = tuple(loc['abbreviations']) if loc['abbreviations'] is not None else None
        except KeyError:
            pass

        if 'toponym' in loc:
            self.toponym = loc['toponym']
        else:
            self.toponym = None

        if scores:
            self.scores = {}

    def __repr__(self):
        text = f"<Loc: {self.location_ID} - {self.type}"
        if self.toponym:
            text += f" - {self.toponym}"
        if hasattr(self, 'score'):
            text += f" - score: {self.score}"
        if hasattr(self, 'scores'):
            if not isinstance(self.scores, dict):
                scores_to_print = self.scores._asdict()
            else:
                scores_to_print = self.scores
            for score_type, score in scores_to_print.items():
                text += f" - {score_type}: {score}"
        text += f" - translations: {self.translations}>"
        return text

    def __str__(self):
        text = f"Loc {self.location_ID} - {self.type} - {create_url(self.location_ID)}"
        if self.toponym:
            text += f" - {self.toponym}"
        if hasattr(self, 'score'):
            text += f" - {self.score}"
        return text

    def is_child_of(self, loc):
        for kind in (
            (0, 'country'),
            (1, 'adm1'),
            (2, 'adm2'),
            (3, 'adm3'),
            (4, 'adm4'),
        ):
            if loc.location_ID == getattr(self, kind[1] + '_location_ID'):
                return True, kind[0]
        else:
            return False

    def contains(self, coordinates, pg):
        assert self.location_ID in (self.level_0_region, self.level_1_region)
        try:
            pg.cur.execute("""
                SELECT ST_Within(ST_SetSRID(ST_Point(%s, %s), 4326), locations.geom)
                FROM locations
                WHERE location_ID = %s
            """, (coordinates[0], coordinates[1], self.location_ID))
        except (psycopg2.ProgrammingError, psycopg2.InternalError):
            print(self.location_ID)
            raise
        res = pg.cur.fetchone()
        if res:
            return res[0]
        else:
            return None

    def is_parental_relation(self, loc):
        normal = self.is_child_of(loc)
        if normal is not False:
            return normal
        else:
            parentified = loc.is_child_of(self)
            if parentified is not False:
                return parentified
            else:
                return False

    def distance_between_sibblings(self, loc, pg, kind=None):
        if self.type in ('town', 'adm5', 'adm4', 'adm3', 'adm2', 'landmark'):
            return self.distance_to_coordinates(loc.coordinates, pg)
        else:
            pg.cur.execute("""
                SELECT ST_Distance(a.geom, b.geom)
                FROM locations a, locations b
                WHERE a.location_ID = %s
                AND b.location_ID = %s
            """, (self.location_ID, loc.location_ID))
            res = pg.cur.fetchone()
            if not res:
                return None
            else:
                return res[0]

    def distance_to_coordinates(self, coordinates, pg):
        if self.type in ('town', 'adm5', 'adm4', 'adm3', 'adm2', 'landmark'):
            if self.coordinates:
                return distance_coords(
                    self.coordinates,
                    coordinates
                )
            else:
                return None
        else:
            pg.cur.execute("""
                SELECT ST_Distance(locations.geom::geography, ST_GeographyFromText('POINT(%s %s)')) FROM locations WHERE location_ID = %s
            """, (coordinates[0], coordinates[1], self.location_ID))
            res = pg.cur.fetchone()
            if not res:
                return None
            else:
                return res[0]

    def matches_time_zone(self, time_zone, time_zones_per_region):
        """Checks if a tweets timezone corresponds with a locations time zone"""
        if time_zone:
            score = 0
            for level in (0, 1):
                try:
                    time_zones_region = time_zones_per_region[getattr(self, f'level_{level}_region')]
                except KeyError:
                    pass
                else:
                    if time_zones_region['total'] > 20:
                        try:
                            score += time_zones_region['tzs'][time_zone]
                        except KeyError:
                            pass
            return score
        else:
            return 0

    def is_sibbling_with(self, loc):
        for kind in (
            (4, 'adm4'),
            (3, 'adm3'),
            (2, 'adm2'),
            (1, 'adm1'),
            (0, 'country'),
        ):
            loc1_kind_location_ID = getattr(self, kind[1] + '_location_ID')
            loc2_kind_location_ID = getattr(loc, kind[1] + '_location_ID')
            if loc1_kind_location_ID is None and loc2_kind_location_ID is None:
                continue
            if loc1_kind_location_ID == loc2_kind_location_ID:
                if kind[1] == 'adm1':
                    if self.type == 'adm1' or loc.type == 'adm1':
                        return False
                    else:
                        return kind[0]
                elif kind[1] == 'country':
                    if self.type == 'adm1' and loc.type == 'adm1':
                        return kind[0]
                    else:
                        return False
                else:
                    return kind[0]
        else:
            return False

    def is_family(self, loc):
        is_child = self.is_child_of(loc)
        if is_child:
            return 'child-parent', is_child
        is_child = self.is_child_of(loc)
        if is_child:
            return 'parent-child', is_child
        return 'sibblings', self.is_sibbling_with(loc)


def locations_to_class(docs):
    for doc in docs:
        if '_source' in doc:
            doc['_source']['locations'] = [
                Location(loc) for loc in doc['_source']['locations']
            ]
        else:
            doc['locations'] = [
                Location(loc) for loc in doc['locations']
            ]
        yield doc


Tweet = namedtuple("Tweet", "id text clean_text language date author_id author_location author_timezone coordinates bbox_center urls media")
Article = namedtuple("Article", "id text language author_id date")
BaseDoc = namedtuple("BaseDoc", "id language author_id date text clean_text")

AnalyzedDoc = namedtuple("AnalyzedDoc", "locations resolved_locations author_id date text clean_text language")


if __name__ == '__main__':
    from db.postgresql import PostgreSQL
    pg = PostgreSQL('gfm')
    loc1 = Location({
        "location_ID": 6252001,
        "type": "country",
        "coordinates": (0, 0),
        "translations": 15,
        "level_0_region": 6252001,
        "level_1_region": None,
        'additional_relations': None,
        'basin_ids': None
    })
    loc2 = Location({
        "location_ID": 2750405,
        "type": "country",
        "coordinates": (0, 0),
        "translations": 15,
        "level_0_region": 6252001,
        "level_1_region": None,
        'additional_relations': None,
        'basin_ids': None
    })
    print(loc1.distance_to_coordinates((-74, 40.7), pg))
    print(loc1.distance_between_sibblings(loc2, pg))
    assert loc1.contains((-104, 39.7), pg) is True
    assert loc1.contains((0, 0), pg) is False
