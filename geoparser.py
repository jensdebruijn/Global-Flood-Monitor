from math import sqrt
from collections import defaultdict
from methods.tweets import LastTweetsDeque
from config import DOCUMENT_INDEX

class Geoparser:
    def __init__(
        self,
        pg,
        es,
        doc_score_types,
        max_distance_entities_doc,
    ):
        self.es = es
        self.pg = pg
        self.doc_score_types = doc_score_types
        self.max_distance_entities_doc = max_distance_entities_doc
        self.geoparse_threshold = 0.01

    def get_update_body(self, ID, doc):
        locations = []
        for resolved_loc in doc.resolved_locations:
            loc = {
                'location_ID': resolved_loc.location_ID,
                'toponym': resolved_loc.toponym,
                'level_0_region': resolved_loc.level_0_region,
                'level_1_region': resolved_loc.level_1_region,
                'translations': resolved_loc.translations,
                'coordinates': resolved_loc.coordinates,
                'type': resolved_loc.type,
                'score': resolved_loc.score
            }

            locations.append(loc)

        return {
            'doc': {
                'locations': locations,
            },
            '_index': DOCUMENT_INDEX,
            '_id': ID,
            '_op_type': 'update',
            'doc_as_upsert': True
        }

    def commit(self, tweets):
        """Commit tweets to the database."""
        self.es.bulk_operation(tweets)

    def gather_package_for_remote_server(self, ID, doc):
        if doc.resolved_locations:
            location_IDs = tuple(loc.location_ID for loc in doc.resolved_locations)
            self.pg.cur.execute("""
                SELECT
                    location_ID,
                    full_name
                FROM locations
                WHERE location_ID IN %s
            """, (location_IDs, ))
            official_names = {
                location_ID: official_name
                for location_ID, official_name
                in self.pg.cur.fetchall()
            }
            locations = [
                {
                    'location_ID': location.location_ID,
                    'loc': {
                        'lat': location.coordinates[1],
                        'lng': location.coordinates[0]
                    },
                    'score': location.score,
                    'official_name': official_names[location.location_ID],
                    'mentioned_name': location.toponym,
                    'type': location.type,
                    'adm1_location_ID': location.adm1_location_ID if location.adm1_location_ID is not None else None,
                    'country_location_ID': location.country_location_ID if location.country_location_ID is not None else None,
                    'additional_relations': [
                        add_loc for add_loc in location.additional_relations
                    ] if location.additional_relations is not None else None
                } for location in doc.resolved_locations
            ]
        else:
            locations = []
        return {
            'id': ID,
            'locations': locations
        }

    def get_one_docloc_per_user(self, doclocs):
        scores_by_user = {}
        for docloc in doclocs:
            author = docloc[0].author_id
            if author in scores_by_user:
                scores_by_user[author].append(docloc)
            else:
                scores_by_user[author] = [docloc]
        one_docloc_per_user = []
        for author, doclocs in scores_by_user.items():
            if len(doclocs) > 1:
                one_docloc_per_user.append(
                    sorted(doclocs, key=lambda docloc: docloc[0].date, reverse=True)[0]
                )
            else:
                one_docloc_per_user.append(doclocs[0])
        return one_docloc_per_user

    def get_non_duplicate_doclocs(self, doclocs):
        docs_wo_duplicates = []
        last_tweets_dict = LastTweetsDeque()
        for docloc in doclocs:
            if not last_tweets_dict.is_similar_to(clean_text=docloc[0].clean_text):
                docs_wo_duplicates.append(docloc)
        return docs_wo_duplicates

    def find_similar_in_country(self, resolved_location, locations):
        if resolved_location.type == 'adm1':
            return resolved_location
        else:
            for loc in locations:
                if (
                    loc.type == 'adm1'
                    and loc.country_location_ID == resolved_location.country_location_ID
                ):
                    try:
                        assert loc.country_location_ID
                    except AssertionError:
                        print(loc.location_ID)
                    return loc
            else:
                return resolved_location

    def score_single(self, doc):
        location_scores = {}
        for _, locations in doc.locations.items():
            for loc in locations:
                score = 0
                for score_type in self.doc_score_types.keys():
                    score += getattr(loc.scores, score_type)
                location_scores[loc.location_ID] = score
        return location_scores

    def score_group(self, docs, timestep_end):
        location_scores = defaultdict(list)
        for doc in docs.values():
            for locations in doc.locations.values():
                for loc in locations:
                    location_scores[loc.location_ID].append((doc, loc))
        total_score_per_type = {
            score_type: 0
            for score_type in self.doc_score_types
        }
        for location_ID, doclocs in location_scores.items():
            total_score = 0
            one_doc_per_user = self.get_one_docloc_per_user(doclocs)
            for score_type in self.doc_score_types.keys():
                if score_type == 'family':
                    if sum(docloc[1].scores.family for docloc in doclocs) > 0:
                        doclocs_wo_duplicates = self.get_non_duplicate_doclocs(doclocs)
                        score = float(sum(
                            docloc[1].scores.family
                            for docloc in doclocs_wo_duplicates
                        )) / len(doclocs_wo_duplicates)
                    else:
                        score = 0
                else:
                    score = float(sum(
                        getattr(docloc[1].scores, score_type) for docloc in one_doc_per_user
                    ) / len(one_doc_per_user))
                total_score += score
                total_score_per_type[score_type] += score
                del score
            location_scores[location_ID] = total_score
        return dict(location_scores)

    def resolve_doc(self, doc, location_scores):
        resolved_locations = []
        for locations in doc.locations.values():
            for location in locations:
                location.score = location_scores[location.location_ID]
            if not locations:
                continue
            locations = sorted(
                locations,
                key=lambda loc: loc.score * sqrt(loc.translations + 1),
                reverse=True
            )
            for loc in locations:
                if loc.type in ('country', 'continent'):
                    resolved_location = loc
                    break
            else:
                resolved_location = locations[0]
                resolved_location = self.find_similar_in_country(
                    resolved_location,
                    locations
                )
            resolved_locations.append(resolved_location)

        if len(resolved_locations) == 1:
            if (
                resolved_locations[0].type in ('country', 'continent')
                or resolved_locations[0].score >= self.geoparse_threshold
            ):
                return resolved_locations
            else:
                return None

        elif all(loc.type in ('country', 'continent') for loc in resolved_locations):
            return resolved_locations

        else:
            fully_resolved_locations = [
                loc for loc in resolved_locations if loc.type in ('country', 'continent')
            ]
            if not fully_resolved_locations:
                fully_resolved_locations = sorted([
                    loc for loc in resolved_locations if loc.score >= self.geoparse_threshold
                ], key=lambda loc: loc.score * sqrt(loc.translations + 1), reverse=True)[:1]
            if fully_resolved_locations:
                possible_locations = [loc for loc in resolved_locations if loc not in fully_resolved_locations]
                n_resolved_locations = len(fully_resolved_locations)
                for possible_loc in possible_locations:
                    for i in range(n_resolved_locations):
                        resolved_location = fully_resolved_locations[i]
                        if possible_loc.is_parental_relation(resolved_location):
                            fully_resolved_locations.append(possible_loc)
                        else:
                            is_sibblings = possible_loc.is_sibbling_with(resolved_location)
                            if is_sibblings:
                                if is_sibblings > 0:
                                    fully_resolved_locations.append(possible_loc)
                                else:
                                    distance_between_sibblings = resolved_location.distance_between_sibblings(
                                        possible_loc,
                                        pg=self.pg
                                    )
                                    if distance_between_sibblings < 200_000:
                                        fully_resolved_locations.append(possible_loc)
            return fully_resolved_locations

    def resolve_group_docs(self, docs, timestep_end):
        location_scores = self.score_group(docs, timestep_end)
        for ID, doc in docs.items():
            resolved_locations = self.resolve_doc(doc, location_scores)
            if resolved_locations:
                yield ID, resolved_locations

    def resolve_single_doc(self, doc):
        location_scores = self.score_single(doc)
        resolved_locations = self.resolve_doc(doc, location_scores)
        doc = doc._replace(resolved_locations=resolved_locations)
        return doc

    def locations_to_commit(
        self,
        resolved_locations,
    ):
        """Run through each document (ID) and its resolved locations and commit that to the database.
        The function first checks with the cache (self.docs) if an update is neccesary"""
        for ID, locations in resolved_locations:
            doc = self.docs[ID]
            locations = sorted(locations, key=lambda loc: loc.toponym)
            # Check if locations key already exists in the tweets dictionary.
            # If so, these are the locations in the database. And the code
            # in the else-block is ran to see if one or more of the locations
            # should be updated.
            # If the locations key does not exist, the db_locations are None,
            # and the new_locations are the currently assigned locations.
            if doc.resolved_locations is None:
                new_locations = locations
            else:
                new_locations = []
                for db_loc in doc.resolved_locations:
                    try:
                        new_locations.append(next(
                            loc for loc in locations
                            if loc.toponym == db_loc.toponym
                            and loc.score > db_loc.score
                        ))
                    except StopIteration:
                        new_locations.append(db_loc)

                for loc in locations:
                    try:
                        next(
                            db_loc for db_loc in doc.resolved_locations
                            if db_loc.toponym == loc.toponym
                        )
                    except StopIteration:
                        new_locations.append(loc)

            if doc.resolved_locations != new_locations:
                doc = doc._replace(resolved_locations=new_locations)
                self.docs[ID] = doc
                body = self.get_update_body(ID, doc)
                yield body


    def geoparse_timestep(self, timestep_end, update_locations, classify=True):
        """This function drives the analysis of a timestep and thus drives most other function."""
        resolved_locations = self.resolve_group_docs(self.docs, timestep_end)
        locations_to_commit = self.locations_to_commit(resolved_locations)
        if update_locations:
            self.commit(locations_to_commit)
        else:
            for _ in locations_to_commit:
                continue
