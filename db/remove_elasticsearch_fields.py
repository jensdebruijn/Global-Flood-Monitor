from db.elastic import Elastic
import sys

es = Elastic()


def remove_field_from_index(index, field):
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "exists": {"field": field}
                    }
                ]
            }
        }
    }
    print(f"removing {es.n_hits(index=index, body=body)} documents from index '{index}'")
    body.update({
        "script": {
            "inline": f"ctx._source.remove(\"{field}\")"
        }
    })
    es.update_by_query(index=index, body=body, conflicts='proceed')


if __name__ == '__main__':
    remove_field_from_index(sys.argv[-2], sys.argv[-1])
