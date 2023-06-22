from db.elastic import Elastic

es = Elastic()

# get the current indices
indices = es.cat.indices(format='json')

# print the index names
for index in indices:
    print(index['index'])

es.indices.delete(index='gfm')