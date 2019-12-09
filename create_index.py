from db.elastic import Elastic
from config import DOC_SCORE_TYPES, DOCUMENT_INDEX

es = Elastic()
es.maybe_create_document_index(DOCUMENT_INDEX, DOC_SCORE_TYPES)
