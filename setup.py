import vectordb
from utils import flatten_json
from pathlib import Path
import json
from embeddings import create_vector

kb_file = "kb.json"


def create_vectors():
    vectordb.clear()
    documents = flatten_json(json.loads(Path(kb_file).read_text()))
    vectors = []

    for id, key in enumerate(documents):
        val = documents[key]
        doc = key + ": " + val
        vectors.append(create_vector(doc, id))

    vectordb.create(vectors)


vectordb.create_index()
create_vectors()
