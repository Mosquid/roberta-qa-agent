import os
from dotenv import load_dotenv
import pinecone


load_dotenv()

INDEX = "reranker-384-dim"
ENV = os.getenv("PINECONE_ENV")
API_KEY = os.getenv("PINECONE_API_KEY")

pinecone.init(api_key=API_KEY, environment=ENV)


def create(vector):
    index = pinecone.Index(INDEX)
    vec = index.upsert([(vector["id"], vector["embeds"], vector["metadata"])])
    print(vec)


def query(vector):
    index = pinecone.Index(INDEX)
    return index.query(vector["embeds"], top_k=3, include_metadata=True)


def clear():
    index = pinecone.Index(INDEX)
    index.delete(delete_all=True)
