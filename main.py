from transformers import AutoTokenizer, AutoModel
import vectordb
import torch
from pathlib import Path
import json
from utils import flatten_json
import sys
import qa

kb_file = "kb.json"

model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def create_vector(doc, id):
    try:
        embeddings = generate_vectors(doc)

        vector = {
            "id": str(id),
            "embeds": embeddings,
            "metadata": {"doc": doc},
        }

        vectordb.create(vector)

    except (RuntimeError, TypeError, NameError) as err:
        print(err)


def generate_vectors(texts):
    """
    Generates vectors for a list of texts using a RoBERTa model.

    :param texts: List of texts to generate vectors for.
    :return: List of vectors.
    """

    # Prepare the texts for the model
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Generate vectors
    with torch.no_grad():
        model_output = model(**encoded_input)

    vectors = mean_pooling(model_output, encoded_input["attention_mask"])

    return vectors.tolist()


def create_vectors():
    vectordb.clear()
    documents = flatten_json(json.loads(Path(kb_file).read_text()))

    for id, key in enumerate(documents):
        val = documents[key]
        doc = key + ": " + val
        create_vector(doc, id)


def query_vector(input):
    embeddings = generate_vectors(input)
    vector = {
        "embeds": embeddings[0],
    }
    return vectordb.query(vector)


def main():
    input = sys.argv[1]
    vectors = query_vector(input)
    match = vectors.matches
    # iterate over the matches and take metadata
    context = ""
    for m in match:
        context += m.metadata["doc"] + "\n"

    answ = qa.answer_question(input, context)
    print(answ)


if __name__ == "__main__":
    main()
