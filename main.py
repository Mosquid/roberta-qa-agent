from transformers import AutoTokenizer, AutoModel
import vectordb
import torch
from pathlib import Path
import json
from utils import flatten_json
import sys
import qa
from embeddings import query_vector


model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


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
