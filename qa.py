from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)


def answer_question(question, context):
    QA_input = {
        "question": question,
        "context": context,
    }
    res = nlp(QA_input)
    return res["answer"]
