# Simple QA chain

_This project utilizes paraphrase-MiniLM-L6-v2 and roberta-base-squad2 to power a simple question-answering (QA) system._

## Overview

This QA system leverages the capabilities of state-of-the-art NLP models to provide accurate and relevant answers to user queries. It is designed to be easy to set up and use, making it ideal for a variety of applications where quick and reliable information retrieval is key.

## Setup Instructions

### Environment Setup:

1. Create a `.env`` file in the root directory of the project. This file should contain the following environment variables:

```
PINECONE_ENV=<your_pinecone_environment>
PINECONE_API_KEY=<your_pinecone_api_key>
```

2. Create a `kb.json` file. This file should contain the knowledge base that the chatbot will use to answer questions. Ensure it follows the required JSON format.
3. Run the `create_vectors`` function. This will generate and insert vectors into your Pinecone database, which are necessary for the chatbot to understand and retrieve information from the knowledge base.
4. To ask a question, use the command line interface:
   `python main.py "<your question>"`
