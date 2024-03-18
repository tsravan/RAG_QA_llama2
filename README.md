# Question Answering Using, Langchain + ChromaDB + llama2 + Question Answering Evaluation

## A functional Question Answering project developed using:
###  1. Langchain building RAG based question answering system using chromadb and llama2
###  2. ChromaDB as vector database for storing the documents.
###  3. llama2 llm for answering the question using the top matching answers retrieved from chromadb.
###  4. Evaluating llama2 model

Currently using this application to query documents related to cord19 [dataset](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge).
Modify the configs if you want to use your own data.

![Architecture text](https://github.com/tsravan/RAG_QA_llama2/blob/main/misc/Architecture.drawio.png))

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/TH0njNC6oAY/0.jpg)](http://www.youtube.com/watch?v=TH0njNC6oAY)

Installation instructions:
1. Install Python Version 3.11.8
2. Install Dependencies by running pip install -r requirements.txt
3. Download Llama2 model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q5_0.bin?download=true

How to run it the tool:
1. Update model_path in inference_config.yml    
2. Run fe_pipeline.py file, this will create chromadb
3. Run app_main.py file -> In the termial run python -m streamlit run app_main.py --client.showErrorDetails=false

