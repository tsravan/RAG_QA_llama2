
import time
import datetime
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

from helper.constants import _Constant
import app_utils

CONST = _Constant()
common_cfg = app_utils.read_yaml_file(CONST.COMMON_CONFIG_YML_PATH)
chromadb_cfg = app_utils.read_yaml_file(CONST.CHROMADB_CONFIG_YML)
inference_cfg = app_utils.read_yaml_file(CONST.INFERENCE_CONFIG_YML)


custom_template = """You are a scholar in medical and clinical research
                    Use the following pieces of context 
                    context: {context}
                    Answer the question
                    question: {question}
                    If you don't know the answer, please think rationally and answer from your own knowledge base 
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    print(f"----------> loading model {inference_cfg['model_path']}")
    if common_cfg['use_cuda'] == 'True':
        llm = CTransformers(
            model = inference_cfg['model_path'],
            model_type = 'llama',
            max_new_tokens = 512,            
            temperature = 0,
            gpu_layers=10)
    else:
        llm = CTransformers(
            model = inference_cfg['model_path'],
            model_type = 'llama',
            max_new_tokens = 4096,            
            temperature = 0)
    return llm


def retrieval_qa_chain(llm,qa_prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs = {'k':3}),
        return_source_documents = True,
        verbose=True,
        # chain_type_kwargs={'prompt':qa_prompt}
    )
    return qa_chain

def qa_bot():
    start_time = time.time()
    if common_cfg['use_cuda'] == 'True':
        print('-----------> Using CUDA')
        print(f"""-----------> Using data in {chromadb_cfg['chromadb_path']} \n
              chromadb collection: {chromadb_cfg['chromadb_collection_name']}""")
        embeddings = HuggingFaceEmbeddings(model_name=common_cfg['embedding_model_name'],
                                            multi_process=True, model_kwargs={'device':'cuda'})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=common_cfg['embedding_model_name'], 
                                           multi_process=True, model_kwargs={'device':'cpu'})
    
    db = Chroma(persist_directory= chromadb_cfg['chromadb_path'], collection_name=chromadb_cfg['chromadb_collection_name'],
                embedding_function=embeddings)
    llm = load_llm()
    
    qa_prompt = set_custom_prompt()
    qa  = retrieval_qa_chain(llm,qa_prompt,db)
    print('---------> Time taken to load QA Bot:',f"{str(datetime.timedelta(seconds=(time.time() - start_time)))} seconds.")
    return qa

