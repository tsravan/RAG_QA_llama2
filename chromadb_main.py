import time
import datetime
import chromadb
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pprint import pprint

from helper.constants import _Constant
import app_utils

CONST = _Constant()
chromadb_cfg = app_utils.read_yaml_file(CONST.CHROMADB_CONFIG_YML)
common_cfg = app_utils.read_yaml_file(CONST.COMMON_CONFIG_YML_PATH)

chromadb_path = chromadb_cfg['chromadb_path']
chromadb_collection_name = chromadb_cfg['chromadb_collection_name']



def create_vector_db(data_dir, data_format=None):
    if common_cfg['use_cuda'] == 'True':
        print('-----------> Using CUDA')
        embeddings = HuggingFaceEmbeddings(model_name=common_cfg['embedding_model_name'], multi_process=True, model_kwargs={'device':'cuda'})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=common_cfg['embedding_model_name'], multi_process=True, model_kwargs={'device':'cpu'})


    loader = DirectoryLoader(data_dir, glob=data_format, show_progress=True, 
                             loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.body_text[].text'})
    print('---------> Reading documents to loader')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chromadb_cfg['chunk_size'], chunk_overlap = chromadb_cfg['chunk_overlap'])
    texts = text_splitter.split_documents(documents)
    print('---------> Creating Chromadb')
    start_time = time.time()
    
    db = Chroma.from_documents(texts, embeddings, 
                               collection_name=chromadb_collection_name, 
                               persist_directory=chromadb_path)    
    print('---------> Time take to create Chromadb collection',f"{str(datetime.timedelta(seconds=(time.time() - start_time)))} seconds.")


def check_create_chromadb(cord19_data_dir, data_format):
    ''' Checks for chroma db in specified chromadb_path along 
        with chromadb_collection_name. 
        If not found creating chromadb with specified chromadb_path and chromadb_collection_name
        '''
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    try:
        chroma_client.get_collection(chromadb_collection_name)
        print(f"------> Using Chromadb collection '{chromadb_collection_name}'")
    except ValueError as val_error:
        if val_error =='Collection abc does not exist':
            print('------> Creating Chromadb')
            create_vector_db(cord19_data_dir, data_format=data_format)