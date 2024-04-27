from docx import Document
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_cohere import CohereRerank
from langchain_community.document_loaders.text import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain import hub
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate   
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pathlib import Path
import pickle
import os
import qdrant_client
import pymongo
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from dotenv import load_dotenv
load_dotenv()


# os.environ['COHERE_API_KEY'] = '7THPmgAjpjThSPmqNcpcER8T4h1FJNgZNEzpLGzv'
# MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://gogorun235:nhathuy@rag-publicsector.amgor2s.mongodb.net/"

# DB_NAME = "rag-db"
# COLLECTION_NAME = "documents"
# ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

myclient = pymongo.MongoClient(os.getenv('MONGODB_ATLAS_CLUSTER_URI'))
mydb = myclient["rag-db"]
mycol = mydb["documents"]

class VectorDatabase():
    def __init__(self, data_path=None, embedding=None, db=None) -> None:
        self.embedding = embedding
        self.retriever = None
        self.db = None

        if db is None:
            chunked_documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            for file_path in Path(data_path).rglob('*.*'):
                if file_path.is_file():
                    loader = TextLoader(file_path, encoding='utf8')
                    data = loader.load()
                    data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0]
                    chunked_documents.extend(text_splitter.split_documents(data))
            self.db = chunked_documents

        if embedding is None:
            self.embedding = SpacyEmbeddings(model_name="en_core_web_sm")

        if db == "mongo":
            self.db = MongoDBAtlasVectorSearch.from_connection_string(
                os.getenv('MONGODB_ATLAS_CLUSTER_URI'),
                os.getenv('DB_NAME') + "." + os.getenv('COLLECTION_NAME'),
                self.embedding,
                index_name=os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME'),
            )
        
        if db == "qdrant":
            if data_path is None:
                raise ValueError("Data path is required!")
            
            chunked_documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=1000)
            for file_path in Path(data_path).rglob('*.*'):
                if file_path.is_file():
                    loader = TextLoader(file_path, encoding='utf8')
                    data = loader.load()
                    data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0]
                    chunked_documents.extend(text_splitter.split_documents(data))
            
            self.db = Qdrant.from_documents(
                chunked_documents,
                self.embedding,
                path="./Qdrant",
                collection_name="my_documents",
                force_recreate = True,  
            )

    def __getattribute__(self, db):
        return super().__getattribute__(db)


class Retriever():
    def __init__(self, type_retriever = None, name_retriever = None, db = None) -> None:
        self.retriever = None

        if name_retriever is None:  
            raise ValueError("Name of retriever is required!")
        
        if type_retriever is None or type_retriever == "indexing":  
            self.type_retriever = "indexing"

            if name_retriever == "bm25":
                self.retriever = BM25Retriever.from_documents(db.db, k = 3)

        else:
            self.retriever = db.db.as_retriever(search_kwargs={"k": 3})
      
    def __getattribute__(self, retriever):
        return super().__getattribute__(retriever)
    
    def get_retriever(self):
        return self.retriever
    

class CompressedRetriever():
    def __init__(self, retriever1=None, retriever2=None, weights=None) -> None:        
        self.compressed_retriever = None
        
        if retriever1 and retriever2:
            self.compressed_retriever = EnsembleRetriever(retrievers=[retriever1, retriever2], weights=weights)
        else:
            raise ValueError("Both retriever1 and retriever2 must be provided.")

    def get_retriever(self):
        return self.compressed_retriever
    
    def re_ranking(self):
        compressor = CohereRerank(user_agent='langchain', cohere_api_key=os.getenv('COHERE_API_KEY'), model="rerank-multilingual-v2.0",top_n=1)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.compressed_retriever,
        )
        return compression_retriever

