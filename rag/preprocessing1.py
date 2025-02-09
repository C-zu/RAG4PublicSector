# from docx import Document
# from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from langchain_community.document_loaders.text import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain_core.prompts import ChatPromptTemplate   
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import pickle
import os
from langchain_openai import ChatOpenAI
import pymongo
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.qdrant import QdrantTranslator
from langchain_core.retrievers import BaseRetriever
# from qdrant_client import QdrantClient, models
from langchain_core.documents.base import Document as Document
from typing import (
    List,
)

import re

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
        if embedding is None:
            self.embedding = SpacyEmbeddings(model_name="en_core_web_sm")
            
        if db is None:
            raise ValueError("'db' cannot be None")
        
        elif db == "pickle_docs":
            with open('./data/docs.pkl', 'rb') as f:
                self.db = pickle.load(f)
        elif db == "pickle_chunks":
            with open('./data/chunks.pkl', 'rb') as f:
                self.db = pickle.load(f)
        
        elif db == "mongo":
            self.db = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=os.getenv('MONGODB_ATLAS_CLUSTER_URI'),
                namespace=os.getenv('DB_NAME') + "." + os.getenv('COLLECTION_NAME'),
                embedding=self.embedding,
                index_name=os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME'),
            )
        elif db == "qdrant_chunks":
            if data_path is None:
                raise ValueError("Data path is required!")
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            self.db = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                #sparse_embedding=sparse_embeddings,
                path="./Qdrant",
                collection_name="chunks",
                #retrieval_mode=RetrievalMode.HYBRID,
            )
        elif db == "qdrant_docs":
            if data_path is None:
                raise ValueError("Data path is required!")
            #sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            self.db = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding,
                sparse_embedding=sparse_embeddings,
                path="./Qdrant",
                collection_name="docs",
                retrieval_mode=RetrievalMode.HYBRID,
            )
        elif db == "qdrant_create":
            if data_path is None:
                raise ValueError("Data path is required!")
        
            with open('./data/chunked_docs.pkl', 'rb') as f:
                chunked_documents = pickle.load(f)
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            self.db = QdrantVectorStore.from_documents(
                chunked_documents,
                self.embedding,
                path="./Qdrant",
                collection_name="selfquery_data",
                sparse_embedding=sparse_embeddings,
                force_recreate = True,
                retrieval_mode=RetrievalMode.HYBRID,
            )
        else:
            chunked_documents = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            for file_path in Path(data_path).rglob('*.*'):
                if file_path.is_file():
                    loader = TextLoader(file_path, encoding='utf8')
                    data = loader.load()
                    # metadata_dict, data = self.add_metadata(data,file_path)
                    data = self.add_metadata(data,file_path)
                    # data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0][:-2]
                    chunked_documents.extend(text_splitter.split_documents(data))
            self.db = chunked_documents
            with open('./data/'+ db, 'wb') as f:
                pickle.dump(self.db, f)
    def extract_procedure_name(self,text):
        match = re.search(r"Tên thủ tục:\s*(.*)", text)
        if match:
            return match.group(1).strip()
        return None
    def extract_summarization_content(self,text):
        match = re.search(r"Nội dung:\s*(.*)", text)
        if match:
            return match.group(1).strip()
        return None
    def add_metadata(self,data,file_path):
        metadata_dict = {}
        data = data[0]
        metadata_dict["name"] = self.extract_procedure_name(data.page_content)
        metadata_dict["content"] = self.extract_summarization_content(data.page_content)
        # data.page_content = "Thủ tục" + " " + self.extract_procedure_name(data.page_content)
        # lines = data.page_content.strip().splitlines()
        # for i, line in enumerate(lines):
        #     if line.startswith("Tên thủ tục:"):
        #         metadata_dict["name"] = line.replace("Tên thủ tục:", "").strip()
        #     if line.startswith("Nội dung:"):
        #         metadata_dict["content"] = line.replace("Nội dung:", "").strip()
        #         break
        metadata_dict["source"] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0][:-2]
        data.metadata = metadata_dict
        return [data]
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
                self.retriever = BM25Retriever.from_documents(db.db, k = 5)
            elif name_retriever == "selfquery":
                metadata_field_info = [
                    AttributeInfo(
                        name="name",
                        description="Tên của thủ tục",
                        type="string",
                    ),
                    AttributeInfo(
                        name="content",
                        description="Tóm tắt nội dung của thủ tục",
                        type="string",
                    ),
                ]
                #llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0)
                llm = ChatOpenAI(
                    model="mistral-large-2407",    
                    openai_api_base="https://api.mistral.ai/v1",
                    openai_api_key="jDVi03OCpbFVWaAZAd8gpa9JOL8mjdqU",
                )
                json_llm = llm.bind(response_format={"type": "json_object"})
                self.retriever = SelfQueryRetriever.from_llm(
                    llm=json_llm,
                    vectorstore=db.db,
                    document_contents="Một đoạn chunk ngắn trong một văn bản thủ tục",
                    metadata_field_info=metadata_field_info,
                    structured_query_translator=QdrantTranslator(metadata_key=db.db.metadata_payload_key),
                    search_kwargs={'k': 5},
                    # enable_limit=True,
                    # use_original_query=True,
                )
            else:
                self.retriever = BM25Retriever.from_documents(db.db, k = 5)
        else:
            self.retriever = db.db.as_retriever(search_kwargs={"k": 5})
      
    def __getattribute__(self, retriever):
        return super().__getattribute__(retriever)
    
    def get_retriever(self):
        return self.retriever
    
class TwoStageRetriever(BaseRetriever):
    def __init__(self, document_store: VectorDatabase, chunk_store: VectorDatabase, top_k_docs: int = 5, top_k_chunks: int = 5):
        """
        Initialize the retriever.

        Args:
            document_store: VectorDatabase for documents.
            chunk_store: VectorDatabase for chunks.
            top_k_docs: Number of documents to retrieve in the first stage.
            top_k_chunks: Number of chunks to retrieve in the second stage.
        """
        self.document_store = document_store
        self.chunk_store = chunk_store
        self.top_k_docs = top_k_docs
        self.top_k_chunks = top_k_chunks

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents and chunks in two stages.

        Args:
            query: The query to retrieve documents for.

        Returns:
            List of relevant Document objects (chunks).
        """
        # Stage 1: Retrieve top_k_docs documents
        retrieved_docs = self.document_store.similarity_search(query, k=self.top_k_docs)
        
        # Extract doc IDs from the retrieved documents
        doc_ids = [doc.metadata["doc_id"] for doc in retrieved_docs]
        
        # Stage 2: Retrieve top_k_chunks chunks from the retrieved documents
        retrieved_chunks = self.chunk_store.similarity_search(
            query,
            k=self.top_k_chunks,
            filter={"doc_id": {"$in": doc_ids}}  # Filter chunks by document IDs
        )
        
        return retrieved_chunks
        
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
        #compressor = CohereRerank(user_agent='langchain', cohere_api_key=os.getenv('COHERE_API_KEY'), model="rerank-multilingual-v3.0",top_n=1)
        # compressor = FlashrankRerank()
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        compressor = CrossEncoderReranker(model=model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.compressed_retriever,
        )
        return compression_retriever


class CompressedSingleRetriever():
    def __init__(self, retriever=None) -> None:        
        self.compressed_retriever = retriever
    def get_retriever(self):
        return self.compressed_retriever
    def re_ranking(self):
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        compressor = CrossEncoderReranker(model=model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.compressed_retriever,
        )
        return compression_retriever

