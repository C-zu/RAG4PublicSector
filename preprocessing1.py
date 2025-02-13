<<<<<<< HEAD
# from docx import Document
# from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from langchain_community.document_loaders.text import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
=======
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
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
from pathlib import Path
import pickle
import os
from langchain_openai import ChatOpenAI
import pymongo
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
<<<<<<< HEAD
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
=======
from dotenv import load_dotenv
load_dotenv()
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd

import re

<<<<<<< HEAD
from dotenv import load_dotenv
load_dotenv()

# os.environ['COHERE_API_KEY'] = '7THPmgAjpjThSPmqNcpcER8T4h1FJNgZNEzpLGzv'
# MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://gogorun235:nhathuy@rag-publicsector.amgor2s.mongodb.net/"

# DB_NAME = "rag-db"
# COLLECTION_NAME = "documents"
# ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

=======
# os.environ['COHERE_API_KEY'] = '7THPmgAjpjThSPmqNcpcER8T4h1FJNgZNEzpLGzv'
# MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://gogorun235:nhathuy@rag-publicsector.amgor2s.mongodb.net/"

# DB_NAME = "rag-db"
# COLLECTION_NAME = "documents"
# ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
myclient = pymongo.MongoClient(os.getenv('MONGODB_ATLAS_CLUSTER_URI'))
mydb = myclient["rag-db"]
mycol = mydb["documents"]

class VectorDatabase():
<<<<<<< HEAD
    def __init__(self, data_path=None, embedding=None, db=None, mode=None) -> None:
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
            if (mode == "dense") or (mode is None):          
                self.db = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding,
                    path="./Qdrant",
                    collection_name="chunks",
                )
            if mode == "sparse":             
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                self.db = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding,
                    sparse_embedding=sparse_embeddings,
                    path="./Qdrant",
                    collection_name="chunks",
                    retrieval_mode=RetrievalMode.SPARSE,
                )
            if mode == "hybrid":             
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                self.db = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding,
                    sparse_embedding=sparse_embeddings,
                    path="./Qdrant",
                    collection_name="chunks",
                    retrieval_mode=RetrievalMode.HYBRID,
                )
        elif db == "qdrant_docs":
            if data_path is None:
                raise ValueError("Data path is required!")
            if (mode == "dense") or (mode is None):          
                self.db = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding,
                    path="./Qdrant",
                    collection_name="docs",
                )
            if mode == "sparse":             
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
                self.db = QdrantVectorStore.from_existing_collection(
                    embedding=self.embedding,
                    sparse_embedding=sparse_embeddings,
                    path="./Qdrant",
                    collection_name="docs",
                    retrieval_mode=RetrievalMode.SPARSE,
                )
            if mode == "hybrid":             
                sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
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

=======
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
    
    def save_db(self):
        return
        
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

>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
    def get_retriever(self):
        return self.compressed_retriever
    
    def re_ranking(self):
<<<<<<< HEAD
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
        # model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        # compressor = CrossEncoderReranker(model=model, top_n=5)
        llm = ChatOpenAI(
            model="Meta-Llama-3_1-70B-Instruct",    
            openai_api_base="https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
            openai_api_key="eyJhbGciOiJFZERTQSJ9.eyJwcm9qZWN0IjoiOTU0ZDdjMGNhNWI5NDAxMzgwZDhkOGEyZDg3YmJhYmUiLCJhdWQiOiIzNzM4NjExNjY0MDQzMDM0IiwiZXhwIjoxNzcwNjM1ODk0LCJqdGkiOiJmNjkxZGI4ZS05MmYwLTRiNTktYjFjNC00MzI0Nzg0MjBhNTciLCJpc3MiOiJ0cmFpbmluZy5haS5jbG91ZC5vdmgubmV0Iiwic3ViIjoibmcxMjkwNDgtb3ZoIiwib3ZoVG9rZW4iOiJGX1RMUnp0bG5MYmIyanpsZWVremhPVjNQcVA1UFJPaktQVDQzT1pyMG1xZTJUUTdHcE9HaXNPVnBnRFhIU2NUNm96VU95QkcxdlpOUzR0VWpoTVA5UFc0UlhrcVJzeVlrZ0F6Yzh1TE9vRlk5S0Q3dE1rYW1RN0taZXZsUk5nRXBRZjdoVURWeUZMaVlMcTJ6Z3RfWkQwZmVsUHNDOEw5c2JyYUNtSGQzbExKdnA5WWM3X0VpZ0tCeG9fNGhObWpsN3lWUWt2UTJwcHNJUk1BWUpuYmtkamFXN19CdldFU19sSE5xVy1OMTJZNnpXU1VVM0FwcDBWNmcxODRaRG9SS0JmR2ZvbGxWdjRscTIzb043WlA1Y0Zmd0xYUk5LTm44SGJqcGlPVGhpTEtGY05rNzc1UGlnbWdxb1h2ZHhkeC1iQXVfMG1Xa0U0VE1jWG1wZmsyOGlReUEwVWdxQ2RHaHJTOEt4bWlrbFJZQzVJMWljXy1qU2hfSnFZanROWFM5VVlucmlfbHQ4Qy1OVjI5R1RGOFNzRFYtVHY5MXh3aTNsb3p4Y2Vsei12MEtJTXdmVHJscU4ta0ZSd1ZhLVM5M1dXLWJub3hlZjNUMU5YQ0ktRVVIeGJjMk8xZk1TQ0RUUXRZTW5oM2JuUVpUZ3JlTmg0Q0Q2dU5lb2YwRlh5VjhzLUNqdkk5R3RFaTBZSUFnd2RTNzUxNGJyMmozQTlVcDRGTkdFcXVuV0lmUk4zbFRla082clY1UEhLRDczQVEtdkRqVF9haDdyX1BuTjZTd3FraUZhazdtSld1MmRHOXJ4VWxIRmFNalVyeE1qQm1rZG9MMEpDSVlaNkExTC0xOEVDMS1WNU9xeFZFNHY2WFkxREVHTXR1SmVvdmEySmJDVk5uUHZxNm5EZ1I3Mm9HMzkyT1hYaGxXVGlTcU9NX2VDcVQyNEY5V0k3RGpXM1E2VS10Sm9PeUV0WGJERFJLRzVHQUdMVTFlQS1FaDVzSUttem1iRlhkcHNMaWUxaEswblA2VFA2bDZIa3dDME42MklJaVRCeGlNbDItdmtNeHJvbDFRVHp4TzhIbG9raWlkV3RSTjZTckttRnE5ZU8yLTU2RW5uUHVxSnBBNG5FVi1DWUdMLXFCRS1oQjUyNXJ5SU1vSmZCMjZCM3E2MTZxNU42VHZzSzJ2a3p1Tm9kMjZzblRTNzRNemsxT2kzbFcya3lVNzRGaVBRVDEzZVJCdEo2ZU41dHMwSnN5MktseV95WnduSDMwSTFrbW9OM1hPV1ZoaTItbjB4YnhsclV3T3NTbHZuQWpyQ0JOVkxtX1E5cUR3UmNEM2ZKYSJ9.IWuqfMSqSoEOTTEHOxeyXKE7ZyMB_LXbVqh8h-PNJQwQhyeaIPa64Cp9GDaXhvQhZHpfBccLqWWbli--A4HKCg",
        )
        compressor = LLMChainExtractor.from_llm(llm)
=======
        compressor = CohereRerank(user_agent='langchain', cohere_api_key=os.getenv('COHERE_API_KEY'), model="rerank-multilingual-v2.0",top_n=1)
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.compressed_retriever,
        )
        return compression_retriever

