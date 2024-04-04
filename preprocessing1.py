from docx import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain import hub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pathlib import Path
import pickle
import os
import qdrant_client
import pymongo
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings


os.environ['COHERE_API_KEY'] = '7THPmgAjpjThSPmqNcpcER8T4h1FJNgZNEzpLGzv'
MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://gogorun235:nhathuy@rag-publicsector.amgor2s.mongodb.net/"

DB_NAME = "rag-db"
COLLECTION_NAME = "documents"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

myclient = pymongo.MongoClient("mongodb+srv://gogorun235:nhathuy@rag-publicsector.amgor2s.mongodb.net/")
mydb = myclient["rag-db"]
mycol = mydb["documents"]


def load_chunk(directory_path):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    chunked_documents = []

    for file_path in Path(directory_path).rglob('*.*'):
        if file_path.is_file():
            loader = TextLoader(file_path, encoding='utf8')
            data = loader.load()
            data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0]
            chunked_documents.extend(text_splitter.split_documents(data))
    bm25_retriever = BM25Retriever.from_documents(chunked_documents, k = 5)
    
    
    Qdrant.from_documents(
        chunked_documents,
        embeddings,
        path="./Qdrant",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )
    
    # vectorstore = MongoDBAtlasVectorSearch.from_documents(
    # chunked_documents,
    # embeddings,
    # collection=COLLECTION_NAME,
    # index_name=DB_NAME,
    # )
    
    with open('./data/bm25_retriever.pkl', 'wb') as f:
        pickle.dump(bm25_retriever, f)
    

def load_db():
    # model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # embeddings = HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"})
    
    embeddings = embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    # client = qdrant_client.QdrantClient(
    #     path="./Qdrant",
    # )
    # docsearch = Qdrant(
    #     client=client, collection_name="my_documents", embeddings=embeddings
    # )
    with open('./data/bm25_retriever.pkl', 'rb') as f:
        bm25_retriever = pickle.load(f)
    
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        MONGODB_ATLAS_CLUSTER_URI,
        DB_NAME + "." + COLLECTION_NAME,
        embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    
    mongodb_retriever = vector_search.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, mongodb_retriever], weights=[0.5, 0.5])

    # Cohere Reranker
    compressor = CohereRerank(user_agent='langchain')
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever

txt_path = './data/txt_file'
    
retriever = load_db()