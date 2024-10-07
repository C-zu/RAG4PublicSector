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
from docx import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings

# os.environ['COHERE_API_KEY'] = 'rSkSDZbstuXIFUuWWlRgGIeNMe6e8u3yzG7cACD0'
myclient = pymongo.MongoClient("mongodb+srv://nghia:nghia@rag-db.3zxzfye.mongodb.net/")
mydb = myclient["rag-db"]
mycol = mydb["documents"]

def load_chunk(directory_path):
    # embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
    # embedding_2 = SpacyEmbeddings(model_name="en_core_web_sm")
    # model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    # hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})
    cohere_embedding = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    chunked_documents = []

    for file_path in Path(directory_path).rglob('*.*'):
        if file_path.is_file():
            loader = TextLoader(file_path, encoding='utf8')
            data = loader.load()
            data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0]
            chunked_documents.extend(text_splitter.split_documents(data))

    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        chunked_documents,
        cohere_embedding,
        collection=mycol,
        index_name=os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')
    )
    
load_chunk('./data/txt_file')