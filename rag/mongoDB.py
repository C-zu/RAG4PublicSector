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
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
# os.environ['COHERE_API_KEY'] = 'rSkSDZbstuXIFUuWWlRgGIeNMe6e8u3yzG7cACD0'
# from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

myclient = pymongo.MongoClient("mongodb+srv://nghia:nghia@rag-db.3zxzfye.mongodb.net/")
mydb = myclient["rag-db"]
mycol = mydb["documents"]

def load_chunk(directory_path):
    #embedding = GoogleGenerativeAIEmbeddings(model="models/text-multilingual-embedding-002",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
    #embedding_2 = SpacyEmbeddings(model_name="xx_ent_wiki_sm")
    # model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    # hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})
    cohere_embedding = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # embed = AI21Embeddings(
    #     api_key="8KTQxqaBgtqRgHAUjg0I2SFIPwSg7nXK",
    # )
    # embedding = GoogleGenerativeAIEmbeddings(model="models/text-multilingual-embedding-002",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
    #custom_embedding = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    # embeddings = OpenAIEmbeddings(
    #     model="hf:BAAI/bge-m3",    
    #     openai_api_key="glhf_7cacb0cf53d5065f1b46d6fcc98dd784",
    #     openai_api_base="https://glhf.chat/api/openai/v1",
    # )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    chunked_documents = []

    for file_path in Path(directory_path).rglob('*.*'):
        if file_path.is_file():
            loader = TextLoader(file_path, encoding='utf8')
            data = loader.load()
            data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0][:-2]
            chunked_documents.extend(text_splitter.split_documents(data))

    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        chunked_documents,
        embedding=embeddings,
        collection=mycol,
        index_name=os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')
    )
    
load_chunk('./data/txt_file')