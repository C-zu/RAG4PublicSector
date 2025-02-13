from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import os
import pymongo
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
<<<<<<< HEAD
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# os.environ['COHERE_API_KEY'] = 'rSkSDZbstuXIFUuWWlRgGIeNMe6e8u3yzG7cACD0'
# from langchain_mistralai import MistralAIEmbeddings
import re
myclient = pymongo.MongoClient(os.getenv('MONGODB_ATLAS_CLUSTER_URI'))
=======
from docx import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings

os.environ['COHERE_API_KEY'] = '7THPmgAjpjThSPmqNcpcER8T4h1FJNgZNEzpLGzv'
myclient = pymongo.MongoClient("mongodb+srv://nghia:nghia@rag-db.3zxzfye.mongodb.net/")
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
mydb = myclient["rag-db"]
mycol = mydb["documents"]

def extract_procedure_name(text):
    match = re.search(r"Tên thủ tục:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return None
def add_metadata(data,file_path):
    metadata_dict = {}
    data = data[0]
    lines = data.page_content.strip().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Tên thủ tục:"):
            metadata_dict["name"] = line.replace("Tên thủ tục:", "").strip()
        if line.startswith("Nội dung:"):
            metadata_dict["content"] = line.replace("Nội dung:", "").strip()
            break
    metadata_dict["source"] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0][:-2]
    data.metadata = metadata_dict
    return [data]
def load_chunk(directory_path):
<<<<<<< HEAD
    #embedding = GoogleGenerativeAIEmbeddings(model="models/text-multilingual-embedding-002",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
    #embedding_2 = SpacyEmbeddings(model_name="xx_ent_wiki_sm")
    # model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    # hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})
    # cohere_embedding = CohereEmbeddings(
    #     model="embed-multilingual-v3.0",
    #     cohere_api_key=os.getenv("COHERE_API_KEY")
    # )
    embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding-LongContext",model_kwargs={"trust_remote_code":True})
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
=======
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
    embedding_2 = SpacyEmbeddings(model_name="en_core_web_sm")
    model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})
    cohere_embedding = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    chunked_documents = []

    for file_path in Path(directory_path).rglob('*.*'):
        if file_path.is_file():
            loader = TextLoader(file_path, encoding='utf8')
            data = loader.load()
            data = add_metadata(data,file_path)
            # data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0][:-2]
            chunked_documents.extend(text_splitter.split_documents(data))

<<<<<<< HEAD
    MongoDBAtlasVectorSearch.from_documents(
        chunked_documents,
        embedding=embeddings,
=======
    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        chunked_documents,
        cohere_embedding,
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
        collection=mycol,
        index_name=os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')
    )
    
<<<<<<< HEAD
load_chunk('./data/chunk/')
=======
load_chunk('E:\\thesis\\Rag4PublicSector\\data\\txt_file')
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
