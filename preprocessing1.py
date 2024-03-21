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
os.environ['COHERE_API_KEY'] = '7THPmgAjpjThSPmqNcpcER8T4h1FJNgZNEzpLGzv'


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
    with open('./data/bm25_retriever.pkl', 'wb') as f:
        pickle.dump(bm25_retriever, f)
    

def load_qdrant():
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"})
    client = qdrant_client.QdrantClient(
        path="./Qdrant",
    )
    docsearch = Qdrant(
        client=client, collection_name="my_documents", embeddings=embeddings
    )
    with open('./data/bm25_retriever.pkl', 'rb') as f:
        bm25_retriever = pickle.load(f)
    
    qdrant_retriever = docsearch.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, qdrant_retriever], weights=[0.5, 0.5])

    # Cohere Reranker
    compressor = CohereRerank(user_agent='langchain')
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever

txt_path = './data/txt_file'
    
retriever = load_qdrant()