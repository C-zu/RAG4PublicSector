# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain_community.llms import HuggingFaceEndpoint    
# from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.llms import HuggingFacePipeline
# from langchain_community.document_loaders import WebBaseLoader, UnstructuredFileLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import torch
# from langchain.memory import ConversationBufferMemory
# import chainlit as cl
# from chainlit.playground.providers.langchain import LangchainGenericProvider
# from preprocessing1 import *
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_cohere.embeddings import CohereEmbeddings
from preprocessing1 import VectorDatabase
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

txt_data = './data/chunk/'
# embedding = GoogleGenerativeAIEmbeddings(model="models/text-multilingual-embedding-002",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
# embedding_2 = SpacyEmbeddings(model_name="xx_ent_wiki_sm")
# model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
# cohere_embedding = CohereEmbeddings(
#     model="embed-multilingual-v3.0",
#     cohere_api_key=os.getenv("COHERE_API_KEY")
# )
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding-LongContext",model_kwargs={"trust_remote_code":True})
# custom_embedding = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
db = VectorDatabase(txt_data, embeddings,"chunks.pkl")
# db = VectorDatabase(txt_data, embeddings, "qdrant_create")