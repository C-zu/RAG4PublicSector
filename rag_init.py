from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint    
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import WebBaseLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from chainlit.playground.providers.langchain import LangchainGenericProvider
from preprocessing1 import *
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_cohere.embeddings import CohereEmbeddings

used_links = []


custom_prompt_testset = """
Bạn là một trợ lý ảo được thiết kế để hỗ trợ trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt.\n\n
Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối.
Nếu bạn không biết câu trả lời, đừng cố trả lời mà hãy nói "Tôi không biết trả lời câu hỏi này.".
Bối cảnh: gồm nhiều văn bản hành chính về dịch vụ công, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}\n\n

Câu hỏi: {question}\n\n

Yêu cầu đối với câu trả lời:\n\n
- Trả lời đúng trọng tâm câu hỏi.
"""
custom_prompt_template2 = """
Bạn là một trợ lý ảo được thiết kế để hỗ trợ trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt.\n\n
Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối, và có thể sử dụng lịch sử trò chuyện để cải thiện câu trả lời từ câu trả lời trước đó.
Hãy trả lời chỉ từ bối cảnh đã cho và nếu câu hỏi mơ hồ không bao gồm tên của một thủ tục cụ thể thì đừng trả lời mà hãy nói "Hãy cung cấp thêm thông tin về câu hỏi" mà đừng cố gắng trả lời.
Nếu bạn không biết câu trả lời, đừng cố trả lời mà hãy nói "Tôi không biết trả lời câu hỏi này.".
Bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}\n\n

Câu hỏi: {question}\n\n
Lịch sử trò chuyện: {chat_history}

Yêu cầu đối với câu trả lời:\n\n
- Nếu gặp hyperlink dạng HTML "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url), thay thế link text thành văn bản tương ứng.\n
- Nếu câu trả lời có dạng markdown, hãy đưa toàn bộ về dạng bảng và trả lời.\n
- Nếu câu trả lời chứa thông tin từ nhiều bảng, vui lòng in đầy đủ tất cả các bảng ra.\n
- Nếu câu trả lời không phải là bảng, vui lòng trả lời bình thường.\n
- Nếu gặp các câu hỏi về số lượng, hãy đi thẳng vào trả lời câu hỏi về số lượng đó.\n
- Nếu được hỏi về tên thủ tục, hãy trả lời tên thủ tục gần nhất đã trả lời trước đó.\n
"""
custom_prompt_template_question = """
    Bạn là một trợ lý ảo giúp đặt câu hỏi vấn đề liên quan tới các thủ tục dịch vụ công từ bối cảnh được cung cấp {context}, lưu ý rằng câu hỏi đặt ra phải có thông tin trả lời trong bối cảnh.
    Với {question} là số lượng câu hỏi được yêu cầu.
    """
# Define
txt_data = './data/txt_file'
# embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=os.getenv("GOOGLE_API_KEY"), task_type="retrieval_document")
# embedding_2 = SpacyEmbeddings(model_name="en_core_web_sm")
# model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
cohere_embedding = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)
# hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})

# Init
db = VectorDatabase(txt_data, cohere_embedding, "pickle")
mongo_db = VectorDatabase(txt_data, cohere_embedding, "mongo")
qdrant_db = VectorDatabase(txt_data, cohere_embedding, "qdrant")

# Retrievers
index_retriever = Retriever("indexing", "bm25", db)
mongo_retriever = Retriever("vector_search", cohere_embedding, mongo_db)
qdrant_retriever = Retriever("vector_search", cohere_embedding, qdrant_db)

# Resemble
compressed_retriever_qdrant = CompressedRetriever(index_retriever.get_retriever(), qdrant_retriever.get_retriever(), [0.5,0.5])
compressed_retriever_mongo = CompressedRetriever(index_retriever.get_retriever(), mongo_retriever.get_retriever(), [0.5,0.5])
compressed_retriever_qm = CompressedRetriever(compressed_retriever_qdrant.get_retriever(), compressed_retriever_mongo.get_retriever(), [0.5,0.5])
retriever = compressed_retriever_qm.re_ranking()