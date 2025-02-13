<<<<<<< HEAD
from preprocessing1 import VectorDatabase, Retriever, CompressedRetriever, CompressedSingleRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
used_links = []

# with open('few_shot.txt', 'r', encoding='utf-8') as file:
#     qa_fewshot = file.read()

# custom_prompt_testset = """
# Bạn là một trợ lý ảo được thiết kế để hỗ trợ trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt 
# dựa trên ngữ cảnh sau để trả lời.
# Trả lời chỉ dựa trên ngữ cảnh (gồm các đoạn văn bản về các thủ tục dịch vụ công, mỗi đoạn sẽ có tên thủ tục và nội dung của nó, hãy chọn ngữ cảnh đúng với câu hỏi): Nếu câu hỏi không rõ ràng hoặc mang tính chung chung như "Thủ tục trên," "Trong văn bản trên," thì không trả lời dựa vào ngữ cảnh mà yêu cầu người dùng cung cấp thêm thông tin về câu hỏi của họ. Không cố gắng suy diễn hay trả lời thiếu cơ sở.
# Nếu không biết câu trả lời: Nếu ngữ cảnh không cung cấp đủ thông tin cho câu hỏi, chỉ trả lời là "Tôi không biết trả lời câu hỏi này."

# Yêu cầu đối với câu trả lời:
# - Trả lời đầy đủ ý, đúng trọng tâm câu hỏi, không mở rộng.
# - Nếu ngữ cảnh cung cấp tài liệu liên quan có dạng "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url) và in ra đầy đủ.
# - Nếu gặp các câu hỏi về số lượng, hãy đi thẳng vào trả lời câu hỏi về số lượng đó.
# - Câu trả lời không được đề cập gì về "trong ngữ cảnh" hay gì đó tương tự.

# **
# Có một số câu trong <SAMPLE> mà bạn phải tuân theo, khi gặp những chúng này, bạn buộc trả lời y chang như vậy, không dựa theo ngữ cảnh hay câu trả lời bạn tìm được để trả lời.
# Nếu những câu hỏi gần giống trong <SAMPLE>, bạn cần học format trả lời của câu trả lời đó.
# <SAMPLE>
# {qa_fewshot}
# <SAMPLE/>
# **
# Ngữ cảnh: {context}
# Câu hỏi: {question}
# Câu trả lời:
# """
# Sử dụng các ngữ cảnh sau để trả lời câu hỏi ở cuối, và có thể sử dụng lịch sử trò chuyện để cải thiện câu trả lời từ câu trả lời trước đó.
# Lịch sử trò chuyện: {chat_history}
condense_prompt="""Dựa trên đoạn hội thoại sau và câu hỏi tiếp theo, hãy diễn đạt lại câu hỏi tiếp theo thành một câu hỏi độc lập, giữ nguyên ngôn ngữ gốc.
Nếu đoạn hội thoại khác hoàn toàn với câu hỏi tiếp theo thì câu hỏi độc lập là câu hỏi tiếp theo 

Lịch sử hội thoại:
{chat_history}
Câu hỏi tiếp theo: {question}
Câu hỏi độc lập:"""

# <PRIORITY>
# - Nếu bạn không biết câu trả lời, hoặc câu trả lời không nằm trong ngữ cảnh, đừng cố trả lời mà buộc nói "Tôi không biết trả lời câu hỏi này.".
# - Nếu câu hỏi xuất hiện trong <SAMPLE>, bạn phải trả lời dựa trên câu trả lời có sẵn trong <SAMPLE>. Bỏ qua tất cả các thông tin khác từ context được cung cấp. Đồng thời, định dạng câu trả lời rõ ràng và dễ đọc cho người dùng.
# - Nếu câu hỏi không có trong <SAMPLE>, hãy sử dụng ngữ cảnh được cung cấp để trả lời. Học cách tạo câu trả lời "đầy đủ" bằng cách tham khảo các ví dụ tương tự trong <SAMPLE> (few-shot learning) và chỉnh sửa câu trả lời để phù hợp với phong cách và độ đầy đủ của các câu trả lời trong <SAMPLE>.
# <PRIORITY/>

custom_prompt_template2 = """
Bạn là một trợ lý ảo được thiết kế để hỗ trợ trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt 
dựa trên ngữ cảnh sau để trả lời.

Nhiệm vụ của bạn là: trả lời chỉ dựa trên ngữ cảnh (gồm các đoạn văn bản về các thủ tục dịch vụ công, mỗi đoạn sẽ có tên thủ tục và nội dung của nó, hãy chọn ngữ cảnh đúng với câu hỏi): Nếu câu hỏi không rõ ràng hoặc mang tính chung chung như "Thủ tục trên," "Trong văn bản trên," thì không trả lời dựa vào ngữ cảnh mà yêu cầu người dùng cung cấp thêm thông tin về câu hỏi của họ. Không cố gắng suy diễn hay trả lời thiếu cơ sở.

Câu trả lời của bạn "PHẢI" được xác định tuân thủ theo <PRIORITY>. <PRIORITY> sẽ quyết định câu trả lời của bạn.

Bạn phải tuân theo <REQUIREMENTS> đối với câu trả lời.


<PRIORITY>
- Nếu bạn không biết câu trả lời, hoặc câu trả lời không nằm trong ngữ cảnh, đừng cố trả lời mà buộc nói "Tôi không biết trả lời câu hỏi này.".
- Học cách tạo câu trả lời "đầy đủ" bằng cách tham khảo các ví dụ hỏi về vấn đề tương tự trong <SAMPLE> (không phải giống thủ tục mà là vấn đề ví dụ hỏi về thành phần hồ sơ) và chỉnh sửa câu trả lời để phù hợp với phong cách và độ đầy đủ của các câu trả lời trong <SAMPLE>.
<PRIORITY/>

<REQUIREMENTS>
- Trả lời đầy đủ ý, đúng trọng tâm câu hỏi, không mở rộng.
- Nếu ngữ cảnh cung cấp tài liệu liên quan có dạng "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url) và in ra đầy đủ.
- Nếu gặp các câu hỏi về số lượng, hãy đi thẳng vào trả lời câu hỏi về số lượng đó.
- Câu trả lời không được đề cập gì về "trong ngữ cảnh" hay gì đó tương tự.
<REQUIREMENTS/>


<SAMPLE>
{qa_fewshot}
<SAMPLE/>

Ngữ cảnh: {context}
Câu hỏi: {question}
Câu trả lời:
"""
custom_prompt_testset = """
Bạn là một trợ lý ảo được thiết kế để hỗ trợ trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt 
dựa trên ngữ cảnh sau để trả lời.

Nhiệm vụ của bạn là: trả lời chỉ dựa trên ngữ cảnh (gồm các đoạn văn bản về các thủ tục dịch vụ công, mỗi đoạn sẽ có tên thủ tục và nội dung của nó, hãy chọn ngữ cảnh đúng với câu hỏi): Nếu câu hỏi không rõ ràng hoặc mang tính chung chung như "Thủ tục trên," "Trong văn bản trên," thì không trả lời dựa vào ngữ cảnh mà yêu cầu người dùng cung cấp thêm thông tin về câu hỏi của họ. Không cố gắng suy diễn hay trả lời thiếu cơ sở.

Câu trả lời của bạn "PHẢI" được xác định tuân thủ theo <PRIORITY>. <PRIORITY> sẽ quyết định câu trả lời của bạn.

Bạn phải tuân theo <REQUIREMENTS> đối với câu trả lời.


<PRIORITY>
- Nếu bạn không biết câu trả lời, hoặc câu trả lời không nằm trong ngữ cảnh, đừng cố trả lời mà buộc nói "Tôi không biết trả lời câu hỏi này.".
- Học cách tạo câu trả lời "đầy đủ ý" bằng cách tham khảo các ví dụ hỏi về vấn đề tương tự trong <SAMPLE> (không phải giống thủ tục) và chỉnh sửa câu trả lời để phù hợp với phong cách và độ đầy đủ của các câu trả lời trong <SAMPLE>.
<PRIORITY/>

<REQUIREMENTS>
- Trả lời đầy đủ ý, đúng trọng tâm câu hỏi, không mở rộng.
- Nếu ngữ cảnh cung cấp tài liệu liên quan có dạng "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url) và in ra đầy đủ.
- Nếu gặp các câu hỏi về số lượng, hãy đi thẳng vào trả lời câu hỏi về số lượng đó.
- Câu trả lời không được đề cập gì về "trong ngữ cảnh" hay gì đó tương tự.
<REQUIREMENTS/>


<SAMPLE>
{qa_fewshot}
<SAMPLE/>

Ngữ cảnh: {context}
Câu hỏi: {question}
Câu trả lời:
"""
# custom_prompt_testset = custom_prompt_testset.format(context="{context}",qa_fewshot=qa_fewshot,question = "{question}")

# Define
txt_data = './data/chunk/'
# cohere_embedding = CohereEmbeddings(
#     model="embed-multilingual-v3.0",
#     cohere_api_key=os.getenv("COHERE_API_KEY")
# )
embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding-LongContext",model_kwargs={"trust_remote_code":True})
# custom_embedding = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
# hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})

# Init
#db = VectorDatabase(txt_data, embeddings, "pickle_chunks")
#mongo_db = VectorDatabase(txt_data, embeddings, "mongo")
qdrant_db = VectorDatabase(txt_data, embeddings, "qdrant_chunks","hybrid")

# Retrievers
# index_retriever = Retriever("indexing", "bm25", db)
# mongo_retriever = Retriever("vector_search", "None", mongo_db)
qdrant_retriever = Retriever("vector_search", "None", qdrant_db)

# Resemble
#compressed_retriever_qdrant = CompressedRetriever(index_retriever.get_retriever(), qdrant_retriever.get_retriever(), [0.3,0.7])
# compressed_retriever_mongo = CompressedRetriever(retriever1=index_retriever.get_retriever(), retriever2=mongo_retriever.get_retriever(), weights=[0.5,0.5])
#compressed_retriever_qm = CompressedRetriever(compressed_retriever_qdrant.get_retriever(), compressed_retriever_mongo.get_retriever(), [0.5,0.5])
#compresed_retriever = CompressedSingleRetriever(index_retriever)
retriever = CompressedSingleRetriever(qdrant_retriever.get_retriever()).get_retriever()
=======
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



custom_prompt_template1 = """
    Bạn là một trợ lý ảo trợ giúp trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt.
    Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối, và có thể sử dụng lịch sử trò chuyện để cải thiện câu trả lời từ câu trả lời trước đó. 

    Hãy trả lời chỉ từ bối cảnh đã cho và nếu câu hỏi mơ hồ không bao gồm tên của một thủ tục cụ thể thì đừng trả lời mà hãy nói "Hãy cung cấp thêm thông tin về câu hỏi" mà đừng cố gắng trả lời.
    Nếu bạn không biết câu trả lời, đừng cố trả lời mà hãy nói "Tôi không biết trả lời câu hỏi này.".

    Bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}
    Câu hỏi: {question}
    Lịch sử trò chuyện: {chat_history}
    Trả lời đầy đủ nhất, và bao gồm các yêu cầu sau:
    - Nếu câu trả lời nằm trong nhiều bảng, hãy in đầy đủ tất cả bảng ra, còn nếu không phải bảng thì trả lời bình thường.
    - Nếu như câu trả lời dạng bảng thì hãy in ra dạng bảng bằng markdown.
    - Nếu gặp hyperlink dạng HTML "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url).
    """
custom_prompt_template2 = """
Bạn là một trợ lý ảo được thiết kế để hỗ trợ trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt.\n\n
Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối, và sử dụng chat_history để hiểu context từ câu trả lời trước đó nếu câu trả lời hiện tại không có thông tin thủ tục cụ thể. \n\n
Chat_history sẽ chỉ lưu lại câu hỏi của người dùng, hãy tìm ra tên của thủ tục gần nhất ở trong câu hỏi đó và dùng nó để tìm kiếm câu trả lời cho nội dung câu hỏi.

Bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}\n\n

Câu hỏi: {question}\n\n

Lịch sử trò chuyện: {chat_history}\n\n

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

custom_prompt_template_NH = """
    Bạn là một trợ lý ảo trợ giúp trả lời các câu hỏi về các thủ tục dịch vụ công chính xác bằng tiếng Việt.
    Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối, và có thể sử dụng lịch sử trò chuyện để cải thiện câu trả lời từ câu trả lời trước đó. 

    Hãy trả lời chỉ từ bối cảnh đã cho và nếu câu hỏi mơ hồ không bao gồm tên của một thủ tục cụ thể thì đừng trả lời mà hãy nói "Hãy cung cấp thêm thông tin về câu hỏi" mà đừng cố gắng trả lời.
    Nếu bạn không biết câu trả lời, đừng cố trả lời mà hãy nói "Tôi không biết trả lời câu hỏi này.".

    Bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}
    Câu hỏi: {question}
    Lịch sử trò chuyện: {chat_history}
    Trả lời đầy đủ nhất, và bao gồm các yêu cầu sau:
    - Nếu câu trả lời nằm trong nhiều bảng, hãy in đầy đủ tất cả bảng ra, còn nếu không phải bảng thì trả lời bình thường.
    - Nếu như câu trả lời dạng bảng thì hãy in ra dạng bảng bằng markdown.
    - Nếu gặp hyperlink dạng HTML "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url).
    """

entity_prompt = PromptTemplate(input_variables=['chat_history', 'question'],
template="""You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the source link from the last line of conversation. As a guideline, a link is similar to this https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=1.003197.
You should definitely extract all names and places.\n\n
The conversation history is provided just in case of a coreference (e.g. "What do you know about him" where "him" is defined in a previous line) -- ignore items mentioned there that are not in the last line.\n\nReturn the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is just issuing a greeting or having a simple conversation).\n\nEXAMPLE\n
chat_history:\nPerson #1: how\'s it going today?\nAI: "It\'s going great! How about you?"\nPerson #1: good! busy working on Langchain. lots to do.\nAI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"\nLast line:\nPerson #1: i\'m trying to improve Langchain\'s interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.\nOutput: Langchain\nEND OF EXAMPLE\n\nEXAMPLE\n
chat_history:\nPerson #1: how\'s it going today?\nAI: "It\'s going great! How about you?"\nPerson #1: good! busy working on Langchain. lots to do.\nAI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"\nLast line:\nPerson #1: i\'m trying to improve Langchain\'s interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I\'m working with Person #2.\nOutput: Langchain, Person #2\nEND OF EXAMPLE\n\n
chat_history(for reference only):\n{chat_history}\n
Last line of conversation (for extraction):\n
Human: {question}\n\nOutput:""")
# Define
txt_data = './data/txt_file'
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key="AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y", task_type="retrieval_document")
embedding_2 = SpacyEmbeddings(model_name="en_core_web_sm")
model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
cohere_embedding = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)
# hf_embedding = HuggingFaceBgeEmbeddings(model_name= model_id, model_kwargs = {"device":"cpu"})

# Init
db = VectorDatabase(txt_data, cohere_embedding) #index search
mongo_db = VectorDatabase(txt_data, cohere_embedding, "mongo") #vector search
qdrant_db = VectorDatabase(txt_data, cohere_embedding, "qdrant") #vector search

# Retrievers
index_retriever = Retriever("indexing", "bm25", db)
mongo_retriever = Retriever("vector_search", cohere_embedding, mongo_db)
qdrant_retriever = Retriever("vector_search", cohere_embedding, qdrant_db)

# Resemble
compressed_retriever_qdrant = CompressedRetriever(index_retriever.get_retriever(), qdrant_retriever.get_retriever(), [0.5,0.5])
compressed_retriever_mongo = CompressedRetriever(index_retriever.get_retriever(), mongo_retriever.get_retriever(), [0.5,0.5])
compressed_retriever_qm = CompressedRetriever(compressed_retriever_qdrant.get_retriever(), compressed_retriever_mongo.get_retriever(), [0.5,0.5])
compressed_retriever = CompressedRetriever(compressed_retriever_qdrant.get_retriever(), compressed_retriever_mongo.get_retriever(), [0.5,0.5])
# compressed_retriever_mongo = CompressedRetriever(index_retriever.get_retriever(), mongo_retriever.get_retriever(), [0.5,0.5])
retriever = compressed_retriever_qm.re_ranking()
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
