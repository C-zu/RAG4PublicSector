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
db = VectorDatabase(txt_data, embeddings, "pickle_chunks")
#mongo_db = VectorDatabase(txt_data, embeddings, "mongo")
qdrant_db = VectorDatabase(txt_data, embeddings, "qdrant_chunks")

# Retrievers
index_retriever = Retriever("indexing", "bm25", db)
# mongo_retriever = Retriever("vector_search", "None", mongo_db)
qdrant_retriever = Retriever("vector_search", "None", qdrant_db)

# Resemble
compressed_retriever_qdrant = CompressedRetriever(index_retriever.get_retriever(), qdrant_retriever.get_retriever(), [0.3,0.7])
# compressed_retriever_mongo = CompressedRetriever(retriever1=index_retriever.get_retriever(), retriever2=mongo_retriever.get_retriever(), weights=[0.5,0.5])
#compressed_retriever_qm = CompressedRetriever(compressed_retriever_qdrant.get_retriever(), compressed_retriever_mongo.get_retriever(), [0.5,0.5])
#compresed_retriever = CompressedSingleRetriever(index_retriever)
retriever = CompressedSingleRetriever(compressed_retriever_qdrant.get_retriever()).get_retriever()