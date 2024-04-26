# custom_prompt_template1 = """
#     <start_of_turn>user
#     Bạn là một trợ lý ảo giúp trả lời chính xác các quy trình bằng tiếng Việt.
#     Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối.

#     Trả lời chỉ từ bối cảnh đã cho. Nếu bạn không biết câu trả lời, hãy nói "Tôi không biết trả lời câu hỏi này.".

#     bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}
#     Câu hỏi: {question}
#     Trả lời đầy đủ nhất, và bao gồm các yêu cầu sau:
#     - Nếu câu hỏi dạng liệt kê thông tin trong bối cảnh có trường hợp đặc biệt thì hãy nêu rõ ra các trường hợp.
#     - Nếu như câu trả lời dạng bảng thì hãy in ra dạng bảng bằng markdown.<end_of_turn>
#     <start_of_turn>model
#     """
# custom_prompt_template2 = """
#     Bạn là một trợ lý ảo giúp trả lời chính xác các quy trình bằng tiếng Việt.
#     Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối.

#     Trả lời chỉ từ bối cảnh đã cho. Nếu bạn không biết câu trả lời, đừng cố trả lời mà hãy nói "Tôi không biết trả lời câu hỏi này.".

#     bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}
#     Câu hỏi: {question}
#     Trả lời đầy đủ nhất, và bao gồm các yêu cầu sau:
#     - Nếu như câu trả lời dạng bảng thì hãy in ra dạng bảng bằng markdown.
#     - Nếu gặp hyperlink dạng HTML "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url).
#     """

# def create_conversational_chain(retriever,llm):
#     # Create chain
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
#     messages = [
#         SystemMessagePromptTemplate.from_template(custom_prompt_template5),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ]
#     qa_prompt = ChatPromptTemplate.from_messages(messages)

#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm, 
#         retriever, 
#         memory=memory,
#         get_chat_history=lambda h : h,
#         combine_docs_chain_kwargs={"prompt": qa_prompt},
#         return_source_documents=True)
    
#     return qa_chain


# def load_llm(model):
#     # Load the locally downloaded model here
#     safe = [
#     {
#         "category": "HARM_CATEGORY_HARASSMENT",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_HATE_SPEECH",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#         "threshold": "BLOCK_NONE",
#     },
#     {
#         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#         "threshold": "BLOCK_NONE",
#     },
# ]
#     llm = ChatGoogleGenerativeAI(model=model,convert_system_message_to_human=True,google_api_key="AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y", __key__=safe)
#     # llm = HuggingFaceEndpoint(repo_id=model, temperature = 0.5, max_new_tokens = 250)
#     # tokenizer = AutoTokenizer.from_pretrained(model,token="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", trust_remote_code=True)
    
#     # pl = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     model_kwargs={"torch_dtype": torch.bfloat16},
#     #     max_new_tokens=256,
#     #     tokenizer=tokenizer,
#     #     temperature=0.1,
#     #     do_sample=True,
#     #     device="cuda",
#     #     token="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",
#     #     trust_remote_code=True,
#     # )
    
#     # llm = HuggingFacePipeline(pipeline=pl)
    
#     return llm

# def llm_test():
#     genai.configure(api_key='AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y')
#     model = genai.GenerativeModel('models/gemini-pro')
#     return model

# def load_llm1(model):
#     # Load the locally downloaded model here
#     # llm = ChatGoogleGenerativeAI(model=model,convert_system_message_to_human=True,google_api_key="AIzaSyCSIVSP2hj6L0h-LZWCEhF5LQ6b9_jPgt4",temperature=0.5)
#     llm = HuggingFaceEndpoint(repo_id=model)
#     # tokenizer = AutoTokenizer.from_pretrained(model,token="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", trust_remote_code=True)
    
#     # pl = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     model_kwargs={"torch_dtype": torch.bfloat16},
#     #     max_new_tokens=256,
#     #     tokenizer=tokenizer,
#     #     temperature=0.1,
#     #     do_sample=True,
#     #     device="cuda",
#     #     token="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",
#     #     trust_remote_code=True,
#     # )
    
#     # llm = HuggingFacePipeline(pipeline=pl)
    
#     return llm

# # QA Model Function



# # def create_db():
# #     # loader = WebBaseLoader(link)
# #     loader  = UnstructuredFileLoader("./data/ChiTietTTHC_1.004194.docx")
# #     docs = loader.load()
# #     doc_text = "\n\n".join([d.page_content for d in docs])
# #     docs[0].page_content = doc_text

# #     # downloaded = fetch_url(link)
# #     # text = extract(downloaded)
# #     # docs[0].page_content = text
    
    
# #     #text splitter
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# #     documents = text_splitter.split_documents(docs)
    
    
#     # Embed
#     # model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
#     # embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,
#     # model_kwargs = {"device":"cpu"})
#     # db = FAISS.from_documents(documents=documents, embedding=embeddings)
#     # return db
    
#     return qa_chain

# def create_chain(retriever,llm):
#     # Create chain
#     prompt = set_custom_prompt()
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                   chain_type="stuff",
#                                   retriever=retriever,
#                                   return_source_documents=True,
#                                   chain_type_kwargs={
#                                         "prompt": prompt,
#                                     },)  
#     return qa_chain


# Chainlit

# chainlit code
# @cl.on_chat_start
# async def init():

#     await cl.Message(content="Xin chào, tôi là chatbot hỗ trợ bạn trong việc thực hiện các thủ tục của các dịch vụ công. Vui lòng hãy hỏi tôi một câu hỏi.").send()
#     # link = None
#     # while link == None:
#     #     link = await cl.AskUserMessage(
#     #         content = "Nhập vào một link ..."
#     #     ).send()
        
#     # await cl.Message(content="Đang khởi tạo, vui lòng đợi ...").send()
#     # Load, chunk and index the contents of the blog.
#     # db = create_db(link['content'])
#     llm = load_llm("gemini-1.0-pro")
#     qa_chain = create_conversational_chain(retriever,llm)
#     # Create user session to store data
#     cl.user_session.set("qa_chain", qa_chain)
#     # Send response back to user
#     await cl.Message(content = "Bây giờ bạn có thể hỏi!").send()

# @cl.on_message # this function will be called every time a user inputs a message in the UI
# async def main(message: str):

#     qa_chain = cl.user_session.get("qa_chain")
#     history = []
#     response = qa_chain({"question": message.content,"chat_history": history})
#     history.append((message.content, response))
#     source_documents = response['source_documents']
#     print(response)
#     # Source Retrieval
#     source = source_documents[0].metadata.get('source', None)
#     # for i in source_documents:
#     #     metadata = i.metadata
#     #     source = metadata.get('source', None)
#     #     name, source = source.split(" - ")
#     #     sources += "["+name+"]("+source+")" + "\n"
#     if source:
#         if (response['answer'] != "Tôi không biết trả lời câu hỏi này."):
#             elements = [
#                 cl.Text(name="Nguồn", content=source, display="inline")
#             ]
#             await cl.Message(
#                 content=response['answer'],
#                 elements=elements,
#             ).send()
#         else:
#             res_full = cl.Message(response['answer'])
#             await res_full.send()      
#     else:
#         res_full = cl.Message(response['answer'])
#         await res_full.send()
#     # elements = [
#     #     cl.Text(name="Nguồn", content="["+name+"]("+source+")", display="inline")
#     # ]
#     # await cl.Message(
#     #     content=response['result'],
#     #     elements=elements,
#     # ).send()


#  ---------------------------------------------This are from preprocessing1.py--------------------------------------------
# def load_chunk(directory_path):
#     model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     embeddings = HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"})
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

#     chunked_documents = []

#     for file_path in Path(directory_path).rglob('*.*'):
#         if file_path.is_file():
#             loader = TextLoader(file_path, encoding='utf8')
#             data = loader.load()
#             data[0].metadata['source'] = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-thu-tuc-hanh-chinh.html?ma_thu_tuc=" + os.path.splitext(os.path.basename(file_path))[0]
#             chunked_documents.extend(text_splitter.split_documents(data))
#     bm25_retriever = BM25Retriever.from_documents(chunked_documents, k = 5)
    
    
#     Qdrant.from_documents(
#         chunked_documents,
#         embeddings,
#         path="./Qdrant",  # Local mode with in-memory storage only
#         collection_name="my_documents",
#     )
    
#     # vectorstore = MongoDBAtlasVectorSearch.from_documents(
#     # chunked_documents,
#     # embeddings,
#     # collection=COLLECTION_NAME,
#     # index_name=DB_NAME,
#     # )
    
#     with open('./data/bm25_retriever.pkl', 'wb') as f:
#         pickle.dump(bm25_retriever, f)
    

# def load_db():
#     # model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     # embeddings = HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"})
    
#     embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
#     # client = qdrant_client.QdrantClient(
#     #     path="./Qdrant",
#     # )
#     # docsearch = Qdrant(
#     #     client=client, collection_name="my_documents", embeddings=embeddings
#     # )
#     with open('./data/bm25_retriever.pkl', 'rb') as f:
#         bm25_retriever = pickle.load(f)
    
#     vector_search = MongoDBAtlasVectorSearch.from_connection_string(
#         MONGODB_ATLAS_CLUSTER_URI,
#         DB_NAME + "." + COLLECTION_NAME,
#         embeddings,
#         index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
#     )
    
#     mongodb_retriever = vector_search.as_retriever(search_kwargs={"k": 10})
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, mongodb_retriever], weights=[0.5, 0.5])

#     # Cohere Reranker
#     compressor = CohereRerank(user_agent='langchain')
#     compression_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=ensemble_retriever,
#     )

#     return compression_retriever