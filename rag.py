import bs4
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['HF_HOME'] = 'D:/RAG4'

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
# from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from chainlit.playground.providers.langchain import LangchainGenericProvider
import preprocessing1


retriever = preprocessing1.retriever

custom_prompt_template1 = """
    <start_of_turn>user
    Bạn là một trợ lý ảo giúp trả lời chính xác các quy trình bằng tiếng Việt.
    Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối.

    Trả lời chỉ từ bối cảnh đã cho. Nếu bạn không biết câu trả lời, hãy nói "Tôi không biết trả lời câu hỏi này.".

    bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}
    Câu hỏi: {question}
    Trả lời đầy đủ nhất, và bao gồm các yêu cầu sau:
    - Nếu câu hỏi dạng liệt kê thông tin trong bối cảnh có trường hợp đặc biệt thì hãy nêu rõ ra các trường hợp.
    - Nếu như câu trả lời dạng bảng thì hãy in ra dạng bảng bằng markdown.<end_of_turn>
    <start_of_turn>model
    """

custom_prompt_template2 = """
    Bạn là một trợ lý ảo giúp trả lời chính xác các quy trình bằng tiếng Việt.
    Sử dụng các bối cảnh sau để trả lời câu hỏi ở cuối.

    Trả lời chỉ từ bối cảnh đã cho. Nếu bạn không biết câu trả lời, đừng cố trả lời mà hãy nói "Tôi không biết trả lời câu hỏi này.".

    bối cảnh: gồm nhiều văn bản hành chính, hãy xác định chính xác văn bản cần trích xuất thông tin. {context}
    Câu hỏi: {question}
    Trả lời đầy đủ nhất, và bao gồm các yêu cầu sau:
    - Nếu như câu trả lời dạng bảng thì hãy in ra dạng bảng bằng markdown.
    - Nếu gặp hyperlink dạng HTML "<a href="url">link text</a>", hãy thay nó sang dạng markdown [link text](url).
    """

custom_prompt_template3 = """
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
    
custom_prompt_template4 = """
    Bạn là một trợ lý ảo giúp đặt câu hỏi vấn đề liên quan tới các thủ tục dịch vụ công từ bối cảnh được cung cấp {context}, lưu ý rằng câu hỏi đặt ra phải có thông tin trả lời trong bối cảnh.
    Với {question} là số lượng câu hỏi được yêu cầu.
    """

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template2, input_variables=["context", "question"]
    )
    return prompt


# Loading the model
def load_llm(model):
    # Load the locally downloaded model here
    llm = ChatGoogleGenerativeAI(model=model,convert_system_message_to_human=True,google_api_key="AIzaSyCSIVSP2hj6L0h-LZWCEhF5LQ6b9_jPgt4",temperature=0.1)
    # llm = HuggingFaceEndpoint(repo_id=model, temperature = 0.5, max_new_tokens = 250)
    # tokenizer = AutoTokenizer.from_pretrained(model,token="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", trust_remote_code=True)
    
    # pl = pipeline(
    #     "text-generation",
    #     model=model,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     max_new_tokens=256,
    #     tokenizer=tokenizer,
    #     temperature=0.1,
    #     do_sample=True,
    #     device="cuda",
    #     token="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",
    #     trust_remote_code=True,
    # )
    
    # llm = HuggingFacePipeline(pipeline=pl)
    
    return llm

def load_llm1(model):
    # Load the locally downloaded model here
    # llm = ChatGoogleGenerativeAI(model=model,convert_system_message_to_human=True,google_api_key="AIzaSyCSIVSP2hj6L0h-LZWCEhF5LQ6b9_jPgt4",temperature=0.5)
    llm = HuggingFaceEndpoint(repo_id=model)
    # tokenizer = AutoTokenizer.from_pretrained(model,token="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", trust_remote_code=True)
    
    # pl = pipeline(
    #     "text-generation",
    #     model=model,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     max_new_tokens=256,
    #     tokenizer=tokenizer,
    #     temperature=0.1,
    #     do_sample=True,
    #     device="cuda",
    #     token="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",
    #     trust_remote_code=True,
    # )
    
    # llm = HuggingFacePipeline(pipeline=pl)
    
    return llm

# QA Model Function



# def create_db():
#     # loader = WebBaseLoader(link)
#     loader  = UnstructuredFileLoader("./data/ChiTietTTHC_1.004194.docx")
#     docs = loader.load()
#     doc_text = "\n\n".join([d.page_content for d in docs])
#     docs[0].page_content = doc_text

#     # downloaded = fetch_url(link)
#     # text = extract(downloaded)
#     # docs[0].page_content = text
    
    
#     #text splitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     documents = text_splitter.split_documents(docs)
    
    
    # Embed
    # model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    # embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,
    # model_kwargs = {"device":"cpu"})
    # db = FAISS.from_documents(documents=documents, embedding=embeddings)
    # return db


def create_conversational_chain(retriever,llm):
    # Create chain
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    messages = [
        SystemMessagePromptTemplate.from_template(custom_prompt_template3),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever, 
        memory=memory,
        get_chat_history=lambda h : h,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True)
    
    return qa_chain

def create_chain(retriever,llm):
    # Create chain
    prompt = set_custom_prompt()
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs={
                                        "prompt": prompt,
                                    },)  
    return qa_chain


# chainlit code
@cl.on_chat_start
async def init():

    await cl.Message(content="Xin chào, tôi là chatbot hỗ trợ bạn trong việc thực hiện các thủ tục của các dịch vụ công. Vui lòng hãy hỏi tôi một câu hỏi.").send()
    # link = None
    # while link == None:
    #     link = await cl.AskUserMessage(
    #         content = "Nhập vào một link ..."
    #     ).send()
        
    # await cl.Message(content="Đang khởi tạo, vui lòng đợi ...").send()
    # Load, chunk and index the contents of the blog.
    # db = create_db(link['content'])
    llm = load_llm("gemini-pro")
    qa_chain = create_conversational_chain(retriever,llm)
    # Create user session to store data
    cl.user_session.set("qa_chain", qa_chain)
    # Send response back to user
    await cl.Message(content = "Bây giờ bạn có thể hỏi!").send()

@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):

    qa_chain = cl.user_session.get("qa_chain")
    history = []
    response = qa_chain({"question": message.content,"chat_history": history})
    history.append((message.content, response))
    source_documents = response['source_documents']
    print(response)
    # Source Retrieval
    source = source_documents[0].metadata.get('source', None)
    # for i in source_documents:
    #     metadata = i.metadata
    #     source = metadata.get('source', None)
    #     name, source = source.split(" - ")
    #     sources += "["+name+"]("+source+")" + "\n"
    if source:
        if (response['answer'] != "Tôi không biết trả lời câu hỏi này."):
            elements = [
                cl.Text(name="Nguồn", content=source, display="inline")
            ]
            await cl.Message(
                content=response['answer'],
                elements=elements,
            ).send()
        else:
            res_full = cl.Message(response['answer'])
            await res_full.send()      
    else:
        res_full = cl.Message(response['answer'])
        await res_full.send()
    # elements = [
    #     cl.Text(name="Nguồn", content="["+name+"]("+source+")", display="inline")
    # ]
    # await cl.Message(
    #     content=response['result'],
    #     elements=elements,
    # ).send()