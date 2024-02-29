import bs4
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
os.environ['HF_HOME'] = 'D:/RAG4'

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from trafilatura import fetch_url, extract
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, pipeline
import torch
import chainlit as cl
from chainlit.playground.providers.langchain import LangchainGenericProvider

db = None



custom_prompt_template = """Sử dụng các bối cảnh dưới đây để trả lời câu hỏi ở cuối. Trả lời chỉ từ bối cảnh đã cho. Nếu bạn không biết câu trả lời, hãy nói bạn không biết trả lời câu hỏi này.
    Bối cảnh: {context}
    Trả lời ngắn gọn.
    Câu hỏi: {question}
    """

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt



# Loading the model
def load_llm(model):
    # Load the locally downloaded model here
    llm = ChatGoogleGenerativeAI(model=model,convert_system_message_to_human=True,google_api_key="AIzaSyCSIVSP2hj6L0h-LZWCEhF5LQ6b9_jPgt4",temperature=0.01)
    
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



def create_db(link):
    loader = WebBaseLoader(link)
    docs = loader.load()
    downloaded = fetch_url(link)
    text = extract(downloaded)
    docs[0].page_content = text
    #text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    # Embed
    model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,
    model_kwargs = {"device":"cpu"})
    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    return db


def create_chain(retriever,llm):
    # Create chain
    prompt = set_custom_prompt()
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs={"prompt": prompt},)   
    return qa_chain


# chainlit code
@cl.on_chat_start
async def init():

    await cl.Message(content="Xin chào, tôi là chatbot hỗ trợ bạn trong việc thực hiện các thủ tục của các dịch vụ công. Vui lòng hãy hỏi tôi một câu hỏi.").send()
    link = None
    while link == None:
        link = await cl.AskUserMessage(
            content = "Nhập vào một link ..."
        ).send()
        
    await cl.Message(content="Đang khởi tạo, vui lòng đợi ...").send()
    # Load, chunk and index the contents of the blog.
    global db 
    db = create_db(link['content'])
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = load_llm("gemini-pro")
    qa_chain = create_chain(retriever,llm)
    # Create user session to store data
    cl.user_session.set("qa_chain", qa_chain)
    # Send response back to user
    await cl.Message(content = "Bây giờ bạn có thể hỏi!").send()

@cl.on_message # this function will be called every time a user inputs a message in the UI
async def main(message: str):

    qa_chain = cl.user_session.get("qa_chain")
    res = qa_chain.invoke(message.content)
    res_full = cl.Message(cl.Text(name="ChatGPT", content="https://chat.openai.com/c/221c00d2-bbcb-4453-94c7-cab52d47ff1a", display="inline"))
    await res_full.send()