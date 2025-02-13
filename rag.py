from typing import Any
import rag_init
<<<<<<< HEAD
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_cohere.chat_models import ChatCohere
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import pickle
from langchain_community.retrievers.bm25 import BM25Retriever
=======
import chainlit as cl
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationKGMemory
from langchain.memory import ConversationBufferMemory

>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
class Prompt():
    def __init__(self, template=None) -> None:
        self.prompt_template = rag_init.custom_prompt_template2
    
        if template is not None:
            self.prompt_template = template
    def set_prompt(self):
<<<<<<< HEAD
        prompt = PromptTemplate(
=======
        prompt = rag_init.PromptTemplate(
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
            template=self.prompt_template, input_variables=["context", "question"]
        )
        return prompt
    def __getattribute__(self, name: str):
        return super().__getattribute__(name)

class LLM():
    def __init__(self, llm = None) -> None:
<<<<<<< HEAD
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=self.safety_settings, temperature=0.1, stream=True,callbacks=self.callbacks)

        if llm is not None:
            if llm == "command-r-plus":
                self.llm = ChatCohere(
                    model="command-r-plus",
                    temperature=0.1,
                    max_tokens=None,
                    timeout=None,
                    streaming=True,
                    callbacks=self.callbacks
                )
            elif llm == "command-r-plus-no-streaming":
                self.llm = ChatCohere(
                    model="command-r-plus",
                    temperature=0.1,
                    max_tokens=None,
                    timeout=None,
                )
            elif llm == "llama3.1":
                self.llm = ChatOpenAI(
                    model="Meta-Llama-3.1-70B-Instruct",    
                    openai_api_base="https://api.sambanova.ai/v1",
                    openai_api_key="95990e56-0bdb-4bbb-baf5-49dd62b2387b",
                )
            elif llm == "deepseek-r1":
                self.llm = ChatOpenAI(
                    model="deepseek/deepseek-r1:free",    
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key="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",
                )
                self.llm = self.llm.bind(temperature=1.3)
            else:
                self.llm = ChatGoogleGenerativeAI(model=llm,convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=self.safety_settings, temperature=0.1, stream=True,callbacks=self.callbacks)
=======
        self.safe_settings = [{
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
            },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
        ]
        self.llm = rag_init.ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,google_api_key="AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y", __key__=self.safe_settings, temperature=0.1)

        if llm is not None:
            if llm == "gemini-1.0-pro":
                self.llm = rag_init.ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,google_api_key="AIzaSyD3NCZLaMXUpG1UvStJMN8eYB1QeleOg6Y", __key__=self.safe_settings, temperature=0.1)
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
    def __getattribute__(self, any: str) -> Any:
        return super().__getattribute__(any)

class RAG():
    def __init__(self, prompt=None, llm = None, retriever = None) -> None:
        self.chain = None
        self.prompt = rag_init.custom_prompt_template2
<<<<<<< HEAD
        self.condense_prompt = PromptTemplate.from_template(rag_init.condense_prompt)
        self.llm = LLM(llm=llm)
        self.retriever = rag_init.retriever
        self.fewshot_retriever = None
        self.fewshot_check = False
        if os.path.exists("fewshot_db.pkl") and os.path.getsize("fewshot_db.pkl") > 0:
            self.fewshot_check = True
            with open("fewshot_db.pkl", 'rb') as file:
                fewshot_db = pickle.load(file)
            self.fewshot_retriever = BM25Retriever.from_documents(documents=fewshot_db,k = 5)
        if prompt is not None:
            self.prompt = prompt

        if llm is not None:
            pass
        
        if retriever is not None:
            self.retriever = retriever      
        # Khởi tạo memory ban đầu, giữ nguyên suốt quá trình
        self.message_history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(
            llm=self.llm.llm,
            memory_key="chat_history",
            input_key='question',
            output_key="answer",
            chat_memory=self.message_history,
            return_messages=True,
        )

    # def _initialize_chain(self):
    #     """Khởi tạo `ConversationalRetrievalChain` một lần duy nhất"""
    #     self.chain = ConversationalRetrievalChain.from_llm(
    #         self.llm.llm,
    #         self.retriever,
    #         condense_question_prompt=self.condense_prompt,
    #         condense_question_llm=self.llm.llm,
    #         memory=self.memory,
    #         return_source_documents=True,
    #     )

    def update_prompt(self, query):
        """Cập nhật prompt mà không khởi tạo lại `self.chain`"""
        fewshot_prompt = ""
        if self.fewshot_check:
            fewshot_subset = self.fewshot_retriever.get_relevant_documents(query)
            for fewshot_qa in fewshot_subset:
                print(fewshot_qa.page_content+"\n")
                fewshot_prompt += f"""Câu hỏi: "{fewshot_qa.page_content}"\nCâu trả lời: "{fewshot_qa.metadata["answer"]}"\n\n"""

        self.prompt = self.prompt.format(context="{context}", qa_fewshot=fewshot_prompt, question="{question}")
        qa_prompt = Prompt(template=self.prompt).set_prompt()
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm.llm, 
            self.retriever,
            condense_question_prompt=self.condense_prompt,
            condense_question_llm= self.llm.llm,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            memory=self.memory,
            return_source_documents=True,
        )
=======
        self.llm = LLM("gemini-pro")
        self.retriever = rag_init.retriever
        
        if prompt is not None:
            self.prompt = prompt

        if llm is not None:
            pass
        
        if retriever is not None:
            self.retriever = retriever

        memory=ConversationBufferWindowMemory(memory_key="chat_history", output_key="answer", k = 5)        
        messages = [
            rag_init.SystemMessagePromptTemplate.from_template(self.prompt),
            rag_init.HumanMessagePromptTemplate.from_template("{question}")
        ]
        qa_prompt = rag_init.ChatPromptTemplate.from_messages(messages)

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm.llm, 
            self.retriever, 
            memory=memory,
            get_chat_history=lambda h : h,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True)
        
>>>>>>> 84722205a9bd9bc4b75930f8856c4fdda5fd1ebd
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)
    
    def __str__(self):
        return f"RAG Object with prompt: {self.prompt}, LLM: {self.llm}, Retriever: {self.retriever}"