from typing import Any
import rag_init
import chainlit as cl
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_cohere.chat_models import ChatCohere
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
class Prompt():
    def __init__(self, template=None) -> None:
        self.prompt_template = rag_init.custom_prompt_template2
    
        if template is not None:
            self.prompt_template = template
    def set_prompt(self):
        prompt = rag_init.PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )
        return prompt
    def __getattribute__(self, name: str):
        return super().__getattribute__(name)

class LLM():
    def __init__(self, llm = None) -> None:
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=self.safety_settings, temperature=0.1, stream=True,callbacks=self.callbacks)

        if llm is not None:
            if llm == 'llama3' or llm =='gemma2':
                self.llm = ChatOllama(model=llm, streaming=True,callbacks=self.callbacks,temperature=0.2)
            elif llm == "command-r-plus":
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
            else:
                self.llm = ChatGoogleGenerativeAI(model=llm,convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=self.safety_settings, temperature=0.1, stream=True,callbacks=self.callbacks)
    def __getattribute__(self, any: str) -> Any:
        return super().__getattribute__(any)

class RAG():
    def __init__(self, prompt=None, llm = None, retriever = None) -> None:
        self.chain = None
        self.prompt = rag_init.custom_prompt_template2
        self.llm = LLM(llm=llm)
        self.retriever = rag_init.retriever
        
        if prompt is not None:
            self.prompt = prompt

        if llm is not None:
            pass
        
        if retriever is not None:
            self.retriever = retriever      
        qa_prompt = Prompt(template=self.prompt).set_prompt()
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            llm = self.llm,
            memory_key="chat_history",
            input_key='question',
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm.llm, 
            self.retriever, 
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            memory=memory,
            return_source_documents=True,
        )
        
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)
    
    def __str__(self):
        return f"RAG Object with prompt: {self.prompt}, LLM: {self.llm}, Retriever: {self.retriever}"