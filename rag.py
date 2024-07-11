from typing import Any
import rag_init
import chainlit as cl
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationKGMemory
from langchain.memory import ConversationBufferMemory
import os
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory


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
        self.llm = rag_init.ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=self.safety_settings, temperature=0.1)

        if llm is not None:
            if llm == "gemini-pro":
                self.llm = rag_init.ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True,google_api_key=os.getenv("GOOGLE_API_KEY"), safety_settings=self.safety_settings, temperature=0.1)
    def __getattribute__(self, any: str) -> Any:
        return super().__getattribute__(any)

class RAG():
    def __init__(self, prompt=None, llm = None, retriever = None) -> None:
        self.chain = None
        self.prompt = rag_init.custom_prompt_template2
        self.llm = LLM("gemini-pro")
        self.retriever = rag_init.retriever
        
        if prompt is not None:
            self.prompt = prompt

        if llm is not None:
            pass
        
        if retriever is not None:
            self.retriever = retriever      
        qa_prompt = rag_init.PromptTemplate(template=self.prompt, input_variables=["question", "context"])

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm.llm, 
            self.retriever, 
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            get_chat_history=lambda h : h,
            return_source_documents=True,
        )
        
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)
    
    def __str__(self):
        return f"RAG Object with prompt: {self.prompt}, LLM: {self.llm}, Retriever: {self.retriever}"