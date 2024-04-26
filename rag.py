from typing import Any
import rag_init
import chainlit as cl

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

        memory = rag_init.ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        messages = [
            rag_init.SystemMessagePromptTemplate.from_template(self.prompt),
            rag_init.HumanMessagePromptTemplate.from_template("{question}")
        ]
        qa_prompt = rag_init.ChatPromptTemplate.from_messages(messages)
        self.chain = rag_init.ConversationalRetrievalChain.from_llm(
            self.llm.llm, 
            self.retriever, 
            memory=memory,
            get_chat_history=lambda h : h,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True)
        
    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)