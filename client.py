from rag import *

rag = RAG(prompt=rag_init.custom_prompt_template2, llm = "gemini-pro", retriever=rag_init.retriever)

@cl.on_chat_start
async def init():
    await cl.Message(content="Xin chào, tôi là chatbot hỗ trợ bạn trong việc thực hiện các thủ tục của các dịch vụ công. Vui lòng hãy hỏi tôi một câu hỏi.").send()
    llm = rag.llm
    qa_chain = rag.chain
    cl.user_session.set("qa_chain", qa_chain)
    await cl.Message(content = "Bây giờ bạn có thể hỏi!").send()

@cl.on_message
async def main(message: str):

    qa_chain = cl.user_session.get("qa_chain")
    history = []
    response = qa_chain({"question": message.content,"chat_history": history})
    history.append((message.content, response))
    source_documents = response['source_documents']
    print(response)
    source = source_documents[0].metadata.get('source', None)
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