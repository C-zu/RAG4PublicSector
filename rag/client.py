from rag import *
from dotenv import load_dotenv
load_dotenv()


rag = RAG(prompt=rag_init.custom_prompt_template2, llm = "command-r-plus", retriever=rag_init.retriever)
print(rag)
history = []
@cl.on_chat_start
async def init():
    await cl.Message(content="Xin chào, tôi là chatbot hỗ trợ bạn trong việc thực hiện các thủ tục của các dịch vụ công. Vui lòng hãy hỏi tôi một câu hỏi.").send()
    # llm = rag.llm
    qa_chain = rag.chain
    cl.user_session.set("qa_chain", qa_chain)
    await cl.Message(content = "Bây giờ bạn có thể hỏi!").send()

@cl.on_message
async def main(message: str):

    qa_chain = cl.user_session.get("qa_chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    cb.answer_reached = True
    response = await qa_chain.acall(message.content, callbacks=[cb])
    # chat_history = []
    # response = qa_chain({"question": message.content,"chat_history": chat_history})
    # chat_history.append(message.content) 
    # print(response)
    source_documents = response['source_documents']
    source = max((d.metadata.get('source', None), d.metadata.get('relevance_score', 0)) for d in source_documents)
    
    list_acc = [i.metadata.get('relevance_score', 0) for i in source_documents]
    print(list_acc)
    # if (source[1] <= 0.05):
    #     res_full = cl.Message("Tôi không biết trả lời câu hỏi này.")
    #     await res_full.send()  
    # elif (source[1] <= 0.1):
    #     res_full = cl.Message("Tôi không biết bạn đang hỏi về thủ tục nào, hãy nêu rõ tên thủ tục.")
    #     await res_full.send()  
    # else:
    elements = [
            cl.Text(name="Nguồn", content=source[0], display="inline")
        ]
    # await cl.Message(
    #     content=response['answer'],
    #     elements=elements,
    # ).send()
    if cb.has_streamed_final_answer:
        cb.final_stream.content = response['answer']
        cb.final_stream.elements = elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=response['answer'], elements=elements).send()
