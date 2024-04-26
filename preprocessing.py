from docx import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain import hub
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pathlib import Path
import pickle
import os


def extract_text_and_tables_from_docx(file, txt_path):
    final_txt = []
    if file.endswith('.doc'):
        filename = txt_path +'/'+ os.path.splitext(os.path.basename(file))[0] + '.txt'
        doc = Document(docx_path + '/' + file)
        output = ""

        for element in doc.element.body:
            if element.tag.endswith('tbl'):
                # Extract table
                table_data = []
                for row in element.iterfind('.//w:tr', namespaces=element.nsmap):
                    row_data = []
                    for cell in row.iterfind('.//w:tc', namespaces=element.nsmap):
                        cell_text = ''.join(node.text for node in cell.iterfind('.//w:t', namespaces=element.nsmap) if node.text)
                        row_data.append(cell_text.strip())
                    table_data.append(row_data)
                for row in table_data:
                    output += '|' + '|'.join(row) + '|\n'
                output += '\n'  # Separate tables with an empty line
            elif element.tag.endswith('p'):
                # Extract paragraph text
                paragraph_text = ''
                for run in element.iterfind('.//w:r', namespaces=element.nsmap):
                    for node in run.iterfind('.//w:t', namespaces=element.nsmap):
                        text = node.text.strip() if node.text else ''
                        if run.find('.//w:b', namespaces=element.nsmap) is not None:
                            text = f"**{text}**"  # Wrap text in '**' if it's bold
                        paragraph_text += text
                output += f"{paragraph_text.strip()}\n"
        final_txt.append(output)
        final_txt.append("[END]")
        with open(filename, "w", encoding="utf-8") as output:
            output.write(str(final_txt))


def load_chunk(directory_path):
    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_id, model_kwargs={"device": "cpu"})

    text_splitter = RecursiveCharacterTextSplitter(separators='\n', chunk_size=1000, chunk_overlap=200)

    chunked_documents = []

    for file_path in Path(directory_path).rglob('*.*'):
        if file_path.is_file():
            loader = TextLoader(file_path, encoding='utf8')
            data = loader.load()
            data[0].metadata['source'] = os.path.splitext(os.path.basename(file_path))[0] + '.doc - ' + 'https://git-link.vercel.app/api/download?url=https%3A%2F%2Fgithub.com%2FC-zu%2FRAG4PublicSector%2Fblob%2Fmain%2Fdata%2F' + os.path.splitext(os.path.basename(file_path))[0] + '.doc'
            chunked_documents.extend(text_splitter.split_documents(data))
    bm25_retriever = BM25Retriever.from_documents(chunked_documents)
    bm25_retriever.k = 5

    docsearch = Qdrant.from_documents(
        chunked_documents,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )

    qdrant_retriever = docsearch.as_retriever(search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, qdrant_retriever], weights=[0.5, 0.5])

    # Cohere Reranker
    compressor = CohereRerank(user_agent='langchain')
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever

# def save_retriever_to_pickle(retriever, filename):
#     with open(filename, 'wb') as file:
#         pickle.dump(retriever, file)
        

txt_path = './data/txt_file'
docx_path = './data'
if not os.path.exists(txt_path):
        os.makedirs(txt_path)
for file in os.listdir(docx_path):           
    extract_text_and_tables_from_docx(file,txt_path)
    
retriever = load_chunk(txt_path)