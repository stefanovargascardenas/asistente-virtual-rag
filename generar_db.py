import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Cargando información de la carpeta documentos...")
# Le indicamos explícitamente la codificación UTF-8 para tildes y eñes
loader = DirectoryLoader(
    'documentos/', 
    glob="*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

print("Creando base de datos de vectores en chroma_db...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

print("¡Base de datos regenerada con éxito!")