import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Cargar variables de entorno (.env)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("⚠️ No se encontró GOOGLE_API_KEY en tu archivo .env")

# 1. Cargar el archivo de documento
loader = TextLoader("documentos/manual_prueba.txt", encoding="utf-8")
documents = loader.load()

# 2. Fragmentar el texto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Embeddings de Google (models/gemini-embedding-001)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# 4. Generar / Reemplazar la base de datos ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Base de datos 'chroma_db' regenerada exitosamente con Google Embeddings (models/gemini-embedding-001).")