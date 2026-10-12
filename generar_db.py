import os
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Cargar variables de entorno (.env)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("⚠️ No se encontró GOOGLE_API_KEY en tu archivo .env")

# 1. Leer el archivo directamente con Python nativo
path_manual = os.path.join("documentos", "manual_prueba.txt")
with open(path_manual, "r", encoding="utf-8") as f:
    contenido = f.read()

# 2. Fragmentar el texto por párrafos/bloques usando Python puro (evita torch)
bloques = [bloque.strip() for bloque in contenido.split("\n\n") if bloque.strip()]
documents = [Document(page_content=bloque) for bloque in bloques]

# 3. Embeddings oficiales de Google Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# 4. Limpieza previa de chroma_db para evitar bloqueos de Windows
if os.path.exists("chroma_db"):
    try:
        shutil.rmtree("chroma_db", ignore_errors=True)
    except Exception:
        pass

# 5. Generar la base de datos ChromaDB
print("Generando base de datos vectorial con Google Embeddings...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Base de datos 'chroma_db' regenerada exitosamente sin dependencias locales pesadas.")