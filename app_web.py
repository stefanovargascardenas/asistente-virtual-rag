import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración de página
st.set_page_config(
    page_title="Copiloto de Ventas - Premium English",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Copiloto de Ventas en Vivo")
st.caption("Escribe la objeción o consulta rápida para obtener el guion exacto.")

# Carga de la API Key
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key de Google Gemini. Configúrala en Streamlit Secrets.")
    st.stop()

# Cargar componentes RAG con caché para máxima velocidad
@st.cache_resource
def cargar_componentes():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un copiloto de telemercadeo en tiempo real para el equipo de ventas de Premium English.
Tu única función es darle al asesor el guion EXACTO que debe leerle al cliente en la llamada de inmediato.

Reglas estrictas de respuesta:
1. **Guion directo:** Responde ÚNICAMENTE con las palabras exactas que el asesor debe decir en voz alta. Jamás agregues introducciones, saludos ni frases como "Dile esto:" o "Puedes responder:".
2. **Fidelidad al manual:** Utiliza la respuesta textual que figura en el contexto para esa objeción, precio, link o cuenta bancaria.
3. **Pregunta de cierre:** Incluye siempre al final la pregunta de filtro o cierre del manual para mantener el control de la llamada.
4. **Instrucción de acción (si aplica):** Si el manual indica una regla o acción interna (ej: "Si dice Sí: agendar 5 min"), colócala en una línea aparte entre paréntesis al final.

Contexto disponible:
{context}"""),
        ("human", "{input}")
    ])

    return retriever, prompt, llm

retriever, prompt, llm = cargar_componentes()

# Historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada de usuario y generación de respuesta
if user_input := st.chat_input("Ej: no escuché antes, caro, cuenta bcp, profesores nativos..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Buscando guion..."):
            # 1. Recuperar documentos relevantes
            docs = retriever.invoke(user_input)
            contexto = "\n\n".join([doc.page_content for doc in docs])

            # 2. Ejecutar cadena LCEL
            chain = prompt | llm | StrOutputParser()
            respuesta_texto = chain.invoke({"context": contexto, "input": user_input})

            st.write(respuesta_texto)

    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})