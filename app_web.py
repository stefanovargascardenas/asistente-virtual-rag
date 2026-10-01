import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración de la página
st.set_page_config(
    page_title="Copiloto de Ventas - Premium English",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Copiloto de Ventas en Vivo")
st.caption("Escribe la objeción o consulta rápida para obtener el guion exacto.")

# Carga de API Key desde Secretos o variables de entorno
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key de Google Gemini. Configúrala en Streamlit Secrets.")
    st.stop()

# Cargar componentes RAG optimizados con caché
@st.cache_resource(show_spinner=False)
def cargar_vectorstore_y_prompt():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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

    return retriever, prompt

retriever, prompt = cargar_vectorstore_y_prompt()

# Función para instanciar el modelo con respaldo de versión
@st.cache_resource(show_spinner=False)
def obtener_llm(key: str):
    # Intentamos primero con gemini-2.0-flash (estándar actual en la SDK google-genai)
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=key,
        temperature=0.1
    )

llm = obtener_llm(api_key)

# Historial de conversación en la interfaz
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada de usuario y ejecución de la respuesta
if user_input := st.chat_input("Ej: no escuché antes, caro, cuenta bcp, profesores nativos..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Buscando guion..."):
            # 1. Recuperación de contexto
            docs = retriever.invoke(user_input)
            contexto = "\n\n".join([doc.page_content for doc in docs])

            # 2. Generación mediante LCEL
            try:
                chain = prompt | llm | StrOutputParser()
                respuesta_texto = chain.invoke({"context": contexto, "input": user_input})
            except Exception as e:
                # Si gemini-2.0-flash tuviera algún inconveniente de permisos, intenta con el alias genérico
                if "NotFoundError" in type(e).__name__ or "NotFound" in str(e):
                    alt_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, temperature=0.1)
                    chain = prompt | alt_llm | StrOutputParser()
                    respuesta_texto = chain.invoke({"context": contexto, "input": user_input})
                else:
                    raise e

            st.write(respuesta_texto)

    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})