import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuración de la página Web
st.set_page_config(
    page_title="Soporte Interno | Cursos de Inglés",
    page_icon="📘",
    layout="centered"
)

# Cargar la base de datos vectorial pre-procesada (Carga Ultra-Rápida)
@st.cache_resource
def preparar_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Lee directamente la base de datos Chroma ya generada
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    system_prompt = (
        "Eres un Asistente Virtual Interno de Soporte para el equipo de ventas y supervisión "
        "de cursos de inglés para profesionales.\n"
        "Tu función es resolver dudas técnicas, procedimientos de trabajo, precios, temarios "
        "y políticas internas de la empresa.\n\n"
        "REGLAS ESTRICTAS DE RESPUESTA:\n"
        "1. Responde ÚNICAMENTE utilizando la información del contexto provisto a continuación.\n"
        "2. Si la respuesta no está explícitamente en el contexto, responde únicamente: "
        "'No dispongo de esa información en los manuales internos. Por favor consulta con supervisión.'\n"
        "3. Sé claro, profesional y directo.\n\n"
        "Contexto del manual interno:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

rag_chain = preparar_rag()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712010.png", width=80)
    st.title("Panel de Control")
    st.markdown("Este asistente responde dudas usando exclusivamente los **manuales y políticas oficiales** de la empresa.")
    
    st.divider()
    
    # Botón para reiniciar chat
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.divider()
    st.caption("📌 **Nota:** Si la consulta no está en el manual, contacta directamente con Supervisión.")

# --- ENCABEZADO PRINCIPAL ---
st.title("📘 Asistente Virtual Interno")
st.caption("Consulta rápida de duraciones, temarios, precios y procedimientos internos.")
st.divider()

# Inicializar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mensaje de bienvenida / sugerencias cuando el chat está vacío
if len(st.session_state.messages) == 0:
    st.info(
        "💡 **Sugerencias de consulta:**\n"
        "- ¿Cuánto dura el curso de inglés?\n"
        "- ¿Cuáles son las políticas de pago y matrícula?\n"
        "- ¿Qué temas incluye el programa ejecutivo?"
    )

# Mostrar historial de conversación con avatares
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Campo de entrada de texto
if pregunta := st.chat_input("Escribe tu pregunta aquí..."):
    # Mostrar pregunta del usuario
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)
    st.session_state.messages.append({"role": "user", "content": pregunta})

    # Generar respuesta del asistente
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Consultando los manuales internos..."):
            respuesta = rag_chain.invoke({"input": pregunta})
            texto_respuesta = respuesta["answer"]
            st.markdown(texto_respuesta)
            
    st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})