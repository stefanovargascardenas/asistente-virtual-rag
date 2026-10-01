import os
import shutil
import streamlit as st
from collections import Counter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Asistente Virtual RAG",
    page_icon="🤖",
    layout="wide"
)

# Inicializar estados en la sesión si no existen
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_history" not in st.session_state:
    st.session_state.query_history = []  # Para registrar todas las preguntas hechas

# 1. Carga de VectorStore y Modelos (Caché de Streamlit)
@st.cache_resource
def init_rag_system():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: NO se encontró la variable GOOGLE_API_KEY en el entorno o en el archivo .env.")
        st.stop()
        
    # Usar embeddings locales con HuggingFace (dimensión 384)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Manejo de incompatibilidad de dimensiones previa en Chroma:
    db_dir = "./chroma_db"
    if os.path.exists(db_dir):
        try:
            temp_check = Chroma(persist_directory=db_dir, embedding_function=embeddings)
        except Exception:
            shutil.rmtree(db_dir)
            
    vectorstore = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    
    # LLM principal con Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )
    
    return vectorstore, llm

# Inicializar sistema RAG
vectorstore, llm = init_rag_system()

# Configurar el retriever
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.3}
)

# Prompt del sistema
template = """Eres un asistente virtual experto y amable. 
Usa estrictamente el siguiente contexto recuperado para responder a la pregunta del usuario.
Si la respuesta no se encuentra en el contexto, di educadamente que no tienes información al respecto basada en los documentos provistos, sin inventar datos.

Contexto:
{context}

Pregunta:
{question}

Respuesta clara y detallada:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Interfaz de Usuario en Streamlit
st.title("🤖 Asistente Virtual RAG con Gemini")
st.write("Pregúntame sobre tus documentos y te responderé con base en la información oficial.")

# Pestañas principales
tab_chat, tab_dashboard = st.tabs(["💬 Chat Asistente", "📊 Dashboard y Analítica"])

with tab_chat:
    # ── Sección Dinámica: Preguntas Más Frecuentes de los Asesores ──────────────────
    st.markdown("### 💡 Preguntas más frecuentes (Detectadas automáticamente)")
    
    # Preguntas predeterminadas iniciales por si la sesión recién empieza
    default_frequent_queries = [
        "¿Qué duda o requerimiento tiene el cliente?",
        "¿Cuáles son los detalles sobre BCP y pagos?",
        "¿De qué trata el servicio de asesoría?",
        "Haz un resumen general de los documentos."
    ]
    
    # Combinar historial real de la sesión con las predeterminadas para calcular frecuencias
    all_recorded_queries = st.session_state.query_history + default_frequent_queries
    query_counts = Counter(all_recorded_queries)
    
    # Obtener las 4 preguntas más comunes ordenadas de mayor a menor frecuencia
    top_frequent = query_counts.most_common(4)
    
    # Renderizar botones dinámicos con el contador al costado
    col_q1, col_q2 = st.columns(2)
    quick_query = None
    
    for i, (q_text, count) in enumerate(top_frequent):
        button_label = f"{q_text} ({count})"
        target_col = col_q1 if i % 2 == 0 else col_q2
        
        with target_col:
            if st.button(button_label, key=f"freq_btn_{i}"):
                quick_query = q_text

    st.divider()

    # Contenedor del historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Escribe tu pregunta aquí...")
    
    if quick_query:
        user_input = quick_query

    if user_input:
        # Registrar la consulta en el historial para actualizar los contadores
        st.session_state.query_history.append(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos y generando respuesta..."):
                try:
                    retrieved_docs = retriever.invoke(user_input)
                    
                    if not retrieved_docs:
                        response = "Lo siento, no he encontrado información relevante en los documentos para responder a tu consulta con suficiente precisión."
                        st.markdown(response)
                    else:
                        response_container = st.empty()
                        full_response = ""
                        
                        for chunk in chain.stream(user_input):
                            full_response += chunk
                            response_container.markdown(full_response + "▌")
                        
                        response_container.markdown(full_response)
                        response = full_response
                        
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.last_query = user_input
                    st.session_state.last_response = response
                    
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar tu consulta: {e}")

    # Sección de Feedback
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.divider()
        st.markdown("### ¿Te fue útil esta respuesta?")
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            if st.button("👍 Sí, útil"):
                st.session_state.feedback_data.append({
                    "pregunta": st.session_state.get("last_query", ""),
                    "feedback": "Positivo",
                    "timestamp": pd.Timestamp.now()
                })
                st.success("¡Gracias por tu feedback positivo!")
        with col2:
            if st.button("👎 No útil"):
                st.session_state.feedback_data.append({
                    "pregunta": st.session_state.get("last_query", ""),
                    "feedback": "Negativo",
                    "timestamp": pd.Timestamp.now()
                })
                st.warning("Gracias. Tomaremos en cuenta este reporte para mejorar.")

with tab_dashboard:
    st.header("📊 Panel de Analítica y Uso del Asistente")
    st.write("Métricas de interacción, volumen de consultas y registros de satisfacción de los usuarios.")
    
    if len(st.session_state.feedback_data) == 0:
        st.info("Aún no hay registros de feedback en esta sesión. Interactúa con el chat para generar datos en el dashboard.")
    else:
        df_feedback = pd.DataFrame(st.session_state.feedback_data)
        
        total_interacciones = len(st.session_state.messages) // 2
        total_feedbacks = len(df_feedback)
        positivos = len(df_feedback[df_feedback["feedback"] == "Positivo"])
        satisfaccion = (positivos / total_feedbacks * 100) if total_feedbacks > 0 else 0
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Consultas Realizadas", total_interacciones)
        col_m2.metric("Total de Feedbacks", total_feedbacks)
        col_m3.metric("Tasa de Satisfacción", f"{satisfaccion:.1f}%")
        
        st.divider()
        
        st.subheader("Distribución de Feedback del Usuario")
        fig_feedback = px.pie(
            df_feedback, 
            names="feedback", 
            title="Proporción de Feedback (Positivo vs Negativo)",
            color="feedback",
            color_discrete_map={"Positivo": "#00CC96", "Negativo": "#EF553B"}
        )
        st.plotly_chart(fig_feedback, use_container_width=True)
        
        st.subheader("Historial de Retroalimentación Registrada")
        st.dataframe(df_feedback, use_container_width=True)