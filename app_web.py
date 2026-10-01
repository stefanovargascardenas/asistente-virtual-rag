import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la página
st.set_page_config(
    page_title="Copiloto de Ventas IA",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# Archivo de persistencia de analítica
FAQS_FILE = "faqs.json"

def load_faqs():
    if os.path.exists(FAQS_FILE):
        try:
            with open(FAQS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_faqs(data):
    with open(FAQS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def track_query(query_text):
    data = load_faqs()
    clean_q = query_text.strip().capitalize()
    data[clean_q] = data.get(clean_q, 0) + 1
    save_faqs(data)

# 2. Carga de VectorStore y Modelos Gemini (Caché de Streamlit)
@st.cache_resource
def init_rag_system():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: NO se encontró la variable GOOGLE_API_KEY en el entorno.")
        st.stop()
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # LLM principal con fallback
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=api_key
    )
    
    return vectorstore, llm

vectorstore, llm = init_rag_system()

# Template de prompt enfocado en respuestas directas para asesores
SYSTEM_PROMPT = """Eres el copiloto en tiempo real para un asesor de ventas. 
Tu trabajo es dar la respuesta exacta que el asesor debe leer o enviar al cliente.

Reglas de respuesta:
1. Sé directo, profesional y cordial.
2. Si la información incluye datos sensibles (cuentas, CCI, precios), facilítalos estructurados.
3. Termina siempre con una pregunta de cierre oportuna.
4. Si hay alguna aclaración interna para el asesor, colócala al final entre paréntesis (ejemplo: (Nota: Validar comprobante en el sistema)).

Contexto relevante del manual:
{context}

Pregunta del cliente:
{question}
"""

prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

# 3. Interfaz Principal con Pestañas
tab_copiloto, tab_analytics = st.tabs(["💬 Copiloto de Ventas", "📊 Panel de Analítica"])

# ----------------------------------------------------
# PESTAÑA 1: COPILOTO DE VENTAS
# ----------------------------------------------------
with tab_copiloto:
    st.title("🤖 Copiloto de Ventas en Vivo")
    st.caption("Asistente IA para atención y ventas con búsqueda semántica y control de alucinaciones.")

    # Sección Dinámica de Preguntas Frecuentes
    faqs_data = load_faqs()
    if faqs_data:
        st.markdown("### ⚡ Accesos Rápidos (FAQs Más Frecuentes)")
        top_faqs = sorted(faqs_data.items(), key=lambda x: x[1], reverse=True)[:4]
        cols = st.columns(len(top_faqs))
        for idx, (faq_text, count) in enumerate(top_faqs):
            if cols[idx].button(f"📌 {faq_text} ({count})", use_container_width=True):
                st.session_state["user_input_val"] = faq_text

    # Estado de chat
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "¡Hola! Estoy listo para ayudarte en la llamada. ¿Qué duda o requerimiento tiene el cliente?"}
        ]

    # Mostrar historial
    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Agregar componente de feedback en las respuestas del asistente
            if msg["role"] == "assistant" and i > 0:
                st.feedback("thumbs", key=f"fb_{i}", disabled=True)

    # Captura de entrada
    user_input = st.chat_input("Escribe la consulta del cliente aquí...")
    
    # Manejo de selección de FAQs rápidas
    if "user_input_val" in st.session_state and st.session_state["user_input_val"]:
        user_input = st.session_state.pop("user_input_val")

    if user_input:
        # Registrar pregunta para analítica
        track_query(user_input)

        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # FILTRO DE SIMILITUD / ALUCINACIONES (Costo $0)
            # Retorna documentos con sus puntuaciones de relevancia (0 a 1)
            results_with_scores = vectorstore.similarity_search_with_relevance_scores(user_input, k=3)
            
            UMBRAL_SIMILITUD = 0.5  # Si la coincidencia es menor al 50%, se activa la respuesta segura
            
            # Validar si hay resultados válidos por encima del umbral
            relevant_docs = [doc for doc, score in results_with_scores if score >= UMBRAL_SIMILITUD]

            if not relevant_docs:
                # Mensaje de resguardo en caso de no encontrar coincidencia confiable
                fallback_msg = (
                    "No encontré información específica sobre esa consulta en la base de conocimientos. "
                    "Por favor, consulta con el supervisor de turno o valida el manual interno.\n\n"
                    "*(Nota interna: Respuesta detenida por el filtro de seguridad para evitar alucinaciones)*"
                )
                st.markdown(fallback_msg)
                st.session_state["messages"].append({"role": "assistant", "content": fallback_msg})
            else:
                # Construir contexto acumulado de los documentos relevantes
                context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
                
                chain = prompt_template | llm | StrOutputParser()
                
                # Ejecución con Streaming
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in chain.stream({"context": context_text, "question": user_input}):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Feedback del usuario (👍 / 👎)
                st.feedback("thumbs", key=f"fb_{len(st.session_state['messages'])}")
                
                st.session_state["messages"].append({"role": "assistant", "content": full_response})

# ----------------------------------------------------
# PESTAÑA 2: DASHBOARD DE ANALÍTICA E INTELIGENCIA
# ----------------------------------------------------
with tab_analytics:
    st.title("📊 Dashboard de Analítica de Consultas")
    st.caption("Métricas en tiempo real sobre las dudas recurrentes atendidas por el copiloto.")

    faqs = load_faqs()
    if not faqs:
        st.info("Aún no hay datos de consultas registrados. Haz algunas preguntas en el copiloto para ver el gráfico aquí.")
    else:
        df = pd.DataFrame(list(faqs.items()), columns=["Consulta", "Frecuencia"])
        df = df.sort_values(by="Frecuencia", ascending=False)

        # KPIs Principales
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Consultas Registradas", df["Frecuencia"].sum())
        col2.metric("Tipos de Preguntas Únicas", len(df))
        col3.metric("Consulta Más Repetida", df.iloc[0]["Consulta"] if not df.empty else "N/A")

        st.markdown("---")

        # Gráfico interactivo con Plotly
        fig = px.bar(
            df.head(10),
            x="Frecuencia",
            y="Consulta",
            orientation="h",
            title="Top 10 Consultas Más Frecuentes de los Clientes",
            color="Frecuencia",
            color_continuous_scale="Blues"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        # Tabla de Datos
        st.markdown("### 📋 Detalle Completo")
        st.dataframe(df, use_container_width=True)