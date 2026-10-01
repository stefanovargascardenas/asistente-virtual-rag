import os
import json
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Archivo local para almacenar el conteo de preguntas
FAQ_FILE = "faqs.json"

# --- FUNCIONES PARA GESTIONAR LAS PREGUNTAS FRECUENTES (FAQs) ---
def cargar_faqs():
    """Carga el contador de preguntas desde session_state o archivo local."""
    if "faqs_data" in st.session_state:
        return st.session_state.faqs_data
    
    faqs_iniciales = {
        "cuenta bcp": 10,
        "caro": 8,
        "profesores nativos": 6,
        "no escuché antes": 4
    }
    
    if os.path.exists(FAQ_FILE):
        try:
            with open(FAQ_FILE, "r", encoding="utf-8") as f:
                faqs_iniciales = json.load(f)
        except Exception:
            pass
            
    st.session_state.faqs_data = faqs_iniciales
    return faqs_iniciales

def registrar_consulta(query: str):
    """Incrementa la frecuencia de la consulta en el registro."""
    query_norm = query.strip().lower()
    if len(query_norm) < 2:
        return
    faqs = cargar_faqs()
    faqs[query_norm] = faqs.get(query_norm, 0) + 1
    st.session_state.faqs_data = faqs
    
    try:
        with open(FAQ_FILE, "w", encoding="utf-8") as f:
            json.dump(faqs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def obtener_top_faqs(limite=4):
    """Devuelve las preguntas más frecuentes ordenadas por repetición."""
    faqs = cargar_faqs()
    faqs_ordenadas = sorted(faqs.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in faqs_ordenadas[:limite]]

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Copiloto de Ventas - Premium English",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Copiloto de Ventas en Vivo")
st.caption("Escribe la objeción o haz clic en las preguntas más frecuentes.")

# Carga de API Key
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key de Google Gemini. Configúrala en Streamlit Secrets.")
    st.stop()

# Cargar componentes RAG optimizados con reintentos integrados
@st.cache_resource(show_spinner=False)
def cargar_componentes():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Modelo ultra-estable gemini-1.5-flash con 3 reintentos automáticos
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1,
        streaming=True,
        max_retries=3
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

# --- GENERADOR CON STREAMING Y RESPALDO ROBUTO ---
def stream_con_respaldo(chain, inputs, prompt, api_key):
    try:
        for chunk in chain.stream(inputs):
            yield chunk
    except Exception:
        # Fallback instantáneo en modo síncrono si el stream parpadea
        try:
            llm_backup = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.1,
                max_retries=3
            )
            chain_backup = prompt | llm_backup | StrOutputParser()
            yield chain_backup.invoke(inputs)
        except Exception:
            yield "⚠️ Hubo una microinterrupción temporal con Google. Por favor, presiona el botón nuevamente."

# --- SECCIÓN DE BOTONES DE ACCESO RÁPIDO (TOP PREGUNTAS) ---
st.markdown("##### 💡 **Consultas más frecuentes (haz clic para guion rápido):**")
top_preguntas = obtener_top_faqs(limite=4)

faq_seleccionada = None
cols = st.columns(len(top_preguntas))
for idx, preg in enumerate(top_preguntas):
    if cols[idx].button(f"📌 {preg.capitalize()}", use_container_width=True, key=f"btn_faq_{idx}"):
        faq_seleccionada = preg

st.divider()

# Historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Capturar entrada
chat_input_val = st.chat_input("Ej: no escuché antes, caro, cuenta bcp, profesores nativos...")
query_final = chat_input_val or faq_seleccionada

if query_final:
    registrar_consulta(query_final)

    st.session_state.messages.append({"role": "user", "content": query_final})
    with st.chat_message("user"):
        st.write(query_final)

    with st.chat_message("assistant"):
        docs = retriever.invoke(query_final)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        chain = prompt | llm | StrOutputParser()
        inputs = {"context": contexto, "input": query_final}

        # Generación con reintentos y streaming estable
        respuesta_texto = st.write_stream(stream_con_respaldo(chain, inputs, prompt, api_key))

    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})