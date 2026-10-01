import os
import json
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

FAQ_FILE = "faqs.json"

# --- FUNCIONES PARA GESTIONAR LAS PREGUNTAS FRECUENTES (FAQs) ---
def cargar_faqs():
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

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la API Key de Google Gemini en Streamlit Secrets.")
    st.stop()

# --- CARGA DE COMPONENTES RAG CON GOOGLE EMBEDDINGS (0% CPU LOCAL) ---
@st.cache_resource(show_spinner=False)
def cargar_componentes():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Modelo principal según la recomendación oficial de Google
    llm_principal = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        api_key=api_key,
        temperature=0.1,
        streaming=True
    )
    
    # Modelo de respaldo
    llm_respaldo_1 = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        api_key=api_key,
        temperature=0.1,
        streaming=True
    )

    llm_con_fallbacks = llm_principal.with_fallbacks([llm_respaldo_1])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un copiloto de telemercadeo en tiempo real para el equipo de ventas de Premium English.
Tu única función es darle al asesor el guion EXACTO que debe leerle al cliente en la llamada de inmediato.

Reglas strictly directas de respuesta:
1. **Guion directo:** Responde ÚNICAMENTE con las palabras exactas que el asesor debe decir en voz alta. Jamás agregues introducciones, saludos ni frases como "Dile esto:" o "Puedes responder:".
2. **Fidelidad al manual:** Utiliza la respuesta textual que figura en el contexto para esa objeción, precio, link o cuenta bancaria.
3. **Pregunta de cierre:** Incluye siempre al final la pregunta de filtro o cierre del manual para mantener el control de la llamada.
4. **Instrucción de acción (si aplica):** Si el manual indica una regla o acción interna (ej: "Si dice Sí: agendar 5 min"), colócala en una línea aparte entre paréntesis al final.

Contexto disponible:
{context}"""),
        ("human", "{input}")
    ])

    return retriever, prompt, llm_con_fallbacks

retriever, prompt, llm = cargar_componentes()

# --- GENERADOR DE STREAMING CON ESCUDO DE ERRORES ---
def stream_con_respaldo(chain, inputs):
    try:
        for chunk in chain.stream(inputs):
            yield chunk
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            yield "⚠️ Se alcanzó el límite momentáneo de consultas. Espera unos segundos e intenta nuevamente."
        else:
            yield f"⚠️ Ocurrió una microinterrupción con el servicio de Google: {err_str}"

# --- SECCIÓN DE BOTONES DE ACCESO RÁPIDO ---
st.markdown("##### 💡 **Consultas más frecuentes (haz clic para guion rápido):**")
top_preguntas = obtener_top_faqs(limite=4)

faq_seleccionada = None
cols = st.columns(len(top_preguntas))
for idx, preg in enumerate(top_preguntas):
    if cols[idx].button(f"📌 {preg.capitalize()}", use_container_width=True, key=f"btn_faq_{idx}"):
        faq_seleccionada = preg

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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

        respuesta_texto = st.write_stream(stream_con_respaldo(chain, inputs))

    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})