import os
import json
import time
import streamlit as st
from langchain_core.documents import Document
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

# --- CARGA EN MEMORIA (IN-MEMORY): 100% ESTABLE EN LA NUBE ---
@st.cache_resource(show_spinner=False)
def cargar_componentes():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    path_manual = os.path.join("documentos", "manual_prueba.txt")
    
    if not os.path.exists(path_manual):
        st.error("⚠️ No se encontró el archivo 'documentos/manual_prueba.txt'. Asegúrate de subirlo a GitHub.")
        st.stop()
        
    with open(path_manual, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    # Fragmentar por bloques completos (respeta secciones, RUCs y ofertas)
    bloques = [bloque.strip() for bloque in contenido.split("\n\n") if bloque.strip()]
    documents = [Document(page_content=bloque) for bloque in bloques]
    
    # Crear la base vectorial 100% en memoria RAM (sin depender de carpetas de disco)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Modelo oficial gemini-3.8-flash optimizado
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        api_key=api_key,
        temperature=0.1,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un copiloto de telemercadeo en tiempo real para el equipo de ventas de Premium English.
Tu función es darle al asesor la información exacta o el guion que necesita de inmediato para la llamada.

Reglas de respuesta:
1. **Directo y sin rodeos:** Responde ÚNICAMENTE con los datos solicitados (como RUCs, razones sociales, cuentas bancarias, links, precios) o el guion exacto que debe leer el asesor. Jamás agregues saludos ni introducciones como "Dile esto:".
2. **Fidelidad al manual:** Utiliza la información textual exacta que figura en el contexto.
3. **Versatilidad según la consulta:** 
   - Si piden un dato puntual (ej: "RUC", "cuenta", "oferta expomina", "planes"), entrégalo limpio y ordenado tal como figura en el manual.
   - Si es una objeción (ej: "caro", "no escuché antes"), entrega el guion de ventas exacto con su respectiva pregunta de cierre o instrucción si el manual lo indica.

Contexto disponible:
{context}"""),
        ("human", "{input}")
    ])

    return retriever, prompt, llm

retriever, prompt, llm = cargar_componentes()

# --- GENERADOR DE STREAMING CON REINTENTO AUTOMÁTICO INTELIGENTE ---
def stream_con_respaldo(chain, inputs):
    max_intentos = 3
    espera = 2
    for intento in range(max_intentos):
        try:
            for chunk in chain.stream(inputs):
                yield chunk
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if intento < max_intentos - 1:
                    time.sleep(espera)
                    espera *= 2
                    continue
                else:
                    yield "⚠️ Alta demanda momentánea en la red. Por favor, reintenta en unos segundos."
            else:
                yield f"⚠️ Ocurrió una microinterrupción con el servicio de Google: {err_str}"
                break

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

chat_input_val = st.chat_input("Ej: no escuché antes, caro, cuenta bcp, RUC, oferta expomina...")
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