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

# --- CARGA EN MEMORIA BLINDADA CONTRA RATE LIMITS ---
@st.cache_resource(show_spinner=False)
def cargar_componentes():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    
    # Documentos consolidados en 3 bloques exactos (consumo mínimo de API para evitar bloqueos)
    docs_manual = [
        Document(page_content="""OBJECIONES NIVEL 1 Y NIVEL 2:
        - NO HABÍA ESCUCHADO ANTES DE USTEDES: Premium English tiene 26 años en el mercado, publicidad dirigida a profesionales en EXPOMINA, TECNOSALUD, FERIA DEL LIBRO. ¿Tienes interés en aprender inglés actualmente?
        - ¿CÓMO TIENES MI NÚMERO?: Diversas bases de datos de empresas y eventos profesionales. ¿Tienes interés en aprender inglés actualmente?
        - ¿ME PUEDE DEVOLVER LA LLAMADA LUEGO?: Con gusto, ¿me confirmas tu interés actual? Sí: Agendar llamada de 5 min. No: Tachar dato.
        - MÁNDAME LA INFORMACIÓN POR CORREO O WHATSAPP: Te envío info general, pero tenemos beneficios económicos y clase modelo gratuita. ¿En qué horario tienes 5 minutos libres?
        - ¿CUÁNTO CUESTA?: Rango de inversión entre S/300 y S/400 según tu nivel. ¿Tienes algún nivel o quieres comenzar de cero?
        - ME PARECE MUY CARO: ¿Cuál es un monto que puedes manejar mensualmente? S/200-S/250: Tarjeta de crédito para descuento adicional y agendar clase modelo. Menos de S/200: No es el perfil.
        - ME QUEDA BIEN CUALQUIER PLAN: Si tuvieras que elegir uno ahora, ¿cuál se alinea mejor a tu presupuesto?
        - TOMO DECISIONES CON ALGUIEN MÁS: Es importante comentarle la propuesta antes de la clase modelo.
        - ¿PUEDO HACER MÁS DE 4 HORAS?: Extensión opcional de horas adicionales. Con 6 horas sigues el syllabus para lograr el objetivo en 12 meses.
        - ¿PUEDO HACER LAS 4 HORAS EN UN DÍA?: Ideal distribuir en 3 días por rendimiento cerebral, pero manejable como excepción.
        - ¿LOS PROFESORES SON NATIVOS?: Sí, desde el módulo intermedio en adelante para evitar inseguridad inicial.
        - ¿QUÉ DIFERENCIA TU PROGRAMA?: Flexibilidad de horarios (ingresar solo una hora antes) y seguimiento/asesoría constante post-matrícula.
        - ¿QUIÉN ME GARANTIZA QUE VOY A APRENDER?: El alumno cumpliendo las 6 horas. Damos opción de repetir clases no comprendidas.
        - CLASE SIN COMPROMISO: La clase es para vivir la experiencia antes de inscribirte. ¿Estamos de acuerdo?"""),
        
        Document(page_content="""PLANES DE INVERSIÓN Y PRECIOS:
        - Opción Contado 1: Matrícula S/ 3600 (Total: S/ 3600).
        - Opción Contado 2: Matrícula S/ 3990 (Total: S/ 3990).
        - Plan 2 Cuotas: Matrícula S/ 1900 + 2 cuotas de S/ 1200 (Total: S/ 4300).
        - Plan 4 Cuotas: Matrícula S/ 1400 + 4 cuotas de S/ 800 (Total: S/ 4600).
        - Plan 10 Cuotas: Matrícula S/ 1250 + 10 cuotas de S/ 350 (Total: S/ 4750).
        PLANES INHOUSE:
        - Contado Inhouse: Matrícula S/ 6500 (Total: S/ 6500).
        - Plan 2 Cuotas Inhouse: Matrícula S/ 2550 + 3 cuotas de S/ 1600 (Total: S/ 7350).
        - Plan 4 Cuotas Inhouse: Matrícula S/ 1850 + 6 cuotas de S/ 1100 (Total: S/ 8450)."""),
        
        Document(page_content="""LINKS DE PAGO, CUENTAS BANCARIAS Y DATOS FISCALES (RUC):
        LINKS ESEC:
        - Niubiz BBVA/Diners: https://pagolink.niubiz.com.pe/pagoseguro/ESECPERU/3250291
        - Mercado Pago BCP/Interbank: https://link.mercadopago.com.pe/premiumenglishprogra
        - Mercado Pago S/ 3600: https://mpago.la/2mgU3Bk
        - Mercado Pago S/ 3990: https://mpago.la/2JapmHy
        LINKS INTERPASS:
        - Niubiz S/ 2500: https://pagolink.niubiz.com.pe/pagoseguro/PREMIUMENGLISHPROGRAM/722304
        - Mercado Pago S/ 2500: https://mpago.la/1mLXnoU
        CUENTAS ESEC:
        - BBVA ESEC: 0011-0355-0100001919 | CCI: 011-355-000100001919-58
        - BCP ESEC: 194-1927206-0-26 | CCI: 002-194-001927206026-90
        - INTERBANK ESEC: 067-3000004152 | CCI: 003-067-003000004152-62
        - YAPE ESEC (Máx 500): 962911799
        CUENTAS INTERPASS:
        - BBVA INTERPASS: 0011-0189-0100035765 | CCI: 011-189-000100035765-85
        DATOS FISCALES Y RUCs:
        1. Razón Social: Escuela Sudamericana de Extensión Cultural para Perú S.A. | RUC: 20428652577
        2. Razón Social: Premium Cultural Institute S.A.C. | RUC: 20502705131
        3. Razón Social: INTERNATIONAL PASSWORD SYSTEM PERU S.A.C. | RUC: 20523074653
        OFERTA EXPOMINA: De S/4500 a S/2500 con horarios flexibles, clases personalizadas, TOEFL/CAMBRIDGE.
        PLANTILLAS DE SEGUIMIENTO: Cierre/pausa, falta de contacto, contacto inicial de Samy Pinto.""")
    ]
    
    # Crear base vectorial 100% en memoria RAM sin carpetas físicas
    vectorstore = Chroma.from_documents(
        documents=docs_manual,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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

# --- GENERADOR DE STREAMING CON REINTENTO AUTOMÁTICO ---
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