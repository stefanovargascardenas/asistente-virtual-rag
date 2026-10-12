import os
import json
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
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

# --- CARGA DEL MODELO Y MANUAL DIRECTO EN EL PROMPT ---
@st.cache_resource(show_spinner=False)
def cargar_componentes():
    # Modelo exacto de versión estable para evitar errores 404 y ampliar cuota a 1500 diarias
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002",
        api_key=api_key,
        temperature=0.1,
        streaming=True
    )

    manual_completo = """
OBJECIONES NIVEL 1
Objeciones en el filtro

NO HABÍA ESCUCHADO ANTES DE USTEDES
Comprendo nombre, Premium English tiene 26 años en el mercado, sin embargo, no manejamos publicidad masiva sino dirigida directamente a nuestro público objetivos que son profesionales y la mejor forma de llegar a ellos es través de eventos como EXPOMINA, TECNOSALUD, LA FERIA DE LIBRO, donde siempre participamos. La idea no es incomodarte, en todo caso coméntame nombre ¿tienes un interés en aprender inglés actualmente?

¿CÓMO TIENES MI NÚMERO?
Manejamos diversas bases de datos correspondientes a diferentes empresas y eventos profesionales, seguro participaste en alguno de ellos. Ahora nombre, la idea no es incomodarte, en todo caso coméntame ¿tienes un interés en aprender inglés actualmente?

¿ME PUEDE DEVOLVER LA LLAMADA LUEGO?
Con mucho gusto, pero como la idea no es incomodarte nombre, ¿me confirmas tu interés actual por aprender inglés? 
Sí: Agendar horario para la segunda llamada, reforzar que es una llamada breve “de 5 minutos”. No: Tachar dato (no interesado)

MANDAME LA INFORMACIÓN POR CORREO O A MI WHATSAPP Y YO ME COMUNICO
Claro que sí, te envío información general para que conozcas más de la empresa y del programa; sin embargo, te comento que estamos manejando beneficios en la parte económica e incluso una opción de darte una clase modelo gratuita para que antes de tomar cualquier decisión puedas vivir la experiencia con nosotros. 
La idea es explicarte en qué consisten estos beneficios y como puedes llevar el programa de forma más personalizada. En todo caso, ¿en qué horario tienes unos 5 minutos libres para agendarte?
*Evitar mandar información sin previa conversación. NO HAY RETORNO.

¿CUÁNTO CUESTA? (en un momento que no corresponde en el telemercadeo)
Nombre nuestros planes tienen un rango de inversión entre S/300 y S/400, va a depender mucho del nivel en el que estés, cuéntame, tienes algún nivel o quieres comenzar de cero?
(la pregunta debe buscar retomar la estructura del telemercadeo)

ME PARECE MUY CARO (continuación de ¿CUANTO CUESTA?)
Comprendo nombre, para ver que opciones darte, ¿cuál es un monto que puedes manejar mensualmente que no afecte tu presupuesto?
S/200 – S/250: preguntar si tiene alguna tarjeta de crédito “porque tenemos algunos beneficios adicionales”. Si tiene, agendar clase modelo y comprometerse a ayudarlo con un descuento adicional.
Menos de S/200: Enviar información y quedar bien con el alumno. NO ES EL PERFIL QUE BUSCAMOS.

“ME QUEDA BIEN CUALQUIERA DE LOS PLANES” (no elije uno)
Excelente nombre, pero si tuvieras que elegir uno ahora, ¿cuál se alinea mejor a tu presupuesto?
*Si no elije uno probablemente no se le acomode ninguno realmente.

“TOMO LAS DECISIONES CON ALGUIEN MÁS” (no autonomía)
Comprendo nombre, en ese caso es importante que puedas comentarle la propuesta para que conversen sobre el tema antes de la clase modelo.

OBJECIONES NIVEL 2

¿PUEDO HACER MÁS DE CUATRO HORAS?
Por supuesto, te damos una extensión de horas adicionales opcionales. Entonces si tienes disponibilidad de tiempo y quieres hacer más horas no hay problema, de igual forma te damos la tranquilidad que con 6 horas estás siguiendo el orden del syllabus para lograr el objetivo en el tiempo indicado (12 meses).

¿PUEDO HACER LAS CUATRO HORAS EN UN DÍA?
Lo ideal es que puedas distribuir las 6 horas al menos en tres días, porque en un solo día no es pedagógicamente recomendable, está comprobando que el cerebro toma atención las primeras dos horas, la tercera puedes estar ahí pero probablemente tu mente ya está pensando en otras cosas. Pero si se trata de una excepción, es manejable, lo podemos conversar, la idea es apoyarte para que tengas constancia en tus clases.

¿LOS PROFESORES SON NATIVOS?
Sí, contamos con profesores nativos para el módulo de intermedio en adelante donde el alumno ya se adaptó; inicialmente trabajábamos con profesores nativos desde las primeras unidades, pero esto resultó contraproducente porque generaba inseguridad y miedo en los alumnos.

¿QUÉ DIFERENCIA TU PROGRAMA CON EL RESTO?
En realidad, eso depende de con quién nos compares, si se trata de un sistema tradicional; por ejemplo, Icpna, británico o Euroidiomas la diferencia es abismal, en cuanto a la flexibilidad de horarios, número de personas en aula, público objetivo…
Si nos comparas con sistemas de avanzada como Wall Street, Berlitz, Open English, la diferencia está en detalles; por ejemplo, el manejo de los horarios: todos son flexibles y tú puedes elegir en qué horario ingresar, pero en algunos institutos con un día o hasta con una semana de anticipación y con nosotros solamente una hora antes.
Pero, sobre todo, la diferencia principal está en el servicio que recibes después de matricularte, hablamos de seguimiento y asesoría constante.

¿QUIÉN ME GARANTIZA QUE VOY A APRENDER?
La garantía de las instituciones educativas como universidades e institutos siempre será el propio alumno. Por ejemplo, el programa puede ser espectacular, pero si tú no cumples con las seis horas a la semana no lograrás el resultado que esperas porque no es un programa mágico.
Sin embargo, buscando garantizar tu aprendizaje, te damos la opción de repetir clases que no hayan sido comprendidas en un 100% con la finalidad de que no estés pasando por pasar, sino realmente aprendiendo. ¿Qué mejor garantía que esa? Es algo que ninguna otra institución te dará.

LA CLASE ES SIN COMPROMISO, ¿CIERTO?
El único compromiso es el de tu decisión, la idea de la clase es que además de la información que te estoy facilitando puedas vivir la experiencia con nosotros antes de matricularte. Si la clase no te gusta, simplemente quedamos en buenos términos y ya te diste la oportunidad de conocer una nueva alternativa. Pero si todo se te acomoda en cuanto a horarios, duración y metodología y, además de eso, la clase cubre tus expectativas te doy la cordial bienvenida a la institución y liberamos tu cupo académico mediante tu inscripción. ¿Estamos de acuerdo?

PLANES DE INVERSIÓN

PLANES REGULARES / GENERALES:
- Opción Contado 1: Matrícula S/ 3600 (Total: S/ 3600).
- Opción Contado 2: Matrícula S/ 3990 (Total: S/ 3990).
- Plan 2 Cuotas: Matrícula S/ 1900 + 2 cuotas de S/ 1200 cada una (Total: S/ 4300).
- Plan 4 Cuotas: Matrícula S/ 1400 + 4 cuotas de S/ 800 cada una (Total: S/ 4600).
- Plan 10 Cuotas: Matrícula S/ 1250 + 10 cuotas de S/ 350 cada una (Total: S/ 4750).

PLANES INHOUSE:
- Contado Inhouse: Matrícula S/ 6500 (Total: S/ 6500).
- Plan 2 Cuotas Inhouse: Matrícula S/ 2550 + 3 cuotas de S/ 1600 cada una (Total: S/ 7350).
- Plan 4 Cuotas Inhouse: Matrícula S/ 1850 + 6 cuotas de S/ 1100 cada una (Total: S/ 8450).

LINKS DE PAGO Y CUENTAS BANCARIAS

LINKS DE PAGO ESEC:
- Niubiz BBVA / Diners (Sin Monto): https://pagolink.niubiz.com.pe/pagoseguro/ESECPERU/3250291
- Mercado Pago BCP / Interbank (Sin Monto): https://link.mercadopago.com.pe/premiumenglishprogra
- Mercado Pago (Monto S/ 3600): https://mpago.la/2mgU3Bk
- Mercado Pago (Monto S/ 3990): https://mpago.la/2JapmHy

LINKS DE PAGO INTERPASS:
- Niubiz BBVA / Diners (Monto S/ 2500): https://pagolink.niubiz.com.pe/pagoseguro/PREMIUMENGLISHPROGRAM/722304
- Mercado Pago BCP / Interbank (Monto S/ 2500): https://mpago.la/1mLXnoU

CUENTAS BANCARIAS ESEC:
- BBVA ESEC: Cuenta 0011-0355-0100001919 | CCI: 011-355-000100001919-58
- BCP ESEC: Cuenta 194-1927206-0-26 | CCI: 002-194-001927206026-90
- INTERBANK ESEC: Cuenta 067-3000004152 | CCI: 003-067-003000004152-62
- YAPE ESEC (Máximo 500 soles): 962911799

CUENTAS BANCARIAS INTERPASS:
- BBVA INTERPASS: Cuenta 0011-0189-0100035765 | CCI: 011-189-000100035765-85

DATOS FISCALES Y RUCs:
- Razón Social: Escuela Sudamericana de Extensión Cultural para Perú S.A.
  RUC: 20428652577
- Razón Social: Premium Cultural Institute S.A.C.  
  RUC: 20502705131
- Razón Social: INTERNATIONAL PASSWORD SYSTEM PERU S.A.C.
  RUC: 20523074653

OFERTAS RELÁMPAGO Y PROMOCIONES:
- Oferta Expomina:
Vanessa tenemos una OFERTA RELÁMPAGO ⚡️ 
ÚLTIMOS CUPOS EXPOMINA🚨
De S/4500 a S/2500
✔️ Horarios flexibles
✔️ Clases personalizadas
✔️ TOEFL/CAMBRIDGE
¿Quieres recibir más información?

MENSAJES DE SEGUIMIENTO Y DEFINICIÓN (PLANTILLAS):
1. Mensaje de cierre o pausa de prioridad:
"Hola [Nombre], he estado intentando comunicarme debido a tu interés por mejorar tu inglés. Si en todo caso ya no lo estás viendo como una prioridad, déjame saberlo para retirarte de nuestra base de datos, pues la idea tampoco es incomodarte. Me comentas por favor"
2. Mensaje por falta de contacto previo:
"Hola [Nombre], he estado intentando comunicarme porque solicitaste información sobre nuestro programa de inglés pero no he podido contactarte. Si en todo caso ya no lo estás viendo como una prioridad, déjame saberlo para retirarte de nuestra base de datos, pues la idea tampoco es incomodarte. Me comentas por favor"
3. Mensaje de contacto inicial (Samy Pinto):
"Hola [Nombre], te saluda Samy Pinto de Premium English Program, solicitaste información sobre nuestro programa de inglés. Déjame saber en qué horario tienes disponible unos minutos para comunicarme."
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres un copiloto de telemercadeo en tiempo real para el equipo de ventas de Premium English.
Tu función es darle al asesor la información exacta o el guion que necesita de inmediato para la llamada.

Reglas de respuesta:
1. **Directo y sin rodeos:** Responde ÚNICAMENTE con los datos solicitados (como RUCs, razones sociales, cuentas bancarias, links, precios) o el guion exacto que debe leer el asesor. Jamás agregues saludos ni introducciones como "Dile esto:".
2. **Fidelidad al manual:** Utiliza la información textual exacta que figura en el manual proporcionado abajo.
3. **Versatilidad según la consulta:** 
   - Si piden un dato puntual (ej: "RUC", "cuenta", "oferta expomina", "planes"), entrégalo limpio y ordenado tal como figura en el manual.
   - Si es una objeción (ej: "caro", "no escuché antes"), entrega el guion de ventas exacto con su respectiva pregunta de cierre o instrucción si el manual lo indica.

Manual completo de la empresa:
{manual_completo}"""),
        ("human", "{{input}}")
    ])

    return prompt, llm

prompt, llm = cargar_componentes()

# --- GENERADOR DE STREAMING CON ESCUDO DE REINTENTOS ---
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
                    yield "⚠️ Límite momentáneo alcanzado. Por favor, reintenta en unos segundos."
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
        chain = prompt | llm | StrOutputParser()
        inputs = {"input": query_final}

        respuesta_texto = st.write_stream(stream_con_respaldo(chain, inputs))

    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})