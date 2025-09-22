import os 
import sys
import time 
import pickle 
from typing import List, Dict, Any 
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain.schema import HumanMessage, SystemMessage, AIMessage # Importa las clases de mensajes para estructurar las conversaciones con el LLM.
from langchain.memory import ConversationBufferMemory # Importa la clase para gestionar la memoria de la conversación.
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importa la clase para dividir textos largos en fragmentos (chunks).
from langchain_community.vectorstores import FAISS 
from langchain.docstore.document import Document 

class ChatbotEvaluacion: # Define la clase principal del chatbot.
    def __init__(self): # Define el método constructor que se ejecuta al crear una instancia de la clase.
        # Configuración del modelo de lenguaje (LLM) con streaming.
        self.llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",  # URL base de la API de modelos de GitHub.
            api_key=os.getenv("GITHUB_TOKEN"),     # Obtiene la clave de API desde las variables de entorno.
            model="openai/gpt-4o-mini", # Especifica el modelo a utilizar.
            temperature=0.7, # Define la "creatividad" del modelo (valores más altos son más creativos).
            streaming=True # Habilita el modo de streaming para recibir la respuesta token por token.
        )
        
        # Configuración del modelo de embeddings.
        self.embeddings = OpenAIEmbeddings(
            base_url="https://models.github.ai/inference", # URL base para el servicio de embeddings.
            api_key=os.getenv("GITHUB_TOKEN") # Usa la misma clave de API.
        )
        
        # Inicialización del sistema de memoria para la conversación.
        self.memory = ConversationBufferMemory(
            return_messages=True, # Configura la memoria para que devuelva los mensajes como objetos (HumanMessage, AIMessage).
            memory_key="chat_history" # Clave bajo la cual se guardará el historial en la memoria.
        )
        
        # Inicialización de la base de datos vectorial (inicialmente vacía).
        self.vectorstore = None
        # Configuración del divisor de texto.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Tamaño máximo de cada fragmento (chunk) en caracteres.
            chunk_overlap=200 # Número de caracteres que se solapan entre chunks consecutivos para mantener el contexto.
        )
        
        # El RAG (Retrieval-Augmented Generation) se inicializará más tarde a través de la interfaz de usuario.
    
    def inicializar_rag(self, docs: List[Document]): # Define un método para inicializar el sistema RAG.
        """Inicializa el sistema RAG con los documentos proporcionados"""
        if not docs: # Comprueba si la lista de documentos está vacía.
            st.error("No se proporcionaron documentos para inicializar RAG.") # Muestra un error en la UI si no hay documentos.
            return # Termina la ejecución del método.
            
        st.info("📚 Inicializando sistema RAG...") # Muestra un mensaje informativo en la UI.
        
        # Divide los documentos en fragmentos (chunks) usando el divisor de texto configurado.
        chunks = self.text_splitter.split_documents(docs)
        st.success(f"✓ Divididos {len(chunks)} chunks") # Muestra un mensaje de éxito con el número de chunks creados.
        
        # Genera los embeddings para cada chunk y crea la base de datos vectorial FAISS.
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        st.success("✓ Base de datos vectorial FAISS creada") # Muestra un mensaje de éxito.
        
        # Guarda la base de datos vectorial en el disco local.
        self.vectorstore.save_local("faiss_index")
        st.success("✓ Base de datos guardada en 'faiss_index'") # Muestra un mensaje de éxito.
    
    def cargar_rag(self, ruta: str = "faiss_index"): # Define un método para cargar una base de datos RAG existente.
        """Carga una base de datos vectorial existente"""
        if os.path.exists(ruta): # Comprueba si la ruta especificada existe en el disco.
            # Carga la base de datos FAISS desde el disco, usando el modelo de embeddings configurado.
            # allow_dangerous_deserialization es necesario por seguridad en versiones recientes de LangChain.
            self.vectorstore = FAISS.load_local(ruta, self.embeddings, allow_dangerous_deserialization=True)
            st.success("✓ Base de datos vectorial cargada") # Muestra un mensaje de éxito.
            return True # Devuelve True si la carga fue exitosa.
        return False # Devuelve False si la ruta no existe.
    
    def buscar_contexto(self, pregunta: str, k: int = 3) -> str: # Define un método para buscar contexto relevante.
        """Busca contexto relevante en la base de datos vectorial"""
        if not self.vectorstore: # Comprueba si la base de datos vectorial ha sido inicializada.
            return "Sistema RAG no inicializado. Por favor, carga documentos primero." # Devuelve un mensaje de advertencia.
        
        # Realiza una búsqueda por similitud en la base de datos vectorial.
        docs = self.vectorstore.similarity_search(pregunta, k=k) # 'k' es el número de documentos más relevantes a recuperar.
        # Une el contenido de los documentos recuperados en un solo string.
        contexto = "\n\n".join([doc.page_content for doc in docs])
        return f"Contexto relevante:\n{contexto}" # Devuelve el contexto formateado.
    
    def generar_respuesta_con_streaming(self, pregunta: str, mostrar_contexto: bool = False): # Define el método principal para generar respuestas.
        """Genera respuesta con streaming integrando RAG y memoria"""
        
        # Busca contexto relevante en la base de datos vectorial si está disponible.
        contexto = self.buscar_contexto(pregunta) if self.vectorstore else ""
        
        # Carga el historial de la conversación desde la memoria.
        historial = self.memory.load_memory_variables({})["chat_history"]
        
        # Construye el prompt del sistema que instruye al LLM sobre su rol y comportamiento.
        system_prompt = """
Eres un asistente de soporte informático experto para oficinas. 
Tu conocimiento se basa **exclusivamente** en el material proporcionado en el archivo 'soporte_informatica.txt'.

Directrices:
1. Responde de manera clara, concisa y profesional.
2. Solo utiliza la información contenida en el archivo; **no inventes respuestas ni uses conocimiento externo**.
3. Si la pregunta no puede ser respondida con la información disponible, responde amablemente:
   "La información para responder a tu pregunta no se encuentra en el material disponible."
4. Siempre prioriza la seguridad informática y las buenas prácticas de oficina.
5. Para problemas técnicos, proporciona pasos prácticos y ordenados.
6. Mantén un tono amigable y de ayuda.

Formato:
- Si es un procedimiento paso a paso, enumera claramente los pasos.
- Para información de contacto o horarios, inclúyelos tal como aparecen en el material.
"""

        
        # Inicializa la lista de mensajes con el prompt del sistema.
        messages = [SystemMessage(content=system_prompt)]
        
        # Agrega el historial de la conversación a la lista de mensajes.
        for msg in historial:
            messages.append(msg)
        
        # Si se debe mostrar el contexto, lo agrega como un mensaje del sistema.
        if contexto and mostrar_contexto:
            messages.append(SystemMessage(content=contexto))
        
        # Agrega la pregunta actual del usuario como un mensaje humano.
        messages.append(HumanMessage(content=pregunta))
        
        # Inicia la generación de respuesta con streaming.
        respuesta_completa = "" # Variable para almacenar la respuesta completa.
        try:
            # Crea un contenedor vacío en Streamlit que se actualizará dinámicamente.
            respuesta_placeholder = st.empty()
            respuesta_texto = "🤖 **Asistente:** " # Texto inicial de la respuesta.
            
            # Itera sobre los chunks de respuesta que llegan desde el LLM en modo streaming.
            for chunk in self.llm.stream(messages):
                contenido = chunk.content # Obtiene el contenido del chunk actual.
                respuesta_texto += contenido # Agrega el contenido al texto de la respuesta.
                respuesta_placeholder.markdown(respuesta_texto + "▌") # Actualiza el placeholder con el texto y un cursor parpadeante.
                respuesta_completa += contenido # Acumula el contenido en la variable de respuesta completa.
                time.sleep(0.02) # Pequeña pausa para un efecto visual más suave.
            
            # Muestra la respuesta final en el placeholder sin el cursor.
            respuesta_placeholder.markdown(respuesta_texto)
            
            # Guarda el par pregunta/respuesta en la memoria para mantener el contexto de la conversación.
            self.memory.save_context(
                {"input": pregunta},
                {"output": respuesta_completa}
            )
            
            return respuesta_completa # Devuelve la respuesta completa generada.
            
        except Exception as e: # Maneja posibles errores durante la generación.
            error_msg = f"❌ Error en la generación: {e}" # Formatea el mensaje de error.
            st.error(error_msg) # Muestra el error en la interfaz de Streamlit.
            return error_msg # Devuelve el mensaje de error.
    
    def evaluar_respuesta(self, pregunta: str, respuesta_usuario: str) -> Dict[str, Any]: # Define un método para evaluar la respuesta de un usuario.
        """Evalúa una respuesta del usuario comparándola con el contexto"""
        if not self.vectorstore: # Comprueba si el sistema RAG está inicializado.
            return {"error": "Sistema RAG no inicializado"} # Devuelve un error si no lo está.
        
        # Busca el contexto relevante para la pregunta.
        contexto = self.buscar_contexto(pregunta)
        
        # Crea un prompt específico para que el LLM evalúe la respuesta del estudiante.
        prompt_evaluacion = f'''
        Evalúa la siguiente respuesta del estudiante:
        
        Pregunta: {pregunta}
        Respuesta del estudiante: {respuesta_usuario}
        
        Contexto de referencia: {contexto}
        
        Proporciona una evaluación con:
        1. Puntuación (0-10)
        2. Puntos fuertes
        3. Áreas de mejora
        4. Explicación breve
        '''
        
        try:
            # Llama al LLM con el prompt de evaluación.
            respuesta = self.llm.invoke([HumanMessage(content=prompt_evaluacion)])
            # Devuelve un diccionario con el feedback y el contexto utilizado.
            return {
                "puntuacion": 0,  # Placeholder, idealmente se extraería del 'feedback' con parsing.
                "feedback": respuesta.content, # El feedback generado por el LLM.
                "contexto_utilizado": contexto[:500] + "..." if len(contexto) > 500 else contexto # Muestra una parte del contexto.
            }
        except Exception as e: # Maneja errores durante la evaluación.
            return {"error": f"Error en evaluación: {e}"}
    
    def mostrar_estadisticas(self): # Define un método para mostrar estadísticas.
        """Muestra estadísticas del chatbot"""
        # Carga el historial de la conversación desde la memoria.
        historial = self.memory.load_memory_variables({})["chat_history"]
        st.subheader("📊 Estadísticas del Chatbot") # Título para la sección de estadísticas.
        st.write(f"💬 **Mensajes en memoria:** {len(historial)}") # Muestra el número de mensajes en memoria.
        st.write(f"📚 **Base vectorial:** {'✓ Cargada' if self.vectorstore else '✗ No disponible'}") # Indica si la base vectorial está cargada.
        if self.vectorstore: # Si la base vectorial existe...
            st.write(f"📖 **Documentos indexados:** {self.vectorstore.index.ntotal}") # Muestra el número de documentos (chunks) indexados.

# FUNCIONES DE UTILIDAD PARA LA EVALUACIÓN

def cargar_material() -> List[Document]: # Define una función para cargar los archivos de material de estudio.
    """Carga el material de estudio de los módulos RA1 desde archivos markdown."""
    st.info("Cargando material de estudio de RA1...") # Mensaje informativo en la UI.
    documentos = [] # Inicializa una lista vacía para almacenar los documentos.
    # Define un diccionario con nombres lógicos y rutas relativas a los archivos.
    rutas_archivos = {
        "soporte_informatica": "soporte_informatica.txt",
    }
    
    # Obtiene la ruta absoluta del directorio donde se encuentra el script actual.
    dir_actual = os.path.dirname(os.path.abspath(__file__))

    # Itera sobre el diccionario de rutas de archivos.
    for nombre, ruta_relativa in rutas_archivos.items():
        try:
            # Construye la ruta absoluta del archivo.
            ruta_absoluta = os.path.normpath(os.path.join(dir_actual, ruta_relativa))
            # Abre y lee el archivo con codificación UTF-8.
            with open(ruta_absoluta, 'r', encoding='utf-8') as f:
                contenido = f.read() # Lee todo el contenido del archivo.
                # Lógica especial para el archivo de Evaluaciones: solo carga hasta la formativa 2.
                if "Evaluaciones" in nombre:
                    if "## EVALUACIÓN FORMATIVA 2" in contenido:
                        contenido = contenido.split("## EVALUACIÓN FORMATIVA 2")[0] # Trunca el contenido.

                # Crea un objeto Document de LangChain con el contenido y metadatos.
                documentos.append(Document(
                    page_content=contenido,
                    metadata={"source": nombre, "file_path": ruta_relativa}
                ))
            st.write(f"✓ Cargado: {nombre}") # Mensaje de éxito por cada archivo cargado.
        except FileNotFoundError: # Maneja el error si un archivo no se encuentra.
            st.warning(f"⚠️ No se encontró el archivo: {ruta_absoluta}")
        except Exception as e: # Maneja cualquier otro error durante la carga.
            st.error(f"❌ Error al cargar {ruta_absoluta}: {e}")
            
    if not documentos: # Si no se cargó ningún documento...
        st.error("No se pudo cargar ningún material de estudio. El chatbot no tendrá contexto.") # Muestra un error.
    
    return documentos # Devuelve la lista de objetos Document.

def main(): # Función principal que define la aplicación Streamlit.
    st.set_page_config( # Configura la página de la aplicación.
        page_title="Asistente de Soporte en informatica", # Título que aparece en la pestaña del navegador.
        page_icon="🤖", # Ícono de la página.
        layout="wide" # Utiliza un diseño de página ancho.
    )
    
    st.title("🤖 Asistente de Soporte en informatica") # Título principal de la aplicación.
    st.markdown("Usa este chatbot para preguntar sobre Soporte en informatica.") # Subtítulo o descripción.
    
    # Inicializa el chatbot en el estado de la sesión de Streamlit si no existe.
    # Esto asegura que el objeto chatbot persista entre interacciones del usuario.
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatbotEvaluacion()
    
    # Crea una barra lateral para las opciones de configuración.
    with st.sidebar:
        st.header("Configuración") # Título de la barra lateral.
        
        # Botón para cargar el material de estudio.
        if st.button("📚 Cargar material de Soporte en informatica"):
            docs = cargar_material() # Llama a la función para cargar los archivos.
            if docs: # Si se cargaron documentos...
                st.session_state.chatbot.inicializar_rag(docs) # Inicializa el RAG con los documentos.
        
        # Botón para cargar una base de datos vectorial existente.
        if st.button("🔍 Cargar base de datos existente"):
            if st.session_state.chatbot.cargar_rag(): # Intenta cargar la base de datos.
                st.success("Base de datos cargada exitosamente") # Mensaje de éxito.
            else:
                st.error("No se encontró base de datos existente") # Mensaje de error.
        
        # Botón para mostrar estadísticas del chatbot.
        if st.button("📊 Mostrar estadísticas"):
            st.session_state.chatbot.mostrar_estadisticas() # Llama al método para mostrar estadísticas.
        
        # Botón para limpiar la memoria de la conversación.
        if st.button("🗑️ Limpiar memoria"):
            st.session_state.chatbot.memory.clear() # Limpia la memoria.
            st.success("Memoria limpiada") # Mensaje de éxito.
        
        st.markdown("---") # Línea divisoria.
        st.markdown("### Modos de uso") # Subtítulo para los modos.
        # Botones de radio para seleccionar el modo de operación.
        modo = st.radio("Selecciona el modo:", 
                       ["💬 Chat", "📝 Evaluación"])
    
    # Lógica para el contenido principal según el modo seleccionado.
    if modo == "💬 Chat":
        st.header("💬 Modo Chat") # Título para el modo chat.
        
        # Muestra el historial de chat si existe.
        if hasattr(st.session_state.chatbot, 'memory'):
            historial = st.session_state.chatbot.memory.load_memory_variables({})["chat_history"]
            if historial:
                with st.expander("📜 Historial de conversación"): # Crea una sección expandible.
                    for i, msg in enumerate(historial): # Itera sobre los mensajes del historial.
                        if isinstance(msg, HumanMessage): # Si es un mensaje de usuario...
                            st.write(f"**🧑 Usuario:** {msg.content}")
                        elif isinstance(msg, AIMessage): # Si es un mensaje del asistente...
                            st.write(f"**🤖 Asistente:** {msg.content}")
        
        # Área de texto para que el usuario ingrese su pregunta.
        pregunta = st.text_area("Escribe tu pregunta:", placeholder="¿En qué puedo ayudarte?")
        
        col1, col2 = st.columns([1, 5]) # Divide el espacio en dos columnas.
        with col1:
            # Checkbox para decidir si se muestra el contexto recuperado por RAG.
            mostrar_contexto = st.checkbox("Mostrar contexto", value=False)
        with col2:
            # Botón para enviar la pregunta.
            if st.button("🚀 Enviar pregunta", type="primary"):
                if pregunta.strip(): # Comprueba que la pregunta no esté vacía.
                    with st.spinner("Generando respuesta..."): # Muestra un indicador de carga.
                        # Llama al método para generar la respuesta.
                        st.session_state.chatbot.generar_respuesta_con_streaming(
                            pregunta, 
                            mostrar_contexto=mostrar_contexto
                        )
                else:
                    st.warning("Por favor, escribe una pregunta") # Advierte si la pregunta está vacía.
    
    elif modo == "📝 Evaluación":
        st.header("📝 Modo Evaluación") # Título para el modo evaluación.
        
        st.markdown("Evalúa una respuesta del estudiante comparándola con el contexto disponible.") # Descripción.
        
        col1, col2 = st.columns(2) # Divide el espacio en dos columnas.
        
        with col1:
            # Área de texto para la pregunta de evaluación.
            pregunta_eval = st.text_area(
                "Pregunta de evaluación:",
                placeholder="Ej: ¿Qué es la inteligencia artificial?",
                height=100
            )
        
        with col2:
            # Área de texto para la respuesta del estudiante.
            respuesta_estudiante = st.text_area(
                "Respuesta del estudiante:",
                placeholder="Ej: La IA es cuando las máquinas piensan como humanos",
                height=100
            )
        
        # Botón para iniciar la evaluación.
        if st.button("📋 Evaluar respuesta", type="primary"):
            if pregunta_eval.strip() and respuesta_estudiante.strip(): # Comprueba que ambos campos estén llenos.
                with st.spinner("Evaluando respuesta..."): # Muestra un indicador de carga.
                    # Llama al método de evaluación.
                    evaluacion = st.session_state.chatbot.evaluar_respuesta(
                        pregunta_eval, 
                        respuesta_estudiante
                    )
                    
                    if "error" not in evaluacion: # Si no hubo error en la evaluación...
                        st.subheader("Resultado de la evaluación") # Muestra el resultado.
                        st.markdown(evaluacion["feedback"]) # Muestra el feedback del LLM.
                        
                        with st.expander("📚 Contexto utilizado"): # Sección expandible para el contexto.
                            st.text(evaluacion["contexto_utilizado"])
                    else:
                        st.error(evaluacion["error"]) # Muestra el error si ocurrió uno.
            else:
                st.warning("Por favor, completa ambos campos") # Advierte si faltan campos.

    # Pie de página de la aplicación.
    st.markdown("---")
    st.markdown("*Asistente de estudio con RAG y memoria de conversación para RA1*")

if __name__ == "__main__": # Punto de entrada del script. Se ejecuta solo si el archivo es corrido directamente.
    # Verifica si la variable de entorno GITHUB_TOKEN está configurada.
    if not os.getenv("GITHUB_TOKEN"):
        st.error("⚠️ Por favor, configura la variable de entorno GITHUB_TOKEN") # Muestra un error si no está configurada.
        st.info("""
        Para usar esta aplicación, necesitas:
        1. Obtener un token de GitHub con permisos adecuados
        2. Configurarlo como variable de entorno:
           ```bash
           export GITHUB_TOKEN="tu_token_aqui"
           ```
        """)
    else:
        main() # Llama a la función principal para iniciar la aplicación.