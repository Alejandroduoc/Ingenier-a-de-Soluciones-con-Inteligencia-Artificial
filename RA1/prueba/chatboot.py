import os
import sys
import time
import pickle
from typing import List, Dict, Any
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class ChatbotEvaluacion:
    def __init__(self, documentos: List[str] = None):
        # Configuración del modelo con streaming
        self.llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",  #descubri que aqui  cambia de url base  
            api_key=os.getenv("GITHUB_TOKEN"),     
            model="openai/gpt-4o-mini",
            temperature=0.7,
            streaming=True
        )
        
        # Modelo de embeddings
        self.embeddings = OpenAIEmbeddings(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN")
        )
        
        # Sistema de memoria
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Base de datos vectorial
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Inicializar con documentos si se proporcionan
        if documentos:
            self.inicializar_rag(documentos)
    
    def inicializar_rag(self, documentos: List[str]):
        """Inicializa el sistema RAG con los documentos proporcionados"""
        st.info("📚 Inicializando sistema RAG...")
        
        # Crear documentos
        docs = []
        for i, texto in enumerate(documentos):
            docs.append(Document(
                page_content=texto,
                metadata={"source": f"doc_{i}", "page": 1}
            ))
        
        # Dividir en chunks
        chunks = self.text_splitter.split_documents(docs)
        st.success(f"✓ Divididos {len(chunks)} chunks")
        
        # Generar embeddings y crear base vectorial
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        st.success("✓ Base de datos vectorial FAISS creada")
        
        # Guardar la base de datos
        self.vectorstore.save_local("faiss_index")
        st.success("✓ Base de datos guardada en 'faiss_index'")
    
    def cargar_rag(self, ruta: str = "faiss_index"):
        """Carga una base de datos vectorial existente"""
        if os.path.exists(ruta):
            self.vectorstore = FAISS.load_local(ruta, self.embeddings)
            st.success("✓ Base de datos vectorial cargada")
            return True
        return False
    
    def buscar_contexto(self, pregunta: str, k: int = 3) -> str:
        """Busca contexto relevante en la base de datos vectorial"""
        if not self.vectorstore:
            return "Sistema RAG no inicializado. Por favor, carga documentos primero."
        
        docs = self.vectorstore.similarity_search(pregunta, k=k)
        contexto = "\n\n".join([doc.page_content for doc in docs])
        return f"Contexto relevante:\n{contexto}"
    
    def generar_respuesta_con_streaming(self, pregunta: str, mostrar_contexto: bool = False):
        """Genera respuesta con streaming integrando RAG y memoria"""
        
        # Buscar contexto relevante
        contexto = self.buscar_contexto(pregunta) if self.vectorstore else ""
        
        # Obtener historial de conversación
        historial = self.memory.load_memory_variables({})["chat_history"]
        
        # Construir mensajes del sistema
        system_prompt = """Eres un asistente educativo especializado en evaluaciones. 
        Responde basándote en el contexto proporcionado y mantén un tono profesional.
        Si la información no está en el contexto, indica claramente que no tienes esa información."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # Agregar historial de conversación
        for msg in historial:
            messages.append(msg)
        
        # Agregar contexto y pregunta actual
        if contexto and mostrar_contexto:
            messages.append(SystemMessage(content=contexto))
        
        messages.append(HumanMessage(content=pregunta))
        
        # Generar respuesta con streaming
        respuesta_completa = ""
        try:
            # Crear un placeholder para la respuesta en streaming
            respuesta_placeholder = st.empty()
            respuesta_texto = "🤖 **Asistente:** "
            
            for chunk in self.llm.stream(messages):
                contenido = chunk.content
                respuesta_texto += contenido
                respuesta_placeholder.markdown(respuesta_texto + "▌")
                respuesta_completa += contenido
                time.sleep(0.02)
            
            # Mostrar respuesta final sin el cursor
            respuesta_placeholder.markdown(respuesta_texto)
            
            # Guardar en memoria
            self.memory.save_context(
                {"input": pregunta},
                {"output": respuesta_completa}
            )
            
            return respuesta_completa
            
        except Exception as e:
            error_msg = f"❌ Error en la generación: {e}"
            st.error(error_msg)
            return error_msg
    
    def evaluar_respuesta(self, pregunta: str, respuesta_usuario: str) -> Dict[str, Any]:
        """Evalúa una respuesta del usuario comparándola con el contexto"""
        if not self.vectorstore:
            return {"error": "Sistema RAG no inicializado"}
        
        # Buscar contexto relevante
        contexto = self.buscar_contexto(pregunta)
        
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
            respuesta = self.llm.invoke([HumanMessage(content=prompt_evaluacion)])
            return {
                "puntuacion": 0,  # Se extraería del análisis
                "feedback": respuesta.content,
                "contexto_utilizado": contexto[:500] + "..." if len(contexto) > 500 else contexto
            }
        except Exception as e:
            return {"error": f"Error en evaluación: {e}"}
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas del chatbot"""
        historial = self.memory.load_memory_variables({})["chat_history"]
        st.subheader("📊 Estadísticas del Chatbot")
        st.write(f"💬 **Mensajes en memoria:** {len(historial)}")
        st.write(f"📚 **Base vectorial:** {'✓ Cargada' if self.vectorstore else '✗ No disponible'}")
        if self.vectorstore:
            st.write(f"📖 **Documentos indexados:** {self.vectorstore.index.ntotal}")

# FUNCIONES DE UTILIDAD PARA LA EVALUACIÓN

def inicializar_documentos_ejemplo() -> List[str]:
    """Proporciona documentos de ejemplo para la evaluación"""
    documentos = [
        """
        La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por máquinas, 
        especialmente sistemas informáticos. Estos procesos incluyen el aprendizaje, el razonamiento 
        y la autocorrección.
        """,
        """
        El machine learning es un subconjunto de la IA que se centra en el desarrollo de algoritmos 
        que permiten a las computadoras aprender y hacer predicciones basadas en datos.
        """,
        """
        LangChain es un framework para desarrollar aplicaciones con modelos de lenguaje. 
        Proporciona herramientas para la gestión de memoria, cadenas de procesamiento 
        y integración con bases de datos vectoriales.
        """,
        """
        Los embeddings son representaciones vectoriales de texto que capturan el significado semántico. 
        Se utilizan en búsqueda semántica y sistemas de recomendación.
        """,
        """
        FAISS (Facebook AI Similarity Search) es una biblioteca para búsqueda eficiente de similitudes 
        en vectores de alta dimensión. Es ideal para sistemas de recuperación de información.
        """
    ]
    return documentos

def main():
    st.set_page_config(
        page_title="Chatbot de Evaluación Educativa",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Chatbot de Evaluación Educativa")
    st.markdown("Sistema RAG con memoria y evaluación de respuestas")
    
    # Inicializar el chatbot en session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatbotEvaluacion()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración")
        
        # Opción para cargar documentos
        if st.button("📚 Cargar documentos de ejemplo"):
            documentos = inicializar_documentos_ejemplo()
            st.session_state.chatbot.inicializar_rag(documentos)
        
        # Opción para cargar base existente
        if st.button("🔍 Cargar base de datos existente"):
            if st.session_state.chatbot.cargar_rag():
                st.success("Base de datos cargada exitosamente")
            else:
                st.error("No se encontró base de datos existente")
        
        # Mostrar estadísticas
        if st.button("📊 Mostrar estadísticas"):
            st.session_state.chatbot.mostrar_estadisticas()
        
        # Limpiar memoria
        if st.button("🗑️ Limpiar memoria"):
            st.session_state.chatbot.memory.clear()
            st.success("Memoria limpiada")
        
        st.markdown("---")
        st.markdown("### Modos de uso")
        modo = st.radio("Selecciona el modo:", 
                       ["💬 Chat", "📝 Evaluación"])
    
    # Contenido principal según el modo seleccionado
    if modo == "💬 Chat":
        st.header("💬 Modo Chat")
        
        # Mostrar historial de chat
        if hasattr(st.session_state.chatbot, 'memory'):
            historial = st.session_state.chatbot.memory.load_memory_variables({})["chat_history"]
            if historial:
                with st.expander("📜 Historial de conversación"):
                    for i, msg in enumerate(historial):
                        if isinstance(msg, HumanMessage):
                            st.write(f"**🧑 Usuario:** {msg.content}")
                        elif isinstance(msg, AIMessage):
                            st.write(f"**🤖 Asistente:** {msg.content}")
        
        # Input del usuario
        pregunta = st.text_area("Escribe tu pregunta:", placeholder="¿En qué puedo ayudarte?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            mostrar_contexto = st.checkbox("Mostrar contexto", value=False)
        with col2:
            if st.button("🚀 Enviar pregunta", type="primary"):
                if pregunta.strip():
                    with st.spinner("Generando respuesta..."):
                        st.session_state.chatbot.generar_respuesta_con_streaming(
                            pregunta, 
                            mostrar_contexto=mostrar_contexto
                        )
                else:
                    st.warning("Por favor, escribe una pregunta")
    
    elif modo == "📝 Evaluación":
        st.header("📝 Modo Evaluación")
        
        st.markdown("Evalúa una respuesta del estudiante comparándola con el contexto disponible.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregunta_eval = st.text_area(
                "Pregunta de evaluación:",
                placeholder="Ej: ¿Qué es la inteligencia artificial?",
                height=100
            )
        
        with col2:
            respuesta_estudiante = st.text_area(
                "Respuesta del estudiante:",
                placeholder="Ej: La IA es cuando las máquinas piensan como humanos",
                height=100
            )
        
        if st.button("📋 Evaluar respuesta", type="primary"):
            if pregunta_eval.strip() and respuesta_estudiante.strip():
                with st.spinner("Evaluando respuesta..."):
                    evaluacion = st.session_state.chatbot.evaluar_respuesta(
                        pregunta_eval, 
                        respuesta_estudiante
                    )
                    
                    if "error" not in evaluacion:
                        st.subheader("Resultado de la evaluación")
                        st.markdown(evaluacion["feedback"])
                        
                        with st.expander("📚 Contexto utilizado"):
                            st.text(evaluacion["contexto_utilizado"])
                    else:
                        st.error(evaluacion["error"])
            else:
                st.warning("Por favor, completa ambos campos")

    # Footer
    st.markdown("---")
    st.markdown("*Sistema de evaluación educativa con RAG y memoria de conversación*")

if __name__ == "__main__":
    # Verificar que la API key esté configurada
    if not os.getenv("GITHUB_TOKEN"):
        st.error("⚠️ Por favor, configura la variable de entorno GITHUB_TOKEN")
        st.info("""
        Para usar esta aplicación, necesitas:
        1. Obtener un token de GitHub con permisos adecuados
        2. Configurarlo como variable de entorno:
           ```bash
           export GITHUB_TOKEN="tu_token_aqui"
           ```
        """)
    else:
        main()