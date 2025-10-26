import os
import glob
import json
from typing import List, Dict, Any

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory # Requiere `pip install langchain`
from langchain.text_splitter import RecursiveCharacterTextSplitter # Requiere `pip install langchain`
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# --- 1. Definición de Herramientas Especializadas (Concepto de RA2) ---
@tool
def knowledge_base_tool(query: str) -> str:
    """
    Busca en la base de conocimientos del curso (documentos .md y .py de RA1 y RA2)
    para encontrar información relevante a la consulta del usuario. Es la herramienta
    principal para responder preguntas sobre el CONTENIDO del curso, como conceptos,
    definiciones o explicaciones.
    """
    # La lógica de búsqueda se inyecta desde el orquestador
    if 'orquestador' in st.session_state and st.session_state.orquestador.vectorstore:
        docs = st.session_state.orquestador.vectorstore.similarity_search(query, k=4)
        if not docs:
            return "No se encontró información relevante en la base de conocimientos."
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Contexto encontrado:\n{context}"
    return "La base de conocimientos no está disponible o inicializada."

@tool
def project_structure_tool(directory: str) -> str:
    """
    Analiza y lista la estructura de archivos de un directorio específico del proyecto
    (ej. 'RA1/IL1.3' o 'RA2/IL2.3'). Es útil para responder preguntas sobre la
    organización del curso, los temas cubiertos en cada módulo o qué archivos existen.
    """
    # La ruta base se inyecta desde el orquestador
    if 'orquestador' in st.session_state:
        project_root = st.session_state.orquestador.project_root
        path_to_scan = os.path.join(project_root, directory)
        if not os.path.isdir(path_to_scan):
            return f"Error: El directorio '{directory}' no existe."
        
        files = glob.glob(os.path.join(path_to_scan, '**', '*.*'), recursive=True)
        relative_files = [os.path.relpath(f, project_root) for f in files]
        
        if not relative_files:
            return f"No se encontraron archivos en el directorio '{directory}'."
            
        return f"Archivos encontrados en '{directory}':\n" + "\n".join(relative_files)
    return "El orquestador no está inicializado para acceder a la estructura del proyecto."


# --- 2. Definición de Agentes Especializados (Concepto de IL2.3) ---

class SpecializedAgent:
    """Clase base para un agente especializado con un rol y herramientas."""
    def __init__(self, name: str, role: str, tools: List[BaseTool], llm: ChatOpenAI):
        self.name = name
        self.role = role
        self.tools = tools
        self.llm = llm

    def __str__(self):
        return f"Agente(name={self.name}, role={self.role}, tools={[t.name for t in self.tools]})"

class OrquestadorCurso:
    """
    Orquesta un equipo de agentes especializados para responder preguntas sobre el curso.
    Aplica conceptos de RAG (RA1), Herramientas (RA2) y Orquestación Multi-Agente (IL2.3).
    Su rol es delegar tareas, no ejecutarlas directamente.
    """
    def __init__(self):
        # --- Configuración Global (LLM y Embeddings) ---
        self.llm = ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
            model="openai/gpt-4o-mini",
            temperature=0.5,
            streaming=False # Desactivado para compatibilidad con AgentExecutor
        )
        self.embeddings = OpenAIEmbeddings(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN")
        )
        
        # --- Memoria Centralizada (Concepto de RA2) ---
        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )
        
        # --- Componentes RAG Centralizados (Concepto de RA1) ---
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=250
        )
        
        # --- 3. Creación del Equipo de Agentes (Concepto de Orquestación de IL2.3) ---
        self.agente_contenido = SpecializedAgent(
            name="AgenteContenidoRAG",
            role="Experto en el contenido y los conceptos del curso. Usa la base de conocimientos para responder preguntas.",
            tools=[knowledge_base_tool],
            llm=self.llm
        )
        
        self.agente_planificador = SpecializedAgent(
            name="AgentePlanificador",
            role="Experto en la estructura y organización del proyecto. Usa la herramienta de estructura de proyecto para listar archivos y entender la planificación.",
            tools=[project_structure_tool],
            llm=self.llm
        )

        self.team = {
            "contenido": self.agente_contenido,
            "planificacion": self.agente_planificador
        }
        
        # Ruta raíz del proyecto para la herramienta de estructura
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.normpath(os.path.join(base_path, '..', '..'))
        
    def inicializar_rag_con_contenido_curso(self):
        """Carga, divide e indexa el contenido de RA1 y RA2."""
        st.info("📚 Cargando material de estudio de RA1 y RA2...")
        
        # --- Carga de Documentos (Lógica de RAG de RA1) ---
        docs = []
        # Asume que el script se ejecuta desde la raíz del proyecto o ajusta la ruta
        
        rutas_a_buscar = [
            os.path.join(self.project_root, 'RA1', '**', '*.md'),
            os.path.join(self.project_root, 'RA2', '**', '*.md'),
            os.path.join(self.project_root, 'RA1', '**', '*.py'),
            os.path.join(self.project_root, 'RA2', '**', '*.py')
        ]
        
        archivos_md = []
        for ruta in rutas_a_buscar:
            archivos_md.extend(glob.glob(ruta, recursive=True))

        for file_path in archivos_md:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    contenido = f.read()
                    docs.append(Document(
                        page_content=contenido,
                        metadata={"source": os.path.basename(file_path)}
                    ))
                st.write(f"✓ Cargado: {os.path.relpath(file_path, self.project_root)}")
            except Exception as e:
                st.warning(f"⚠️ Error al cargar {file_path}: {e}")

        if not docs:
            st.error("No se pudo cargar ningún documento. El agente no tendrá conocimiento.")
            return

        # --- Creación de la Base de Datos Vectorial (RAG de RA1) ---
        chunks = self.text_splitter.split_documents(docs)
        st.success(f"✓ Documentos divididos en {len(chunks)} chunks.")
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        st.success("✓ Base de datos vectorial (FAISS) creada con éxito.")

    def _delegar_tarea(self, pregunta: str) -> SpecializedAgent:
        """
        Paso de Orquestación: Decide qué agente es el más adecuado para la tarea.
        Inspirado en `5-agent-orchestration.py` y `9-multi-agent-coordination.py`.
        """
        prompt_delegacion = f"""
Eres un orquestador de agentes de IA. Tu trabajo es analizar la pregunta del usuario y delegarla al agente más adecuado.

Aquí está tu equipo de agentes especializados:
1.  **Agente de Contenido (clave: 'contenido')**:
    - Rol: {self.agente_contenido.role}
    - Ideal para: Preguntas sobre qué es RAG, cómo funciona un LLM, explicaciones de código, conceptos teóricos.

2.  **Agente de Planificación (clave: 'planificacion')**:
    - Rol: {self.agente_planificador.role}
    - Ideal para: Preguntas sobre la estructura del curso, qué archivos hay en una carpeta, cómo se organizan los módulos, qué temas se ven en IL2.3.

Pregunta del usuario: "{pregunta}"

Analiza la pregunta y responde SOLAMENTE con la clave del agente más adecuado (ej. 'contenido' o 'planificacion').
Si la pregunta es un saludo o no encaja, responde 'contenido' por defecto.
"""
        response = self.llm.invoke(prompt_delegacion)
        decision = response.content.strip().lower()
        
        st.info(f"🧠 **Orquestador:** Delegando a agente de `{decision}`.")
        
        return self.team.get(decision, self.agente_contenido) # Default to content agent

    def ejecutar_agente_especializado(self, agente: SpecializedAgent, pregunta: str):
        """
        Ejecuta un agente especializado para que use sus herramientas y responda.
        Esto hace que el agente sea autónomo para decidir cómo usar sus herramientas.
        """
        if not self.vectorstore:
            st.warning("La base de conocimientos no ha sido inicializada. Por favor, carga el material primero.")
            return

        # 1. Crear un prompt específico para el agente, dándole su rol y herramientas.
        prompt_agente = f"""
Eres un agente asistente especializado. Tu nombre es {agente.name} y tu rol es: {agente.role}.
Usa tus herramientas disponibles para responder a la pregunta del usuario.
Basa tus respuestas únicamente en la información que obtengas de tus herramientas.
"""
        historial = self.memory.load_memory_variables({})["chat_history"]
        
        # 2. Crear un "ejecutor de agente" que une el LLM, las herramientas y el prompt.
        # Esto permite al LLM decidir qué herramienta usar y con qué argumentos.
        agent_runnable = create_openai_tools_agent(agente.llm, agente.tools, SystemMessage(content=prompt_agente))
        agent_executor = AgentExecutor(agent=agent_runnable, tools=agente.tools, verbose=True) # verbose=True para depuración

        # 3. Invocar al agente y hacer streaming de la respuesta.
        respuesta_completa = ""
        respuesta_placeholder = st.empty()
        respuesta_texto = f"🤖 **{agente.name}:** "
        
        # Usamos st.spinner para mostrar que el agente está "pensando" (usando herramientas)
        with st.spinner(f"**{agente.name}** está trabajando..."):
            response = agent_executor.invoke({
                "input": pregunta,
                "chat_history": historial
            })
            respuesta_completa = response.get("output", "No pude procesar la respuesta.")

        respuesta_placeholder.markdown(f"{respuesta_texto}{respuesta_completa}")
        self.memory.save_context({"input": pregunta}, {"output": respuesta_completa})

    def generar_respuesta(self, pregunta: str):
        """Orquesta el flujo completo: delegar y ejecutar."""
        # --- Paso 1: Orquestación y Delegación (Concepto IL2.3) ---
        agente_seleccionado = self._delegar_tarea(pregunta)
        # --- Paso 2: Ejecución del agente especializado ---
        self.ejecutar_agente_especializado(agente_seleccionado, pregunta)

def main():
    st.set_page_config(page_title="Sistema Multi-Agente del Curso", page_icon="🎭", layout="wide")
    st.title("🎭 Sistema Multi-Agente para el Curso (RA1 & RA2)")
    st.markdown("Un **orquestador** delega tu pregunta al agente especializado correcto (Contenido o Planificación), aplicando los conceptos de `IL2.3`.")

    if 'orquestador' not in st.session_state:
        st.session_state.orquestador = OrquestadorCurso()

    with st.sidebar:
        st.header("Configuración del Agente")
        if st.button("🧠 Cargar Conocimiento del Curso (RA1 & RA2)"):
            with st.spinner("Indexando documentos..."):
                st.session_state.orquestador.inicializar_rag_con_contenido_curso()
        
        if st.button("🗑️ Limpiar Memoria"):
            st.session_state.orquestador.memory.clear()
            st.success("Memoria de la conversación limpiada.")

        st.markdown("---")
        if st.session_state.orquestador.vectorstore:
            st.success(f"✅ Base de conocimientos cargada con {st.session_state.orquestador.vectorstore.index.ntotal} chunks.")
        else:
            st.warning("⚠️ La base de conocimientos está vacía. Carga el material del curso.")
        
        st.markdown("---")
        st.subheader("Equipo de Agentes")
        st.markdown(f"**1. {st.session_state.orquestador.agente_contenido.name}**")
        st.caption(st.session_state.orquestador.agente_contenido.role)
        st.markdown(f"**2. {st.session_state.orquestador.agente_planificador.name}**")
        st.caption(st.session_state.orquestador.agente_planificador.role)

    # Mostrar historial de chat
    historial = st.session_state.orquestador.memory.load_memory_variables({})["chat_history"]
    for msg in historial:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg.content)

    # Input del usuario
    pregunta = st.chat_input("Pregunta sobre RA1 o RA2...")
    if pregunta:
        with st.chat_message("user", avatar="🧑"):
            st.write(pregunta)
        with st.chat_message("assistant", avatar="🤖"):
            st.session_state.orquestador.generar_respuesta(pregunta)

if __name__ == "__main__":
    if not os.getenv("GITHUB_TOKEN"):
        st.error("⚠️ Por favor, configura la variable de entorno GITHUB_TOKEN.")
    else:
        main()