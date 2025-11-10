import os
import glob
import json
import numpy as np
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

import streamlit as st
import faiss
from openai import OpenAI

# --- 1. Clases de Datos y Componentes Fundamentales (Sin LangChain) ---

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Action:
    name: str
    preconditions: Set[str]
    add_effects: Set[str]
    delete_effects: Set[str] = field(default_factory=set)
    cost: float = 1.0

    def applicable(self, state: Set[str]) -> bool:
        return self.preconditions.issubset(state)

    def apply(self, state: Set[str]) -> Set[str]:
        new_state = state.copy()
        new_state -= self.delete_effects
        new_state |= self.add_effects
        return new_state

@dataclass
class SubTask:
    id: str
    title: str
    description: str
    goal: str # Objetivo específico para el planificador

class OrquestadorCurso:
    """
    Orquesta la respuesta a preguntas complejas usando Descomposición de Tareas y Planificación.
    Aplica conceptos de RAG (RA1), Descomposición y Planificación (IL2.3) sin usar LangChain.
    """
    def __init__(self):
        # --- Configuración Global (Cliente OpenAI) ---
        self.client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
        )
        self.model = "openai/gpt-4o-mini"
        
        # --- Memoria de Conversación (Simple) ---
        self.memory = []
        
        # --- Componentes RAG (Sin LangChain) ---
        self.vectorstore = None
        self.documents = [] # Almacena los documentos originales (chunks)
        self.chunk_size = 1500
        self.chunk_overlap = 250
        
        # --- Componentes de Planificación (IL2.3) ---
        self.actions = self._definir_acciones()
        
        # --- Ruta raíz del proyecto ---
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.normpath(os.path.join(base_path, '..', '..'))

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Llamada directa al API de OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtiene embeddings directamente del API."""
        response = self.client.embeddings.create(
            input=texts,
            model="openai/text-embedding-ada-002" # Modelo de embedding compatible
        )
        return [item.embedding for item in response.data]

    def _split_text(self, text: str) -> List[str]:
        """Implementación simple de RecursiveCharacterTextSplitter."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def inicializar_rag_con_contenido_curso(self):
        """Carga, divide e indexa el contenido de RA1 y RA2."""
        st.info("📚 Cargando material de estudio de RA1 y RA2...")
        
        # --- Carga de Documentos ---
        loaded_docs = []
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