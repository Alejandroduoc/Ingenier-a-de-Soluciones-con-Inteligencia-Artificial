"""
IL2.3: Descomposición Inteligente de Tareas
==========================================

Este módulo implementa descomposición automática de tareas complejas en
sub-tareas más manejables usando LLMs.

Conceptos Clave:
- Análisis de complejidad de tareas
- Descomposición recursiva
- Identificación de sub-tareas
- Estimación de esfuerzo
- Priorización inteligente

Para Estudiantes:
La descomposición de tareas es esencial en gestión de proyectos y desarrollo de
software. Ayuda a transformar objetivos grandes y abrumadores en pasos accionables.
Los LLMs pueden asistir en este proceso con su comprensión del lenguaje natural.
"""

# Requiere: pip install langchain langchain-openai openai python-dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
from dataclasses import dataclass
import os
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv no está instalado. Instálalo con: pip install python-dotenv")
    exit(1)

# Obtener variables de entorno
github_token = os.getenv("GITHUB_TOKEN")
github_base_url = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")

if not github_token:
    print("❌ GITHUB_TOKEN no está configurado. Por favor verifica tu archivo .env")
    print("💡 Tu archivo .env debe contener: GITHUB_TOKEN=tu_token_aqui")
    exit(1)

# Configurar LLM
llm = ChatOpenAI(
    model="gpt-4o",
    base_url=github_base_url,
    api_key=github_token,
    temperature=0.7
)

print("✅ LLM configurado correctamente\n")


@dataclass
class SubTask:
    """
    Representación de una sub-tarea
    
    Atributos:
        id: Identificador
        title: Título de la sub-tarea
        description: Descripción detallada
        estimated_hours: Horas estimadas
        priority: Prioridad (alta, media, baja)
        dependencies: IDs de sub-tareas de las que depende
        skills_required: Habilidades necesarias
    """
    id: str
    title: str
    description: str
    estimated_hours: float
    priority: str
    dependencies: List[str] = None
    skills_required: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.skills_required is None:
            self.skills_required = []


class TaskDecomposer:
    """
    Descomponedor inteligente de tareas usando LLM
    
    Atributos:
        llm: Modelo de lenguaje
        max_depth: Profundidad máxima de descomposición recursiva
    """
    
    def __init__(self, llm, max_depth: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.decomposition_history = []
    
    def analyze_complexity(self, task: str) -> Dict[str, Any]:
        """
        Analiza la complejidad de una tarea
        
        Args:
            task: Descripción de la tarea
            
        Returns:
            Diccionario con análisis de complejidad
        """
        print(f"\n🔍 Analizando complejidad de tarea...")
        
        prompt = f"""
        Analiza la complejidad de la siguiente tarea y proporciona tu análisis en formato JSON:
        
        Tarea: {task}
        
        Proporciona el análisis con esta estructura:
        {{
            "complexity_level": "baja/media/alta/muy_alta",
            "estimated_hours": número,
            "requires_decomposition": true/false,
            "main_challenges": ["desafío 1", "desafío 2", ...],
            "required_skills": ["habilidad 1", "habilidad 2", ...],
            "suggested_approach": "descripción del enfoque recomendado"
        }}
        
        Responde SOLO con el JSON, sin texto adicional.
        """
        
        try:
            response = self.llm.invoke(prompt)
            analysis = json.loads(response.content)
            
            print(f"\n📊 Análisis de Complejidad:")
            print(f"   Nivel: {analysis['complexity_level']}")
            print(f"   Estimación: {analysis['estimated_hours']} horas")
            print(f"   ¿Requiere descomposición?: {analysis['requires_decomposition']}")
            print(f"   Desafíos principales: {', '.join(analysis['main_challenges'])}")
            print(f"   Habilidades requeridas: {', '.join(analysis['required_skills'])}")
            
            return analysis
            
        except Exception as e:
            print(f"⚠️ Error en análisis: {e}")
            return {
                "complexity_level": "media",
                "estimated_hours": 8,
                "requires_decomposition": True,
                "main_challenges": ["Análisis no disponible"],
                "required_skills": ["General"],
                "suggested_approach": "Enfoque estándar"
            }
    
    def decompose(self, task: str, depth: int = 0) -> List[SubTask]:
        """
        Descompone una tarea en sub-tareas
        
        Args:
            task: Tarea a descomponer
            depth: Profundidad actual de recursión
            
        Returns:
            Lista de sub-tareas
        """
        if depth >= self.max_depth:
            print(f"⚠️ Profundidad máxima alcanzada ({self.max_depth})")
            return []
        
        print(f"\n{'  ' * depth}🔨 Descomponiendo tarea (nivel {depth}):")
        print(f"{'  ' * depth}   '{task[:80]}...'")
        
        prompt = f"""
        Descompón la siguiente tarea en sub-tareas manejables y específicas.
        
        Tarea principal: {task}
        
        Proporciona la descomposición en formato JSON con esta estructura:
        {{
            "subtasks": [
                {{
                    "id": "subtask_1",
                    "title": "Título corto",
                    "description": "Descripción detallada",
                    "estimated_hours": número,
                    "priority": "alta/media/baja",
                    "dependencies": ["id_de_subtarea_previa"],
                    "skills_required": ["habilidad necesaria"]
                }}
            ]
        }}
        
        Directrices:
        - Crea entre 3-7 sub-tareas
        - Cada sub-tarea debe ser específica y accionable
        - Estima horas realísticamente
        - Define dependencias lógicas
        - Indica habilidades necesarias
        
        Responde SOLO con el JSON, sin texto adicional.
        """
        
        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response.content)
            
            subtasks = []
            for st_data in data.get("subtasks", []):
                subtask = SubTask(
                    id=st_data.get("id", f"task_{len(subtasks)}"),
                    title=st_data.get("title", "Sin título"),
                    description=st_data.get("description", ""),
                    estimated_hours=float(st_data.get("estimated_hours", 1)),
                    priority=st_data.get("priority", "media"),
                    dependencies=st_data.get("dependencies", []),
                    skills_required=st_data.get("skills_required", [])
                )
                subtasks.append(subtask)
                
                print(f"{'  ' * depth}   ✅ {subtask.title} ({subtask.estimated_hours}h)")
            
            # Guardar en historial
            self.decomposition_history.append({
                "task": task,
                "depth": depth,
                "subtasks_count": len(subtasks)
            })
            
            return subtasks
            
        except Exception as e:
            print(f"{'  ' * depth}   ❌ Error en descomposición: {e}")
            return []
    
    def recursive_decompose(self, task: str, complexity_threshold: float = 4.0) -> Dict[str, Any]:
        """
        Descomposición recursiva basada en complejidad
        
        Args:
            task: Tarea principal
            complexity_threshold: Umbral de horas para seguir descomponiendo
            
        Returns:
            Árbol de descomposición completo
        """
        print(f"\n{'='*70}")
        print(f"🎯 DESCOMPOSICIÓN RECURSIVA")
        print(f"{'='*70}")
        
        # Analizar complejidad inicial
        analysis = self.analyze_complexity(task)
        
        # Descomponer
        subtasks = self.decompose(task, depth=0)
        
        # Descomponer recursivamente sub-tareas complejas
        detailed_subtasks = []
        for subtask in subtasks:
            detailed_subtasks.append({
                "task": subtask,
                "children": []
            })
            
            # Si la sub-tarea es compleja, descomponerla más
            if subtask.estimated_hours > complexity_threshold:
                print(f"\n  ↳ Sub-tarea compleja detectada: {subtask.title}")
                children = self.decompose(subtask.description, depth=1)
                detailed_subtasks[-1]["children"] = children
        
        return {
            "main_task": task,
            "analysis": analysis,
            "subtasks": detailed_subtasks,
            "total_subtasks": len(subtasks),
            "total_estimated_hours": sum(st.estimated_hours for st in subtasks)
        }
    
    def generate_gantt_data(self, decomposition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera datos para diagrama de Gantt
        
        Args:
            decomposition: Resultado de descomposición
            
        Returns:
            Lista de tareas con fechas
        """
        gantt_tasks = []
        current_day = 0
        
        for item in decomposition["subtasks"]:
            task = item["task"]
            duration = task.estimated_hours / 8  # Convertir a días
            
            gantt_tasks.append({
                "id": task.id,
                "name": task.title,
                "start_day": current_day,
                "duration_days": duration,
                "end_day": current_day + duration,
                "priority": task.priority
            })
            
            # Siguiente tarea comienza después (simplificado)
            current_day += duration
        
        return gantt_tasks
    
    def print_summary(self, decomposition: Dict[str, Any]):
        """Imprime resumen de la descomposición"""
        print(f"\n\n{'='*70}")
        print(f"📋 RESUMEN DE DESCOMPOSICIÓN")
        print(f"{'='*70}")
        
        print(f"\n🎯 Tarea Principal:")
        print(f"   {decomposition['main_task']}")
        
        print(f"\n📊 Estadísticas:")
        print(f"   Total de sub-tareas: {decomposition['total_subtasks']}")
        print(f"   Horas estimadas: {decomposition['total_estimated_hours']:.1f}h")
        print(f"   Días estimados: {decomposition['total_estimated_hours']/8:.1f} días")
        
        print(f"\n📝 Sub-tareas Detalladas:")
        for i, item in enumerate(decomposition['subtasks'], 1):
            task = item['task']
            print(f"\n{i}. {task.title}")
            print(f"   📖 {task.description}")
            print(f"   ⏱️  Estimación: {task.estimated_hours}h")
            print(f"   🎯 Prioridad: {task.priority}")
            print(f"   🔧 Habilidades: {', '.join(task.skills_required)}")
            
            if task.dependencies:
                print(f"   🔗 Dependencias: {', '.join(task.dependencies)}")
            
            if item['children']:
                print(f"   📦 Sub-sub-tareas: {len(item['children'])}")
                for child in item['children']:
                    print(f"      • {child.title} ({child.estimated_hours}h)")


def demo_software_project():
    """
    Demostración: Proyecto de Software
    """
    print("="*70)
    print("  🎓 DEMOSTRACIÓN: DESCOMPOSICIÓN DE PROYECTO DE SOFTWARE")
    print("="*70)
    
    decomposer = TaskDecomposer(llm, max_depth=2)
    
    task = """
    Desarrollar una aplicación web completa de comercio electrónico que incluya:
    - Sistema de autenticación de usuarios
    - Catálogo de productos con búsqueda y filtros
    - Carrito de compras
    - Procesamiento de pagos
    - Panel de administración
    - Sistema de notificaciones
    """
    
    result = decomposer.recursive_decompose(task, complexity_threshold=6.0)
    decomposer.print_summary(result)
    
    # Generar datos de Gantt
    print(f"\n\n📅 CRONOGRAMA ESTIMADO (Diagrama de Gantt):")
    print("="*70)
    gantt_data = decomposer.generate_gantt_data(result)
    
    for task in gantt_data:
        bar_length = int(task['duration_days'] * 2)
        bar = "█" * bar_length
        print(f"{task['name'][:30]:30} |{bar}| {task['duration_days']:.1f} días")


def demo_research_paper():
    """
    Demostración: Paper de Investigación
    """
    print("\n\n" + "="*70)
    print("  📚 DEMOSTRACIÓN: DESCOMPOSICIÓN DE PAPER DE INVESTIGACIÓN")
    print("="*70)
    
    decomposer = TaskDecomposer(llm, max_depth=2)
    
    task = """
    Escribir un paper académico sobre el impacto de la Inteligencia Artificial
    en la educación superior, incluyendo revisión de literatura, metodología,
    análisis de casos de estudio, resultados y conclusiones.
    """
    
    result = decomposer.recursive_decompose(task, complexity_threshold=5.0)
    decomposer.print_summary(result)


def demo_event_planning():
    """
    Demostración: Organización de Evento
    """
    print("\n\n" + "="*70)
    print("  🎉 DEMOSTRACIÓN: DESCOMPOSICIÓN DE EVENTO ACADÉMICO")
    print("="*70)
    
    decomposer = TaskDecomposer(llm, max_depth=2)
    
    task = """
    Organizar una conferencia académica de inteligencia artificial con 200 asistentes,
    incluyendo gestión de ponentes, logística de venue, catering, marketing,
    inscripciones, y material promocional.
    """
    
    result = decomposer.recursive_decompose(task, complexity_threshold=4.0)
    decomposer.print_summary(result)
    
    # Análisis de prioridades
    print(f"\n\n🎯 ANÁLISIS DE PRIORIDADES:")
    print("="*70)
    
    high_priority = [st["task"] for st in result["subtasks"] if st["task"].priority == "alta"]
    medium_priority = [st["task"] for st in result["subtasks"] if st["task"].priority == "media"]
    low_priority = [st["task"] for st in result["subtasks"] if st["task"].priority == "baja"]
    
    print(f"\n🔴 Alta Prioridad ({len(high_priority)} tareas):")
    for task in high_priority:
        print(f"   • {task.title} ({task.estimated_hours}h)")
    
    print(f"\n🟡 Media Prioridad ({len(medium_priority)} tareas):")
    for task in medium_priority:
        print(f"   • {task.title} ({task.estimated_hours}h)")
    
    print(f"\n🟢 Baja Prioridad ({len(low_priority)} tareas):")
    for task in low_priority:
        print(f"   • {task.title} ({task.estimated_hours}h)")


if __name__ == "__main__":
    # Ejecutar demostraciones
    demo_software_project()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver descomposición de Paper de Investigación...")
    demo_research_paper()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver descomposición de Evento Académico...")
    demo_event_planning()
    
    # Lecciones finales
    print("\n\n" + "="*70)
    print("  💡 LECCIONES CLAVE PARA ESTUDIANTES")
    print("="*70)
    print("""
    1. La descomposición convierte tareas grandes en pasos accionables
    2. Los LLMs pueden asistir en identificar sub-tareas lógicas
    3. La estimación de esfuerzo ayuda en la planificación realista
    4. Las dependencias entre tareas determinan el orden de ejecución
    5. La priorización asegura que lo importante se haga primero
    
    💭 Reflexión: ¿Cómo podrías usar esto para organizar tu tesis o proyecto final?
    """)

