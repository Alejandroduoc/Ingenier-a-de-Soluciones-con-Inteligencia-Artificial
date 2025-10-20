"""
IL2.3: Planificación Jerárquica con LangChain
============================================

Este módulo demuestra cómo implementar planificación jerárquica, donde un objetivo 
complejo se descompone en sub-objetivos manejables organizados en niveles de abstracción.

Conceptos Clave:
- Descomposición de objetivos complejos en sub-tareas
- Niveles de abstracción (Alto, Medio, Bajo)
- Gestión de dependencias entre sub-objetivos
- Ejecución ordenada según prioridades

Para Estudiantes:
La planificación jerárquica es útil cuando tienes tareas complejas que necesitan 
dividirse en pasos más pequeños y manejables. Por ejemplo, "Desarrollar una aplicación"
se divide en: diseñar, implementar, probar, desplegar.
"""

# Requiere: pip install langchain langchain-openai openai python-dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
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

print("✅ LLM configurado correctamente")


class HierarchicalPlanner:
    """
    Planificador Jerárquico que descompone objetivos en múltiples niveles
    
    Atributos:
        llm: Modelo de lenguaje para generar planes
        levels: Niveles de abstracción (alto, medio, bajo)
        plan_hierarchy: Estructura del plan jerárquico
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.levels = ["Alto", "Medio", "Bajo"]
        self.plan_hierarchy = {}
    
    def decompose_goal(self, goal: str) -> Dict[str, List[str]]:
        """
        Descompone un objetivo en sub-objetivos jerárquicos
        
        Args:
            goal: Objetivo principal a descomponer
            
        Returns:
            Diccionario con niveles y sus sub-objetivos
        """
        print(f"\n🎯 Descomponiendo objetivo: '{goal}'")
        print("=" * 60)
        
        # Prompt para descomposición jerárquica
        decomposition_prompt = f"""
        Descompón el siguiente objetivo en 3 niveles jerárquicos:
        
        Objetivo Principal: {goal}
        
        Proporciona la descomposición en formato JSON con esta estructura:
        {{
            "Alto": ["sub-objetivo 1", "sub-objetivo 2", ...],
            "Medio": ["tarea 1", "tarea 2", ...],
            "Bajo": ["acción 1", "acción 2", ...]
        }}
        
        - Nivel Alto: Grandes fases o etapas del proyecto
        - Nivel Medio: Tareas específicas dentro de cada fase
        - Nivel Bajo: Acciones concretas y detalladas
        
        Responde SOLO con el JSON, sin texto adicional.
        """
        
        try:
            response = self.llm.invoke(decomposition_prompt)
            hierarchy = json.loads(response.content)
            
            # Mostrar la jerarquía
            for level in self.levels:
                print(f"\n📊 Nivel {level}:")
                for i, item in enumerate(hierarchy.get(level, []), 1):
                    print(f"   {i}. {item}")
            
            self.plan_hierarchy = hierarchy
            return hierarchy
            
        except Exception as e:
            print(f"⚠️ Error en descomposición: {e}")
            # Plan por defecto
            return {
                "Alto": ["Fase 1", "Fase 2", "Fase 3"],
                "Medio": ["Tarea 1", "Tarea 2", "Tarea 3"],
                "Bajo": ["Acción 1", "Acción 2", "Acción 3"]
            }
    
    def create_execution_plan(self, hierarchy: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Crea un plan de ejecución ordenado desde el nivel más bajo al más alto
        
        Args:
            hierarchy: Jerarquía de objetivos
            
        Returns:
            Lista de tareas ordenadas para ejecución
        """
        print("\n\n📋 Generando Plan de Ejecución...")
        print("=" * 60)
        
        execution_plan = []
        task_id = 1
        
        # Ejecutar desde el nivel más bajo (acciones concretas) hacia arriba
        for level in reversed(self.levels):
            items = hierarchy.get(level, [])
            priority = self.levels.index(level) + 1
            
            for item in items:
                task = {
                    "id": task_id,
                    "level": level,
                    "description": item,
                    "priority": priority,
                    "status": "pending"
                }
                execution_plan.append(task)
                task_id += 1
        
        # Mostrar plan de ejecución
        print("\n🔢 Orden de Ejecución (del detalle a lo general):")
        for task in execution_plan:
            print(f"   [{task['id']}] {task['level']}: {task['description']} (Prioridad: {task['priority']})")
        
        return execution_plan
    
    def execute_plan(self, execution_plan: List[Dict[str, Any]]) -> None:
        """
        Simula la ejecución del plan jerárquico
        
        Args:
            execution_plan: Plan de ejecución a ejecutar
        """
        print("\n\n🚀 Ejecutando Plan Jerárquico...")
        print("=" * 60)
        
        for task in execution_plan:
            print(f"\n▶️  Ejecutando Tarea #{task['id']}")
            print(f"    Nivel: {task['level']}")
            print(f"    Descripción: {task['description']}")
            print(f"    Prioridad: {task['priority']}")
            
            # Simular ejecución
            task['status'] = "completed"
            print(f"    ✅ Estado: Completada")
        
        print("\n\n🎉 Plan Jerárquico Completado Exitosamente!")
        
        # Resumen
        print("\n📊 Resumen de Ejecución:")
        for level in reversed(self.levels):
            count = sum(1 for t in execution_plan if t['level'] == level)
            print(f"   - Nivel {level}: {count} tareas completadas")


def demo_hierarchical_planning():
    """
    Demostración educativa de planificación jerárquica
    """
    print("\n" + "="*70)
    print("  🎓 DEMOSTRACIÓN: PLANIFICACIÓN JERÁRQUICA CON LANGCHAIN")
    print("="*70)
    
    # Crear planificador
    planner = HierarchicalPlanner(llm)
    
    # Ejemplo 1: Desarrollo de Software
    print("\n\n📚 EJEMPLO 1: Desarrollo de una Aplicación Web")
    print("-" * 70)
    
    goal1 = "Desarrollar una aplicación web de gestión de tareas"
    hierarchy1 = planner.decompose_goal(goal1)
    plan1 = planner.create_execution_plan(hierarchy1)
    planner.execute_plan(plan1)
    
    # Ejemplo 2: Investigación Académica
    print("\n\n" + "="*70)
    print("\n📚 EJEMPLO 2: Investigación sobre Inteligencia Artificial")
    print("-" * 70)
    
    goal2 = "Realizar una investigación académica sobre IA en educación"
    hierarchy2 = planner.decompose_goal(goal2)
    plan2 = planner.create_execution_plan(hierarchy2)
    planner.execute_plan(plan2)
    
    # Lecciones aprendidas
    print("\n\n" + "="*70)
    print("  💡 LECCIONES CLAVE PARA ESTUDIANTES")
    print("="*70)
    print("""
    1. La planificación jerárquica ayuda a manejar la complejidad
    2. Dividir objetivos grandes en pasos pequeños facilita la ejecución
    3. Los niveles de abstracción permiten ver el proyecto desde diferentes perspectivas
    4. La priorización asegura que se completen primero las tareas fundamentales
    5. Un plan bien estructurado mejora la organización y el seguimiento
    """)


def demo_with_agent():
    """
    Demostración usando un agente LangChain con herramienta de planificación
    """
    print("\n\n" + "="*70)
    print("  🤖 DEMOSTRACIÓN: AGENTE CON PLANIFICACIÓN JERÁRQUICA")
    print("="*70)
    
    # Herramienta de planificación
    def plan_hierarchically(objetivo: str) -> str:
        """Crea un plan jerárquico para un objetivo dado"""
        planner = HierarchicalPlanner(llm)
        hierarchy = planner.decompose_goal(objetivo)
        
        # Formatear resultado
        result = f"Plan Jerárquico para: {objetivo}\n\n"
        for level in ["Alto", "Medio", "Bajo"]:
            result += f"Nivel {level}:\n"
            for item in hierarchy.get(level, []):
                result += f"  - {item}\n"
            result += "\n"
        return result
    
    # Crear herramienta
    planning_tool = Tool(
        name="PlanificadorJerarquico",
        func=plan_hierarchically,
        description="Crea un plan jerárquico descomponiendo un objetivo complejo en sub-objetivos organizados por niveles de abstracción."
    )
    
    # Crear agente
    agent = initialize_agent(
        tools=[planning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Ejecutar agente
    print("\n🤖 Agente trabajando en planificación...\n")
    resultado = agent.run("Crea un plan jerárquico para organizar un evento académico de IA")
    print(f"\n📋 Resultado del Agente:\n{resultado}")


if __name__ == "__main__":
    # Ejecutar demostraciones
    demo_hierarchical_planning()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver la demostración con agente...")
    demo_with_agent()
    
    print("\n\n✅ Demostración completada. ¡Experimenta con tus propios objetivos!")

