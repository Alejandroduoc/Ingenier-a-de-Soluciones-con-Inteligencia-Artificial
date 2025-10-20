"""
IL2.3: Planificación Orientada a Objetivos (STRIPS)
=================================================

Este módulo implementa planificación orientada a objetivos usando un enfoque
similar a STRIPS (Stanford Research Institute Problem Solver).

Conceptos Clave:
- Estados del mundo representados como conjuntos de hechos
- Acciones con precondiciones y efectos
- Búsqueda hacia atrás (backward chaining) desde el objetivo
- Secuencia de acciones que transforman estado inicial → estado objetivo

Para Estudiantes:
La planificación orientada a objetivos es útil cuando conoces exactamente qué 
quieres lograr y necesitas encontrar la secuencia óptima de acciones para llegar
ahí. Por ejemplo, navegación en un mapa, resolución de puzzles, o automatización
de tareas con pasos bien definidos.
"""

# Requiere: pip install langchain langchain-openai openai python-dotenv
from langchain_openai import ChatOpenAI
from typing import Set, List, Dict, Any, Tuple
from dataclasses import dataclass
import os

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
    temperature=0.3
)

print("✅ LLM configurado correctamente\n")


@dataclass
class Action:
    """
    Acción con precondiciones y efectos
    
    Atributos:
        name: Nombre de la acción
        preconditions: Conjunto de hechos que deben ser verdaderos antes
        add_effects: Hechos que se agregan al estado
        delete_effects: Hechos que se eliminan del estado
        cost: Costo de ejecutar la acción
    """
    name: str
    preconditions: Set[str]
    add_effects: Set[str]
    delete_effects: Set[str]
    cost: float = 1.0
    
    def applicable(self, state: Set[str]) -> bool:
        """Verifica si la acción es aplicable en el estado dado"""
        return self.preconditions.issubset(state)
    
    def apply(self, state: Set[str]) -> Set[str]:
        """Aplica la acción al estado y retorna el nuevo estado"""
        new_state = state.copy()
        new_state -= self.delete_effects
        new_state |= self.add_effects
        return new_state


class GoalOrientedPlanner:
    """
    Planificador orientado a objetivos usando búsqueda hacia atrás
    
    Atributos:
        actions: Lista de acciones disponibles
        initial_state: Estado inicial del mundo
        goal_state: Estado objetivo a alcanzar
    """
    
    def __init__(self):
        self.actions: List[Action] = []
        self.initial_state: Set[str] = set()
        self.goal_state: Set[str] = set()
    
    def add_action(self, action: Action):
        """Añade una acción al conjunto de acciones disponibles"""
        self.actions.append(action)
        print(f"   ✅ Acción añadida: {action.name}")
    
    def set_initial_state(self, state: Set[str]):
        """Establece el estado inicial"""
        self.initial_state = state
        print(f"\n📍 Estado inicial: {sorted(state)}")
    
    def set_goal_state(self, goal: Set[str]):
        """Establece el estado objetivo"""
        self.goal_state = goal
        print(f"🎯 Estado objetivo: {sorted(goal)}")
    
    def backward_search(self) -> List[Action]:
        """
        Búsqueda hacia atrás desde el objetivo al estado inicial
        
        Returns:
            Lista de acciones que llevan del estado inicial al objetivo
        """
        print(f"\n\n🔍 Iniciando búsqueda hacia atrás...")
        print("=" * 70)
        
        # Lista de sub-objetivos a alcanzar (empezando por el objetivo final)
        goals = [self.goal_state.copy()]
        plan = []
        
        iteration = 0
        while goals:
            iteration += 1
            current_goal = goals[-1]
            
            print(f"\n--- Iteración {iteration} ---")
            print(f"Sub-objetivo actual: {sorted(current_goal)}")
            
            # Si el objetivo actual ya está satisfecho, pasar al siguiente
            if current_goal.issubset(self.initial_state):
                print(f"✅ Sub-objetivo ya satisfecho")
                goals.pop()
                continue
            
            # Buscar una acción que ayude a lograr el objetivo
            action_found = False
            for action in self.actions:
                # ¿Esta acción contribuye al objetivo?
                if action.add_effects & current_goal:
                    print(f"   🎯 Acción útil encontrada: {action.name}")
                    print(f"      Efectos: +{sorted(action.add_effects)} -{sorted(action.delete_effects)}")
                    
                    # Añadir acción al plan
                    plan.insert(0, action)
                    
                    # Calcular nuevo sub-objetivo
                    new_goal = current_goal.copy()
                    new_goal -= action.add_effects  # Ya no necesitamos lo que esta acción provee
                    new_goal |= action.preconditions  # Ahora necesitamos las precondiciones
                    
                    # Actualizar objetivo actual
                    goals[-1] = new_goal
                    
                    action_found = True
                    break
            
            if not action_found:
                print(f"   ❌ No se encontró acción aplicable")
                return []
            
            # Límite de seguridad
            if iteration > 50:
                print("⚠️ Límite de iteraciones alcanzado")
                break
        
        return plan
    
    def forward_search(self) -> List[Action]:
        """
        Búsqueda hacia adelante desde el estado inicial
        
        Returns:
            Lista de acciones que llevan del estado inicial al objetivo
        """
        print(f"\n\n🔍 Iniciando búsqueda hacia adelante...")
        print("=" * 70)
        
        current_state = self.initial_state.copy()
        plan = []
        visited_states = set()
        
        iteration = 0
        while not self.goal_state.issubset(current_state):
            iteration += 1
            print(f"\n--- Iteración {iteration} ---")
            print(f"Estado actual: {sorted(current_state)}")
            print(f"Falta lograr: {sorted(self.goal_state - current_state)}")
            
            # Convertir estado a tupla para poder agregarlo a visited_states
            state_tuple = tuple(sorted(current_state))
            if state_tuple in visited_states:
                print("⚠️ Estado ya visitado, evitando ciclo")
                break
            visited_states.add(state_tuple)
            
            # Buscar acción aplicable que nos acerque al objetivo
            best_action = None
            best_score = -1
            
            for action in self.actions:
                if action.applicable(current_state):
                    # Calcular cuántos hechos del objetivo esta acción logra
                    new_state = action.apply(current_state)
                    score = len(self.goal_state & new_state) - len(self.goal_state & current_state)
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            if best_action is None:
                print("❌ No hay acciones aplicables")
                return []
            
            print(f"   ➡️  Aplicando: {best_action.name}")
            current_state = best_action.apply(current_state)
            plan.append(best_action)
            
            # Límite de seguridad
            if iteration > 50:
                print("⚠️ Límite de iteraciones alcanzado")
                break
        
        return plan
    
    def execute_plan(self, plan: List[Action]):
        """
        Ejecuta el plan y muestra los resultados
        
        Args:
            plan: Lista de acciones a ejecutar
        """
        if not plan:
            print("\n❌ No se pudo generar un plan válido")
            return
        
        print(f"\n\n🚀 Ejecutando Plan ({len(plan)} pasos)")
        print("=" * 70)
        
        current_state = self.initial_state.copy()
        total_cost = 0
        
        for i, action in enumerate(plan, 1):
            print(f"\nPaso {i}: {action.name}")
            print(f"   Precondiciones: {sorted(action.preconditions)}")
            print(f"   Estado antes: {sorted(current_state)}")
            
            if not action.applicable(current_state):
                print(f"   ❌ ERROR: Acción no aplicable")
                return
            
            current_state = action.apply(current_state)
            total_cost += action.cost
            
            print(f"   Efectos: +{sorted(action.add_effects)} -{sorted(action.delete_effects)}")
            print(f"   Estado después: {sorted(current_state)}")
            print(f"   ✅ Completado (Costo: {action.cost})")
        
        # Verificar si se alcanzó el objetivo
        print(f"\n\n📊 Resultado Final:")
        print("=" * 70)
        print(f"Estado final: {sorted(current_state)}")
        print(f"Estado objetivo: {sorted(self.goal_state)}")
        print(f"Costo total: {total_cost}")
        
        if self.goal_state.issubset(current_state):
            print("\n✅ ¡OBJETIVO ALCANZADO!")
        else:
            print("\n❌ Objetivo no alcanzado")
            print(f"Faltó: {sorted(self.goal_state - current_state)}")


def demo_blocks_world():
    """
    Demostración clásica: Mundo de Bloques
    """
    print("="*70)
    print("  🎓 DEMOSTRACIÓN: MUNDO DE BLOQUES")
    print("="*70)
    print("""
    Configuración inicial:
        [A]
        [B]
        [C]
       -----
       Mesa
    
    Objetivo:
        [C]
        [B]
        [A]
       -----
       Mesa
    """)
    
    planner = GoalOrientedPlanner()
    
    # Definir acciones
    print("\n📋 Definiendo acciones disponibles:")
    
    planner.add_action(Action(
        name="Mover A sobre B",
        preconditions={"A_libre", "B_libre", "A_en_mesa"},
        add_effects={"A_sobre_B"},
        delete_effects={"A_libre", "B_libre", "A_en_mesa"}
    ))
    
    planner.add_action(Action(
        name="Mover A a mesa",
        preconditions={"A_libre"},
        add_effects={"A_en_mesa", "A_libre"},
        delete_effects=set()
    ))
    
    planner.add_action(Action(
        name="Mover B sobre A",
        preconditions={"B_libre", "A_libre", "B_en_mesa"},
        add_effects={"B_sobre_A"},
        delete_effects={"B_libre", "A_libre", "B_en_mesa"}
    ))
    
    planner.add_action(Action(
        name="Mover C sobre B",
        preconditions={"C_libre", "B_libre"},
        add_effects={"C_sobre_B"},
        delete_effects={"C_libre", "B_libre"}
    ))
    
    # Estado inicial: A sobre B sobre C
    planner.set_initial_state({
        "A_libre", "A_sobre_B",
        "B_sobre_C",
        "C_en_mesa"
    })
    
    # Objetivo: C sobre B sobre A
    planner.set_goal_state({
        "C_sobre_B",
        "B_sobre_A",
        "A_en_mesa"
    })
    
    # Planificar y ejecutar
    plan = planner.forward_search()
    planner.execute_plan(plan)


def demo_robot_navigation():
    """
    Demostración: Navegación de Robot
    """
    print("\n\n" + "="*70)
    print("  🤖 DEMOSTRACIÓN: NAVEGACIÓN DE ROBOT")
    print("="*70)
    print("""
    El robot debe navegar de la Habitación A a la Habitación D,
    pasando por puertas que pueden estar cerradas.
    """)
    
    planner = GoalOrientedPlanner()
    
    print("\n📋 Definiendo acciones de navegación:")
    
    # Acciones de movimiento
    planner.add_action(Action(
        name="Ir de A a B",
        preconditions={"robot_en_A", "puerta_AB_abierta"},
        add_effects={"robot_en_B"},
        delete_effects={"robot_en_A"},
        cost=1.0
    ))
    
    planner.add_action(Action(
        name="Ir de B a C",
        preconditions={"robot_en_B", "puerta_BC_abierta"},
        add_effects={"robot_en_C"},
        delete_effects={"robot_en_B"},
        cost=1.0
    ))
    
    planner.add_action(Action(
        name="Ir de C a D",
        preconditions={"robot_en_C", "puerta_CD_abierta"},
        add_effects={"robot_en_D"},
        delete_effects={"robot_en_C"},
        cost=1.0
    ))
    
    # Acciones de abrir puertas
    planner.add_action(Action(
        name="Abrir puerta AB",
        preconditions={"robot_en_A"},
        add_effects={"puerta_AB_abierta"},
        delete_effects=set(),
        cost=0.5
    ))
    
    planner.add_action(Action(
        name="Abrir puerta BC",
        preconditions={"robot_en_B"},
        add_effects={"puerta_BC_abierta"},
        delete_effects=set(),
        cost=0.5
    ))
    
    planner.add_action(Action(
        name="Abrir puerta CD",
        preconditions={"robot_en_C"},
        add_effects={"puerta_CD_abierta"},
        delete_effects=set(),
        cost=0.5
    ))
    
    # Estado inicial
    planner.set_initial_state({
        "robot_en_A"
        # Todas las puertas cerradas
    })
    
    # Objetivo
    planner.set_goal_state({
        "robot_en_D"
    })
    
    # Planificar y ejecutar
    plan = planner.forward_search()
    planner.execute_plan(plan)


def demo_task_automation():
    """
    Demostración: Automatización de Tareas
    """
    print("\n\n" + "="*70)
    print("  ⚙️ DEMOSTRACIÓN: AUTOMATIZACIÓN DE TAREAS")
    print("="*70)
    print("""
    Automatizar el proceso de preparar y enviar un reporte.
    """)
    
    planner = GoalOrientedPlanner()
    
    print("\n📋 Definiendo flujo de trabajo:")
    
    planner.add_action(Action(
        name="Recolectar datos",
        preconditions={"sistema_activo"},
        add_effects={"datos_recolectados"},
        delete_effects=set()
    ))
    
    planner.add_action(Action(
        name="Procesar datos",
        preconditions={"datos_recolectados", "software_disponible"},
        add_effects={"datos_procesados"},
        delete_effects=set()
    ))
    
    planner.add_action(Action(
        name="Generar reporte",
        preconditions={"datos_procesados", "plantilla_disponible"},
        add_effects={"reporte_generado"},
        delete_effects=set()
    ))
    
    planner.add_action(Action(
        name="Revisar reporte",
        preconditions={"reporte_generado"},
        add_effects={"reporte_revisado"},
        delete_effects=set()
    ))
    
    planner.add_action(Action(
        name="Enviar reporte",
        preconditions={"reporte_revisado", "destinatarios_confirmados"},
        add_effects={"reporte_enviado"},
        delete_effects=set()
    ))
    
    # Estado inicial
    planner.set_initial_state({
        "sistema_activo",
        "software_disponible",
        "plantilla_disponible",
        "destinatarios_confirmados"
    })
    
    # Objetivo
    planner.set_goal_state({
        "reporte_enviado"
    })
    
    # Planificar y ejecutar
    plan = planner.forward_search()
    planner.execute_plan(plan)


if __name__ == "__main__":
    # Ejecutar demostraciones
    demo_blocks_world()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver Navegación de Robot...")
    demo_robot_navigation()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver Automatización de Tareas...")
    demo_task_automation()
    
    # Lecciones finales
    print("\n\n" + "="*70)
    print("  💡 LECCIONES CLAVE PARA ESTUDIANTES")
    print("="*70)
    print("""
    1. La planificación orientada a objetivos encuentra secuencias de acciones óptimas
    2. Las precondiciones y efectos definen qué acciones son aplicables
    3. La búsqueda puede ser hacia adelante (estado inicial) o hacia atrás (objetivo)
    4. Es útil cuando el problema tiene una estructura clara de estados
    5. Funciona mejor en entornos deterministas con acciones bien definidas
    
    💭 Reflexión: ¿Qué problemas del mundo real podrían modelarse así?
    """)

