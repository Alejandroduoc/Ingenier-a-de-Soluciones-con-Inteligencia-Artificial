"""
IL2.3: Gestión de Flujos de Trabajo (Workflow Management)
========================================================

Este módulo implementa un sistema de gestión de flujos de trabajo con
soporte para dependencias, ejecución paralela y control de errores.

Conceptos Clave:
- Definición de tareas con dependencias
- Ejecución paralela de tareas independientes
- Grafo de dependencias (DAG - Directed Acyclic Graph)
- Manejo de errores y reintentos
- Monitoreo de progreso

Para Estudiantes:
Los flujos de trabajo permiten automatizar procesos complejos donde unas tareas
dependen de otras. Por ejemplo, en CI/CD, el deployment solo ocurre si las
pruebas pasan. Es fundamental para ETL, pipelines de ML, y automatización.
"""

# Requiere: pip install langchain langchain-openai openai python-dotenv
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import os
import time
from datetime import datetime

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
    temperature=0.5
)

print("✅ LLM configurado correctamente\n")


class TaskStatus(Enum):
    """Estados posibles de una tarea"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    Tarea en un flujo de trabajo
    
    Atributos:
        id: Identificador único
        name: Nombre de la tarea
        function: Función a ejecutar
        dependencies: IDs de tareas de las que depende
        status: Estado actual
        result: Resultado de la ejecución
        error: Error si falló
        start_time: Tiempo de inicio
        end_time: Tiempo de finalización
        retries: Número de reintentos permitidos
        retry_count: Número de reintentos ejecutados
    """
    id: str
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    start_time: float = None
    end_time: float = None
    retries: int = 0
    retry_count: int = 0
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Verifica si la tarea puede ejecutarse"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def execute(self, context: Dict[str, Any]) -> bool:
        """
        Ejecuta la tarea
        
        Args:
            context: Contexto con resultados de tareas previas
            
        Returns:
            True si tuvo éxito, False si falló
        """
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        
        print(f"\n⚡ Ejecutando: {self.name}")
        
        try:
            # Ejecutar función
            self.result = self.function(context)
            self.status = TaskStatus.COMPLETED
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            
            print(f"   ✅ Completada en {duration:.2f}s")
            return True
            
        except Exception as e:
            self.error = str(e)
            self.retry_count += 1
            
            if self.retry_count <= self.retries:
                print(f"   ⚠️ Error (reintento {self.retry_count}/{self.retries}): {e}")
                self.status = TaskStatus.PENDING
                return False
            else:
                self.status = TaskStatus.FAILED
                self.end_time = time.time()
                print(f"   ❌ Falló: {e}")
                return False


class WorkflowManager:
    """
    Gestor de flujos de trabajo con DAG
    
    Atributos:
        name: Nombre del workflow
        tasks: Diccionario de tareas
        execution_log: Log de ejecución
    """
    
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        
        print(f"\n🔧 Workflow '{name}' inicializado")
    
    def add_task(self, task: Task):
        """Añade una tarea al workflow"""
        self.tasks[task.id] = task
        print(f"   ✅ Tarea añadida: {task.name}")
    
    def validate_dag(self) -> bool:
        """
        Valida que el grafo de dependencias sea acíclico (DAG)
        
        Returns:
            True si es válido, False si hay ciclos
        """
        def has_cycle(task_id: str, visited: Set[str], stack: Set[str]) -> bool:
            visited.add(task_id)
            stack.add(task_id)
            
            task = self.tasks.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, visited, stack):
                            return True
                    elif dep in stack:
                        return True
            
            stack.remove(task_id)
            return False
        
        visited = set()
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    return False
        return True
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[Task]:
        """
        Obtiene tareas listas para ejecutar
        
        Args:
            completed_tasks: Set de IDs de tareas completadas
            
        Returns:
            Lista de tareas listas
        """
        ready = []
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                task.can_execute(completed_tasks)):
                ready.append(task)
        return ready
    
    def execute(self, parallel: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el workflow
        
        Args:
            parallel: Si True, ejecuta tareas independientes en paralelo (simulado)
            
        Returns:
            Reporte de ejecución
        """
        print(f"\n\n{'='*70}")
        print(f"🚀 EJECUTANDO WORKFLOW: {self.name}")
        print(f"{'='*70}")
        
        # Validar DAG
        if not self.validate_dag():
            print("❌ Error: El workflow contiene ciclos de dependencia")
            return {"status": "invalid", "error": "Circular dependencies detected"}
        
        print(f"✅ DAG válido - {len(self.tasks)} tareas")
        print(f"⚙️  Modo: {'Paralelo' if parallel else 'Secuencial'}")
        
        completed_tasks = set()
        failed_tasks = set()
        start_time = time.time()
        
        iteration = 0
        while len(completed_tasks) + len(failed_tasks) < len(self.tasks):
            iteration += 1
            print(f"\n--- Iteración {iteration} ---")
            
            # Obtener tareas listas
            ready_tasks = self.get_ready_tasks(completed_tasks)
            
            if not ready_tasks:
                # No hay tareas listas, verificar si quedaron pendientes
                pending = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
                if pending:
                    print("⚠️ Hay tareas pendientes pero ninguna lista (dependencias no satisfechas)")
                    for task in pending:
                        task.status = TaskStatus.SKIPPED
                break
            
            print(f"📋 Tareas listas: {len(ready_tasks)}")
            
            # Ejecutar tareas (simular paralelismo)
            for task in ready_tasks:
                success = task.execute(self.context)
                
                if success:
                    completed_tasks.add(task.id)
                    # Agregar resultado al contexto
                    self.context[task.id] = task.result
                    
                    # Log
                    self.execution_log.append({
                        "task_id": task.id,
                        "task_name": task.name,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    if task.status == TaskStatus.FAILED:
                        failed_tasks.add(task.id)
                        
                        # Log
                        self.execution_log.append({
                            "task_id": task.id,
                            "task_name": task.name,
                            "status": "failed",
                            "error": task.error,
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Límite de seguridad
            if iteration > 100:
                print("⚠️ Límite de iteraciones alcanzado")
                break
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Reporte final
        return self._generate_report(completed_tasks, failed_tasks, total_time)
    
    def _generate_report(self, completed: Set[str], failed: Set[str], total_time: float) -> Dict[str, Any]:
        """Genera reporte de ejecución"""
        print(f"\n\n{'='*70}")
        print(f"📊 REPORTE DE EJECUCIÓN: {self.name}")
        print(f"{'='*70}")
        
        total = len(self.tasks)
        completed_count = len(completed)
        failed_count = len(failed)
        skipped_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)
        
        print(f"\n📈 Estadísticas:")
        print(f"   Total de tareas: {total}")
        print(f"   ✅ Completadas: {completed_count} ({completed_count/total*100:.1f}%)")
        print(f"   ❌ Fallidas: {failed_count} ({failed_count/total*100:.1f}%)")
        print(f"   ⏭️  Omitidas: {skipped_count} ({skipped_count/total*100:.1f}%)")
        print(f"   ⏱️  Tiempo total: {total_time:.2f}s")
        
        # Detalle por tarea
        print(f"\n📋 Detalle de Tareas:")
        for task in self.tasks.values():
            status_icon = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭️",
                TaskStatus.PENDING: "⏸️"
            }.get(task.status, "❓")
            
            duration = ""
            if task.start_time and task.end_time:
                duration = f"({task.end_time - task.start_time:.2f}s)"
            
            print(f"   {status_icon} {task.name} {duration}")
            if task.error:
                print(f"      Error: {task.error}")
        
        return {
            "workflow": self.name,
            "status": "completed" if failed_count == 0 else "partial",
            "total_tasks": total,
            "completed": completed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_time": total_time,
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "status": t.status.value,
                    "result": t.result,
                    "error": t.error
                }
                for t in self.tasks.values()
            ]
        }


def demo_data_pipeline():
    """
    Demostración: Pipeline de Procesamiento de Datos
    """
    print("="*70)
    print("  🎓 DEMOSTRACIÓN: PIPELINE DE DATOS")
    print("="*70)
    
    workflow = WorkflowManager("Data Processing Pipeline")
    
    # Definir tareas
    def extract_data(ctx):
        """Extraer datos de fuente"""
        time.sleep(0.2)
        return {"records": 1000, "source": "database"}
    
    def validate_data(ctx):
        """Validar datos extraídos"""
        time.sleep(0.15)
        data = ctx.get("extract")
        return {"valid": True, "records": data["records"]}
    
    def transform_data(ctx):
        """Transformar datos"""
        time.sleep(0.25)
        data = ctx.get("validate")
        return {"transformed": data["records"], "format": "parquet"}
    
    def load_data(ctx):
        """Cargar datos en destino"""
        time.sleep(0.2)
        data = ctx.get("transform")
        return {"loaded": data["transformed"], "destination": "data_warehouse"}
    
    def generate_report(ctx):
        """Generar reporte"""
        time.sleep(0.1)
        load_result = ctx.get("load")
        return {"report": f"Procesados {load_result['loaded']} registros exitosamente"}
    
    # Añadir tareas con dependencias
    workflow.add_task(Task("extract", "Extraer Datos", extract_data, []))
    workflow.add_task(Task("validate", "Validar Datos", validate_data, ["extract"]))
    workflow.add_task(Task("transform", "Transformar Datos", transform_data, ["validate"]))
    workflow.add_task(Task("load", "Cargar Datos", load_data, ["transform"]))
    workflow.add_task(Task("report", "Generar Reporte", generate_report, ["load"]))
    
    # Ejecutar
    result = workflow.execute(parallel=False)
    
    return result


def demo_ci_cd_pipeline():
    """
    Demostración: Pipeline CI/CD
    """
    print("\n\n" + "="*70)
    print("  🚀 DEMOSTRACIÓN: CI/CD PIPELINE")
    print("="*70)
    
    workflow = WorkflowManager("CI/CD Pipeline")
    
    # Tareas
    def checkout_code(ctx):
        time.sleep(0.1)
        return {"repo": "my-app", "branch": "main"}
    
    def install_dependencies(ctx):
        time.sleep(0.2)
        return {"packages": 50, "installed": True}
    
    def run_linter(ctx):
        time.sleep(0.15)
        return {"errors": 0, "warnings": 3}
    
    def run_unit_tests(ctx):
        time.sleep(0.25)
        return {"tests": 150, "passed": 150, "failed": 0}
    
    def run_integration_tests(ctx):
        time.sleep(0.3)
        return {"tests": 30, "passed": 30, "failed": 0}
    
    def build_docker_image(ctx):
        time.sleep(0.2)
        return {"image": "my-app:latest", "size": "250MB"}
    
    def deploy_staging(ctx):
        time.sleep(0.2)
        return {"environment": "staging", "url": "https://staging.myapp.com"}
    
    def deploy_production(ctx):
        time.sleep(0.2)
        return {"environment": "production", "url": "https://myapp.com"}
    
    # Construir workflow
    workflow.add_task(Task("checkout", "Checkout Code", checkout_code, []))
    workflow.add_task(Task("install", "Install Dependencies", install_dependencies, ["checkout"]))
    workflow.add_task(Task("lint", "Run Linter", run_linter, ["install"]))
    workflow.add_task(Task("unit_tests", "Run Unit Tests", run_unit_tests, ["install"]))
    workflow.add_task(Task("integration_tests", "Run Integration Tests", run_integration_tests, ["install"]))
    workflow.add_task(Task("build", "Build Docker Image", build_docker_image, ["lint", "unit_tests", "integration_tests"]))
    workflow.add_task(Task("deploy_stg", "Deploy to Staging", deploy_staging, ["build"]))
    workflow.add_task(Task("deploy_prod", "Deploy to Production", deploy_production, ["deploy_stg"]))
    
    # Ejecutar
    result = workflow.execute(parallel=True)
    
    return result


def demo_ml_training_pipeline():
    """
    Demostración: Pipeline de Entrenamiento de ML
    """
    print("\n\n" + "="*70)
    print("  🤖 DEMOSTRACIÓN: ML TRAINING PIPELINE")
    print("="*70)
    
    workflow = WorkflowManager("ML Training Pipeline")
    
    # Tareas
    def load_dataset(ctx):
        time.sleep(0.15)
        return {"samples": 10000, "features": 20}
    
    def preprocess_data(ctx):
        time.sleep(0.2)
        return {"cleaned": 9500, "outliers_removed": 500}
    
    def feature_engineering(ctx):
        time.sleep(0.25)
        return {"features": 35, "engineered": 15}
    
    def split_dataset(ctx):
        time.sleep(0.1)
        return {"train": 7600, "val": 950, "test": 950}
    
    def train_model(ctx):
        time.sleep(0.4)
        return {"accuracy": 0.92, "loss": 0.15}
    
    def evaluate_model(ctx):
        time.sleep(0.2)
        return {"test_accuracy": 0.91, "f1_score": 0.89}
    
    def save_model(ctx):
        time.sleep(0.1)
        return {"model_path": "/models/model_v1.pkl", "size": "15MB"}
    
    # Construir workflow
    workflow.add_task(Task("load", "Load Dataset", load_dataset, []))
    workflow.add_task(Task("preprocess", "Preprocess Data", preprocess_data, ["load"]))
    workflow.add_task(Task("features", "Feature Engineering", feature_engineering, ["preprocess"]))
    workflow.add_task(Task("split", "Split Dataset", split_dataset, ["features"]))
    workflow.add_task(Task("train", "Train Model", train_model, ["split"], retries=2))
    workflow.add_task(Task("evaluate", "Evaluate Model", evaluate_model, ["train"]))
    workflow.add_task(Task("save", "Save Model", save_model, ["evaluate"]))
    
    # Ejecutar
    result = workflow.execute(parallel=False)
    
    return result


if __name__ == "__main__":
    # Ejecutar demostraciones
    demo_data_pipeline()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver CI/CD Pipeline...")
    demo_ci_cd_pipeline()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver ML Training Pipeline...")
    demo_ml_training_pipeline()
    
    # Lecciones finales
    print("\n\n" + "="*70)
    print("  💡 LECCIONES CLAVE PARA ESTUDIANTES")
    print("="*70)
    print("""
    1. Los workflows permiten automatizar procesos complejos con dependencias
    2. El DAG (grafo acíclico dirigido) asegura un orden lógico de ejecución
    3. Las tareas independientes pueden ejecutarse en paralelo para eficiencia
    4. El manejo de errores y reintentos aumenta la robustez
    5. Los workflows son fundamentales en DevOps, Data Engineering y MLOps
    
    💭 Reflexión: ¿Qué procesos en tu trabajo/estudio podrían automatizarse con workflows?
    """)

