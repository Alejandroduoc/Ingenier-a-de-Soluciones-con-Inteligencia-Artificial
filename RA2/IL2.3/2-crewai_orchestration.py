"""
IL2.3: Orquestación Multi-Agente con CrewAI
==========================================
Ejemplo de cómo dos agentes CrewAI colaboran para resolver una tarea.
"""

# Requiere: pip install crewai
from crewai import Agent, Task, Crew
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


os.environ["OPENAI_API_KEY"] = github_token
os.environ["OPENAI_API_BASE"] = github_base_url
os.environ["OPENAI_MODEL"] = "gpt-4o"
print("✅ Variables de entorno configuradas correctamente")
# Agente 1: Investigador
investigador = Agent(
    role="Investigador",
    goal="Buscar información sobre la capital de Francia",
    backstory="Eres experto en encontrar datos rápidos."
)

# Agente 2: Redactor
redactor = Agent(
    role="Redactor",
    goal="Redactar una respuesta clara y breve",
    backstory="Eres especialista en explicar conceptos de forma sencilla."
)

# Tareas
tarea_investigar = Task(
    description="Busca cuál es la capital de Francia",
    expected_output="El nombre de la capital de Francia.",
    agent=investigador
)
tarea_redactar = Task(
    description="Redacta una respuesta usando la información encontrada",
    expected_output="Una respuesta clara y breve sobre la capital de Francia.",
    agent=redactor,
    context=[tarea_investigar]
)

# Crew (equipo)
crew = Crew(
    agents=[investigador, redactor],
    tasks=[tarea_investigar, tarea_redactar],
    verbose=True
)

if __name__ == "__main__":
    print("Orquestación multi-agente con CrewAI:")
    print(crew.kickoff()) 