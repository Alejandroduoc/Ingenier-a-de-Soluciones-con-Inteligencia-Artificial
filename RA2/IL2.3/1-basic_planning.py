"""
IL2.3: Planificación Básica con LangChain
========================================
Ejemplo de cómo un agente LangChain puede planificar y ejecutar pasos simples usando una herramienta.
"""

# Requiere: pip install langchain langchain-openai openai python-dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
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

# Configurar usando variables de entorno
llm = ChatOpenAI(
    model="gpt-4o",
    base_url=github_base_url,
    api_key=github_token,
    temperature=0
)

print("✅ LLM configurado correctamente")

# Herramienta personalizada: pasos para preparar café
def pasos_cafe(_):
    return "1. Calentar agua\n2. Añadir café al filtro\n3. Verter agua caliente\n4. Servir en una taza"

herramienta_cafe = Tool(
    name="PasosCafé",
    func=pasos_cafe,
    description="Devuelve los pasos para preparar café."
)

# Inicializa el agente
agente = initialize_agent(
    tools=[herramienta_cafe],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("Planificación con LangChain:")
    print(agente.run("¿Cuáles son los pasos para preparar café?")) 