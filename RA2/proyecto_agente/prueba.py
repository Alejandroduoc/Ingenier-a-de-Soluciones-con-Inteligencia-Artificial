
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
try:
    from langchain_core.memory import ConversationBufferMemory
    print("Importación exitosa desde langchain_core.memory")
except ModuleNotFoundError:
    try:
        from langchain_core.memory.buffer import ConversationBufferMemory
        print("Importación exitosa desde langchain_core.memory.buffer")
    except ModuleNotFoundError as e:
        print(f"Error en ambas importaciones: {e}")
