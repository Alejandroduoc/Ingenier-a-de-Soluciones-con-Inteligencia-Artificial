"""
IL2.3: Coordinación Avanzada Multi-Agente
=======================================

Este módulo implementa mecanismos avanzados de coordinación entre múltiples
agentes, incluyendo comunicación, sincronización y colaboración.

Conceptos Clave:
- Protocolos de comunicación entre agentes
- Sincronización de acciones
- Compartición de conocimiento
- Coordinación distribuida
- Consenso y votación

Para Estudiantes:
La coordinación multi-agente es esencial cuando varios agentes autónomos deben
trabajar juntos hacia un objetivo común. Similar a cómo un equipo humano se
coordina mediante reuniones, mensajes y acuerdos.
"""

from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import time
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

print("✅ Módulo de coordinación multi-agente cargado\n")


class MessageType(Enum):
    """Tipos de mensajes entre agentes"""
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class Message:
    """
    Mensaje entre agentes
    
    Atributos:
        sender: ID del agente emisor
        receiver: ID del agente receptor
        type: Tipo de mensaje
        content: Contenido del mensaje
        timestamp: Marca de tiempo
    """
    sender: str
    receiver: str
    type: MessageType
    content: Any
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        return f"[{self.type.value}] {self.sender} → {self.receiver}: {self.content}"


class CoordinatedAgent:
    """
    Agente con capacidades de coordinación
    
    Atributos:
        id: Identificador único
        name: Nombre del agente
        knowledge: Base de conocimiento
        inbox: Buzón de mensajes recibidos
        capabilities: Capacidades del agente
    """
    
    def __init__(self, id: str, name: str, capabilities: List[str]):
        self.id = id
        self.name = name
        self.knowledge: Dict[str, Any] = {}
        self.inbox: List[Message] = []
        self.capabilities = capabilities
        self.coordinator = None
        
        print(f"🤖 Agente coordinado '{name}' creado")
        print(f"   Capacidades: {', '.join(capabilities)}")
    
    def set_coordinator(self, coordinator):
        """Establece el coordinador del agente"""
        self.coordinator = coordinator
    
    def send_message(self, receiver_id: str, msg_type: MessageType, content: Any):
        """Envía un mensaje a otro agente"""
        message = Message(
            sender=self.id,
            receiver=receiver_id,
            type=msg_type,
            content=content
        )
        
        if self.coordinator:
            self.coordinator.deliver_message(message)
            print(f"   📤 {self.name} envió: {message}")
    
    def receive_message(self, message: Message):
        """Recibe un mensaje"""
        self.inbox.append(message)
        print(f"   📥 {self.name} recibió: {message}")
    
    def process_messages(self):
        """Procesa mensajes en el buzón"""
        if not self.inbox:
            return
        
        print(f"\n⚙️  {self.name} procesando {len(self.inbox)} mensajes...")
        
        for message in self.inbox:
            self._handle_message(message)
        
        self.inbox.clear()
    
    def _handle_message(self, message: Message):
        """Maneja un mensaje recibido"""
        if message.type == MessageType.REQUEST:
            self._handle_request(message)
        elif message.type == MessageType.QUERY:
            self._handle_query(message)
        elif message.type == MessageType.INFORM:
            self._handle_inform(message)
        elif message.type == MessageType.PROPOSE:
            self._handle_propose(message)
    
    def _handle_request(self, message: Message):
        """Maneja una solicitud"""
        print(f"      {self.name} procesando solicitud: {message.content}")
        # Responder
        self.send_message(message.sender, MessageType.RESPONSE, 
                         f"Solicitud procesada: {message.content}")
    
    def _handle_query(self, message: Message):
        """Maneja una consulta"""
        query = message.content
        response = self.knowledge.get(query, "Información no disponible")
        self.send_message(message.sender, MessageType.RESPONSE, response)
    
    def _handle_inform(self, message: Message):
        """Maneja información compartida"""
        if isinstance(message.content, dict):
            self.knowledge.update(message.content)
            print(f"      {self.name} actualizó conocimiento")
    
    def _handle_propose(self, message: Message):
        """Maneja una propuesta"""
        # Decidir si aceptar o rechazar
        accept = True  # Simplificado
        response_type = MessageType.ACCEPT if accept else MessageType.REJECT
        self.send_message(message.sender, response_type, message.content)
    
    def share_knowledge(self, knowledge: Dict[str, Any], target_agents: List[str] = None):
        """Comparte conocimiento con otros agentes"""
        if target_agents is None and self.coordinator:
            target_agents = [a.id for a in self.coordinator.agents.values() if a.id != self.id]
        
        for agent_id in target_agents:
            self.send_message(agent_id, MessageType.INFORM, knowledge)


class Coordinator:
    """
    Coordinador central para múltiples agentes
    
    Atributos:
        name: Nombre del coordinador
        agents: Diccionario de agentes registrados
        message_log: Log de todos los mensajes
    """
    
    def __init__(self, name: str = "Central Coordinator"):
        self.name = name
        self.agents: Dict[str, CoordinatedAgent] = {}
        self.message_log: List[Message] = []
        self.tasks: Dict[str, Any] = {}
        
        print(f"\n🎭 Coordinador '{name}' inicializado")
    
    def register_agent(self, agent: CoordinatedAgent):
        """Registra un agente"""
        self.agents[agent.id] = agent
        agent.set_coordinator(self)
        print(f"   ✅ Agente '{agent.name}' registrado")
    
    def deliver_message(self, message: Message):
        """Entrega un mensaje al destinatario"""
        self.message_log.append(message)
        
        receiver = self.agents.get(message.receiver)
        if receiver:
            receiver.receive_message(message)
        else:
            print(f"   ⚠️ Destinatario {message.receiver} no encontrado")
    
    def broadcast(self, sender_id: str, msg_type: MessageType, content: Any):
        """Envía un mensaje a todos los agentes"""
        print(f"\n📢 Broadcast desde {sender_id}: {content}")
        
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                message = Message(sender_id, agent_id, msg_type, content)
                self.deliver_message(message)
    
    def coordinate_task(self, task_id: str, task: Dict[str, Any]):
        """Coordina la ejecución de una tarea entre agentes"""
        print(f"\n\n🎯 Coordinando tarea: {task_id}")
        print(f"   Descripción: {task.get('description', 'N/A')}")
        
        self.tasks[task_id] = task
        
        # Anunciar tarea
        self.broadcast("coordinator", MessageType.INFORM, {
            "task_id": task_id,
            "task": task
        })
        
        # Procesar mensajes
        self.process_all_messages()
        
        # Asignar sub-tareas según capacidades
        subtasks = task.get("subtasks", [])
        for i, subtask in enumerate(subtasks):
            required_cap = subtask.get("capability")
            suitable_agents = [
                a for a in self.agents.values()
                if required_cap in a.capabilities
            ]
            
            if suitable_agents:
                selected = suitable_agents[0]
                print(f"   ✅ Sub-tarea {i+1} asignada a {selected.name}")
                
                message = Message(
                    "coordinator",
                    selected.id,
                    MessageType.REQUEST,
                    subtask
                )
                self.deliver_message(message)
    
    def process_all_messages(self):
        """Procesa todos los mensajes pendientes"""
        for agent in self.agents.values():
            agent.process_messages()
    
    def voting_consensus(self, proposal: str) -> Dict[str, Any]:
        """
        Realiza votación para llegar a consenso
        
        Args:
            proposal: Propuesta a votar
            
        Returns:
            Resultado de la votación
        """
        print(f"\n\n🗳️  Iniciando votación: '{proposal}'")
        print("="*70)
        
        # Solicitar votos
        self.broadcast("coordinator", MessageType.PROPOSE, proposal)
        self.process_all_messages()
        
        # Contar votos
        votes = {"accept": 0, "reject": 0}
        
        for message in self.message_log[-len(self.agents):]:
            if message.type == MessageType.ACCEPT:
                votes["accept"] += 1
                print(f"   ✅ {message.sender} votó a favor")
            elif message.type == MessageType.REJECT:
                votes["reject"] += 1
                print(f"   ❌ {message.sender} votó en contra")
        
        # Resultado
        total = votes["accept"] + votes["reject"]
        consensus = votes["accept"] > votes["reject"]
        
        result = {
            "proposal": proposal,
            "votes_for": votes["accept"],
            "votes_against": votes["reject"],
            "total_votes": total,
            "consensus": consensus,
            "percentage_for": (votes["accept"] / total * 100) if total > 0 else 0
        }
        
        print(f"\n📊 Resultado:")
        print(f"   A favor: {votes['accept']} ({result['percentage_for']:.1f}%)")
        print(f"   En contra: {votes['reject']}")
        print(f"   Consenso: {'✅ SÍ' if consensus else '❌ NO'}")
        
        return result
    
    def generate_communication_report(self):
        """Genera reporte de comunicación"""
        print(f"\n\n{'='*70}")
        print(f"📊 REPORTE DE COMUNICACIÓN")
        print(f"{'='*70}")
        
        # Estadísticas generales
        print(f"\n📈 Estadísticas Generales:")
        print(f"   Total de mensajes: {len(self.message_log)}")
        print(f"   Agentes registrados: {len(self.agents)}")
        
        # Mensajes por tipo
        type_counts = defaultdict(int)
        for msg in self.message_log:
            type_counts[msg.type.value] += 1
        
        print(f"\n📨 Mensajes por Tipo:")
        for msg_type, count in sorted(type_counts.items()):
            print(f"   {msg_type}: {count}")
        
        # Actividad por agente
        print(f"\n👥 Actividad por Agente:")
        agent_stats = defaultdict(lambda: {"sent": 0, "received": 0})
        
        for msg in self.message_log:
            agent_stats[msg.sender]["sent"] += 1
            agent_stats[msg.receiver]["received"] += 1
        
        for agent_id, stats in agent_stats.items():
            agent_name = self.agents.get(agent_id, type('obj', (), {'name': agent_id})()).name
            total = stats["sent"] + stats["received"]
            print(f"   {agent_name}: {total} mensajes (↗{stats['sent']} ↘{stats['received']})")


def demo_collaborative_research():
    """
    Demostración: Equipo de Investigación Colaborativa
    """
    print("="*70)
    print("  🎓 DEMOSTRACIÓN: INVESTIGACIÓN COLABORATIVA")
    print("="*70)
    
    coordinator = Coordinator("Research Coordinator")
    
    # Crear agentes investigadores
    print("\n👥 Creando equipo de investigación:")
    
    data_scientist = CoordinatedAgent(
        "ds1",
        "Data Scientist",
        ["análisis_datos", "estadística", "ml"]
    )
    
    literature_expert = CoordinatedAgent(
        "lit1",
        "Literature Expert",
        ["revisión_literatura", "bibliografía", "análisis_texto"]
    )
    
    writer = CoordinatedAgent(
        "wr1",
        "Scientific Writer",
        ["redacción", "publicación", "formato"]
    )
    
    # Registrar agentes
    coordinator.register_agent(data_scientist)
    coordinator.register_agent(literature_expert)
    coordinator.register_agent(writer)
    
    # Coordinar tarea de investigación
    research_task = {
        "description": "Investigación sobre IA en Educación",
        "subtasks": [
            {"id": "st1", "capability": "revisión_literatura", 
             "description": "Revisar papers de últimos 5 años"},
            {"id": "st2", "capability": "análisis_datos", 
             "description": "Analizar datos de casos de estudio"},
            {"id": "st3", "capability": "redacción", 
             "description": "Redactar paper científico"}
        ]
    }
    
    coordinator.coordinate_task("research_1", research_task)
    
    # Compartir conocimiento
    print("\n\n📚 Compartiendo conocimiento entre agentes:")
    data_scientist.share_knowledge({
        "hallazgo": "70% de instituciones usan IA en 2024",
        "tendencia": "Crecimiento exponencial"
    })
    
    coordinator.process_all_messages()
    
    # Votación para decisión
    coordinator.voting_consensus("¿Publicar resultados en conferencia internacional?")
    
    # Reporte
    coordinator.generate_communication_report()


def demo_emergency_response():
    """
    Demostración: Sistema de Respuesta a Emergencias
    """
    print("\n\n" + "="*70)
    print("  🚨 DEMOSTRACIÓN: RESPUESTA A EMERGENCIAS")
    print("="*70)
    
    coordinator = Coordinator("Emergency Coordinator")
    
    # Crear agentes de respuesta
    print("\n👥 Creando equipo de respuesta:")
    
    police = CoordinatedAgent("pol1", "Policía", ["seguridad", "orden"])
    fire = CoordinatedAgent("fire1", "Bomberos", ["incendio", "rescate"])
    medical = CoordinatedAgent("med1", "Ambulancia", ["salud", "primeros_auxilios"])
    
    coordinator.register_agent(police)
    coordinator.register_agent(fire)
    coordinator.register_agent(medical)
    
    # Simular emergencia
    emergency = {
        "description": "Incendio en edificio con personas atrapadas",
        "subtasks": [
            {"id": "e1", "capability": "seguridad", 
             "description": "Acordonar área y controlar tráfico"},
            {"id": "e2", "capability": "incendio", 
             "description": "Extinguir incendio y rescatar personas"},
            {"id": "e3", "capability": "salud", 
             "description": "Atender heridos y trasladar a hospital"}
        ]
    }
    
    coordinator.coordinate_task("emergency_1", emergency)
    coordinator.process_all_messages()
    
    # Actualización de situación
    print("\n\n📢 Actualizaciones de situación:")
    fire.share_knowledge({"estado": "Incendio controlado", "rescatados": 5})
    coordinator.process_all_messages()
    
    coordinator.generate_communication_report()


if __name__ == "__main__":
    # Ejecutar demostraciones
    demo_collaborative_research()
    
    print("\n\n" + "="*70)
    input("Presiona ENTER para ver Respuesta a Emergencias...")
    demo_emergency_response()
    
    # Lecciones finales
    print("\n\n" + "="*70)
    print("  💡 LECCIONES CLAVE PARA ESTUDIANTES")
    print("="*70)
    print("""
    1. La comunicación efectiva es clave en sistemas multi-agente
    2. Los protocolos de mensajes estructuran las interacciones
    3. El consenso permite tomar decisiones colectivas
    4. Compartir conocimiento mejora la inteligencia colectiva
    5. La coordinación central simplifica problemas complejos
    
    💭 Reflexión: ¿Qué otros problemas requerirían coordinación multi-agente?
    """)

