# Guía Pedagógica - IL2.3: Planificación y Orquestación

## 📚 Orden Recomendado de Estudio

### Nivel 1: Fundamentos (Semana 1)
**Objetivo**: Entender conceptos básicos de planificación con agentes

1. **`README.md`** - Leer primero para contexto general
2. **`planning-patterns.md`** - Entender patrones de planificación
3. **`1-basic_planning.py`** - Primer agente simple con LangChain
4. **`1-langchain_planning.py`** - Agente con herramientas personalizadas
5. **`1-planning-strategies.py`** - Comparación de estrategias (Python puro)

**Práctica**: Modificar `1-basic_planning.py` para usar tu propia herramienta

---

### Nivel 2: Planificación Avanzada (Semana 2)
**Objetivo**: Dominar diferentes estrategias de planificación

6. **`2-hierarchical-planning.py`** - Descomposición jerárquica con LLM
7. **`3-reactive-planning.py`** - Sistema reactivo con reglas
8. **`4-goal-oriented-planning.py`** - Planificación orientada a objetivos (STRIPS)

**Práctica**: Implementar planificación jerárquica para tu proyecto final

---

### Nivel 3: Orquestación Multi-Agente (Semana 3)
**Objetivo**: Coordinar múltiples agentes

9. **`orchestration-guide.md`** - Leer guía de orquestación
10. **`2-crewai_orchestration.py`** - Primer ejemplo con CrewAI
11. **`5-agent-orchestration.py`** - Orquestación con LangChain
12. **`2-multiagent_orchestration.py`** - Ejemplo básico Python
13. **`6-workflow-management.py`** - Gestión de workflows con DAG

**Práctica**: Crear equipo de 3 agentes que colaboren

---

### Nivel 4: Gestión de Recursos (Semana 4)
**Objetivo**: Optimizar asignación y resolución de conflictos

14. **`7-task-decomposition.py`** - Descomposición inteligente con LLM
15. **`8-resource-allocation.py`** - Asignación optimizada
16. **`10-conflict-resolution.py`** - Resolución de conflictos
17. **`11-negotiation-strategies.py`** - Negociación entre agentes

**Práctica**: Sistema de asignación de recursos para tu dominio

---

### Nivel 5: Coordinación Avanzada (Semana 5)
**Objetivo**: Técnicas avanzadas de coordinación

18. **`coordination-strategies.md`** - Leer estrategias de coordinación
19. **`9-multi-agent-coordination.py`** - Comunicación y consenso
20. **`12-emergence-behaviors.py`** - Comportamientos emergentes

**Práctica**: Implementar sistema con comportamiento emergente

---

## 🎯 Mapa de Herramientas

### Cuándo usar cada herramienta:

#### LangChain (Agentes)
**Usar para**:
- ✅ Agentes individuales con herramientas
- ✅ Cadenas de razonamiento (ReAct)
- ✅ Workflows simples lineales
- ✅ Integración con herramientas externas

**Archivos que lo usan**:
- `1-basic_planning.py`
- `1-langchain_planning.py`
- Parcialmente: `2-hierarchical-planning.py`, `5-agent-orchestration.py`

#### CrewAI (Equipos)
**Usar para**:
- ✅ Múltiples agentes colaborando
- ✅ Roles especializados
- ✅ Workflows jerárquicos o secuenciales
- ✅ Orquestación compleja

**Archivos que lo usan**:
- `2-crewai_orchestration.py`

**⚠️ OPORTUNIDAD**: Deberían agregarse más ejemplos con CrewAI

#### Python Puro
**Usar para**:
- ✅ Demostrar algoritmos fundamentales
- ✅ Lógica de coordinación sin LLM
- ✅ Simulaciones y experimentos
- ✅ Patrones de diseño

**Archivos que lo usan**:
- `1-planning-strategies.py`
- `3-reactive-planning.py`
- `4-goal-oriented-planning.py`
- `6-workflow-management.py`
- `8-resource-allocation.py`
- `9-multi-agent-coordination.py`
- `10-conflict-resolution.py`
- `11-negotiation-strategies.py`
- `12-emergence-behaviors.py`

---

## 🔧 Configuración Requerida

### Variables de Entorno
Todos los archivos que usan LLM requieren:

```bash
# En tu archivo .env
GITHUB_TOKEN=tu_token_aqui
GITHUB_BASE_URL=https://models.inference.ai.azure.com
```

### Dependencias

```bash
# Para LangChain
pip install langchain langchain-openai openai python-dotenv

# Para CrewAI
pip install crewai crewai-tools python-dotenv

# Opcional para ejemplos avanzados
pip install langsmith  # Para evaluación
```

---

## 📈 Progresión de Complejidad

```
Nivel 1: BÁSICO
└── Un agente, una herramienta
    └── Ejemplos: 1-basic_planning.py

Nivel 2: INTERMEDIO
├── Un agente, múltiples herramientas
├── Planificación con descomposición
└── Ejemplos: 1-langchain_planning.py, 2-hierarchical-planning.py

Nivel 3: AVANZADO
├── Múltiples agentes coordinados
├── Orquestación y workflows
└── Ejemplos: 5-agent-orchestration.py, 6-workflow-management.py

Nivel 4: EXPERTO
├── Sistemas complejos multi-agente
├── Emergencia y auto-organización
└── Ejemplos: 9-multi-agent-coordination.py, 12-emergence-behaviors.py
```

---

## ✅ Checklist de Aprendizaje

### Fundamentos
- [ ] Entiendo qué es un agente LLM
- [ ] Puedo crear un agente básico con LangChain
- [ ] Sé crear herramientas personalizadas
- [ ] Entiendo los patrones de planificación

### Planificación
- [ ] Puedo implementar planificación jerárquica
- [ ] Entiendo planificación reactiva vs proactiva
- [ ] Sé cuándo usar cada estrategia
- [ ] Puedo descomponer tareas complejas

### Orquestación
- [ ] Puedo coordinar múltiples agentes
- [ ] Entiendo workflows con dependencias
- [ ] Sé usar tanto LangChain como CrewAI
- [ ] Puedo gestionar recursos entre agentes

### Coordinación
- [ ] Entiendo protocolos de comunicación
- [ ] Puedo resolver conflictos entre agentes
- [ ] Sé implementar negociación
- [ ] Comprendo comportamientos emergentes

---

## 🎓 Proyectos Sugeridos

### Proyecto 1: Asistente de Investigación
**Herramientas**: LangChain + herramientas de búsqueda
**Archivos base**: 1-langchain_planning.py, 7-task-decomposition.py

### Proyecto 2: Sistema de Atención al Cliente
**Herramientas**: CrewAI para equipo especializado
**Archivos base**: 2-crewai_orchestration.py, 8-resource-allocation.py

### Proyecto 3: Pipeline de Procesamiento de Datos
**Herramientas**: Python + LangChain para orquestación
**Archivos base**: 6-workflow-management.py, 4-goal-oriented-planning.py

### Proyecto 4: Sistema de Colaboración Académica
**Herramientas**: Multi-agente con coordinación
**Archivos base**: 9-multi-agent-coordination.py, 10-conflict-resolution.py

---

## ⚠️ Errores Comunes a Evitar

### 1. Usar LLM para Todo
❌ **Error**: Usar LLM para cálculos simples o lógica básica
✅ **Correcto**: Usar Python puro cuando sea apropiado

### 2. No Gestionar Contexto
❌ **Error**: Perder contexto entre llamadas
✅ **Correcto**: Usar memoria o pasar contexto explícitamente

### 3. Ignorar Dependencias
❌ **Error**: Ejecutar tareas sin verificar precondiciones
✅ **Correcto**: Validar dependencias antes de ejecutar

### 4. Sobrecomplicar
❌ **Error**: Usar multi-agente cuando un agente basta
✅ **Correcto**: Empezar simple, complejizar solo si es necesario

---

## 📊 Evaluación y Métricas

### Auto-evaluación
Después de cada nivel, pregúntate:
1. ¿Puedo explicar el concepto a un compañero?
2. ¿Puedo implementarlo desde cero?
3. ¿Entiendo cuándo usarlo y cuándo no?
4. ¿Puedo identificar problemas y solucionarlos?

### Métricas de Proyecto
Para tu proyecto final, evalúa:
- ✓ Funcionalidad (¿Hace lo que debe?)
- ✓ Eficiencia (¿Usa recursos apropiadamente?)
- ✓ Robustez (¿Maneja errores?)
- ✓ Escalabilidad (¿Puede crecer?)
- ✓ Mantenibilidad (¿Es fácil de entender?)

---

## 🔗 Recursos Adicionales

### Documentación Oficial
- [LangChain Docs](https://python.langchain.com/)
- [CrewAI Docs](https://docs.crewai.com/)
- [OpenAI API](https://platform.openai.com/docs)

### Papers Clásicos
1. "STRIPS: A New Approach to the Application of Theorem Proving" (1971)
2. "BDI Agents: From Theory to Practice" (1995)
3. "Contract Net Protocol" (Smith, 1980)

### Comunidad
- Discord de LangChain
- GitHub Discussions de CrewAI
- Stack Overflow (tag: langchain, multi-agent-systems)

---

**Última actualización**: Octubre 2024
**Autor**: Módulo IL2.3 - Ingeniería de Soluciones con IA
**Versión**: 1.0


