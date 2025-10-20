# Patrones de Planificación para Agentes LLM

## 📋 Introducción

Este documento presenta patrones comunes de planificación utilizados en sistemas de agentes inteligentes. Cada patrón resuelve problemas específicos y tiene sus propias ventajas y limitaciones.

---

## 🎯 Patrones Principales

### 1. Planificación Jerárquica

**Descripción**: Descompone objetivos complejos en sub-objetivos organizados en niveles de abstracción.

**Cuándo usar**:
- Proyectos grandes con múltiples fases
- Cuando necesitas visión desde diferentes niveles
- Tareas que requieren descomposición estructurada

**Ejemplo**:
```
Objetivo: Desarrollar Aplicación Web
├── Nivel Alto
│   ├── Fase de Diseño
│   ├── Fase de Desarrollo
│   └── Fase de Deployment
├── Nivel Medio
│   ├── Diseñar Base de Datos
│   ├── Crear API REST
│   └── Desarrollar Frontend
└── Nivel Bajo
    ├── Definir esquemas de tablas
    ├── Implementar endpoints
    └── Crear componentes UI
```

**Ventajas**:
- ✅ Estructura clara y organizada
- ✅ Fácil de entender y comunicar
- ✅ Permite delegar en diferentes niveles

**Desventajas**:
- ❌ Puede ser rígida ante cambios
- ❌ Requiere planificación previa extensa

---

### 2. Planificación Reactiva

**Descripción**: Responde inmediatamente a eventos del entorno mediante reglas if-then.

**Cuándo usar**:
- Entornos dinámicos e impredecibles
- Sistemas de tiempo real
- Cuando la velocidad de respuesta es crítica

**Ejemplo**:
```python
Reglas:
- SI temperatura > 30°C → ENTONCES activar aire acondicionado
- SI presión < 980 hPa → ENTONCES alerta de tormenta
- SI movimiento detectado → ENTONCES activar alarma
```

**Ventajas**:
- ✅ Respuesta rápida
- ✅ No requiere planificación compleja
- ✅ Funciona bien en tiempo real

**Desventajas**:
- ❌ Sin visión a largo plazo
- ❌ Puede entrar en ciclos reactivos
- ❌ Limitado para problemas complejos

---

### 3. Planificación Orientada a Objetivos (STRIPS-like)

**Descripción**: Búsqueda de secuencia de acciones que transforma estado inicial en estado objetivo.

**Cuándo usar**:
- Cuando conoces exactamente qué quieres lograr
- Problemas con estados bien definidos
- Necesitas la secuencia óptima de pasos

**Ejemplo**:
```
Estado Inicial: {robot_en_A}
Estado Objetivo: {robot_en_D}

Acciones:
1. ir_A_a_B (precondición: robot_en_A, puerta_AB_abierta)
2. abrir_puerta_AB (precondición: robot_en_A)
3. ir_B_a_C (precondición: robot_en_B, puerta_BC_abierta)
...

Plan generado:
1. abrir_puerta_AB
2. ir_A_a_B
3. abrir_puerta_BC
4. ir_B_a_C
5. ir_C_a_D
```

**Ventajas**:
- ✅ Encuentra caminos óptimos
- ✅ Garantiza alcanzar el objetivo
- ✅ Lógica clara de precondiciones/efectos

**Desventajas**:
- ❌ Computacionalmente costoso
- ❌ Requiere modelo del mundo completo
- ❌ Difícil en entornos inciertos

---

### 4. Planificación Adaptativa

**Descripción**: Combina planificación inicial con re-planificación basada en resultados.

**Cuándo usar**:
- Entornos parcialmente predecibles
- Cuando las suposiciones pueden fallar
- Proyectos a largo plazo con incertidumbre

**Patrón**:
```
1. Crear plan inicial
2. Ejecutar primer paso
3. Observar resultados
4. SI resultados != esperados:
     Re-planificar
5. Continuar con siguiente paso
```

**Ventajas**:
- ✅ Flexibilidad ante imprevistos
- ✅ Combina proactividad y reactividad
- ✅ Robusto en entornos reales

**Desventajas**:
- ❌ Más complejo de implementar
- ❌ Overhead de re-planificación

---

### 5. Planificación Continua

**Descripción**: Planificación y ejecución suceden simultáneamente.

**Cuándo usar**:
- Robots en entornos dinámicos
- Sistemas que no pueden pausar para planificar
- Cuando el entorno cambia constantemente

**Características**:
- 🔄 Planificación en paralelo con ejecución
- 🔄 Ajustes en tiempo real
- 🔄 No espera a plan completo

**Ventajas**:
- ✅ Muy adaptable
- ✅ Funciona en tiempo real
- ✅ Responde a cambios inmediatos

**Desventajas**:
- ❌ Complejo de coordinar
- ❌ Puede ser menos óptimo

---

## 🔄 Patrones Híbridos

### Jerárquico-Reactivo

Combina niveles altos de planificación jerárquica con respuesta reactiva en niveles bajos.

**Ejemplo**: Planificación estratégica + Reflejos tácticos

### Reactivo con Memoria

Planificación reactiva que aprende de experiencias pasadas.

**Ejemplo**: Sistema de reglas que se ajustan según resultados

---

## 📊 Comparación de Patrones

| Patrón | Complejidad | Flexibilidad | Optimalidad | Uso de Recursos |
|--------|-------------|--------------|-------------|-----------------|
| Jerárquica | Media | Baja | Alta | Medio |
| Reactiva | Baja | Alta | Baja | Bajo |
| Orientada a Objetivos | Alta | Baja | Muy Alta | Alto |
| Adaptativa | Alta | Alta | Media | Medio-Alto |
| Continua | Muy Alta | Muy Alta | Variable | Alto |

---

## 💡 Guía de Selección

### Usa Planificación Jerárquica si:
- ✓ El proyecto es grande y estructurado
- ✓ Necesitas comunicar plan a stakeholders
- ✓ Hay equipos trabajando en paralelo

### Usa Planificación Reactiva si:
- ✓ El entorno cambia rápidamente
- ✓ La velocidad es más importante que la optimalidad
- ✓ Las reglas son claras y no muy numerosas

### Usa Planificación Orientada a Objetivos si:
- ✓ Conoces el estado inicial y objetivo
- ✓ Necesitas el camino óptimo
- ✓ El entorno es predecible

### Usa Planificación Adaptativa si:
- ✓ Hay incertidumbre significativa
- ✓ Puedes permitir re-planificación
- ✓ El costo de error es alto

---

## 🎓 Ejercicios para Estudiantes

### Ejercicio 1: Identificación
Identifica qué patrón de planificación usarías para:
1. Sistema de control de tráfico urbano
2. Planificación de tu tesis de grado
3. Robot aspiradora autónoma
4. Sistema de recomendaciones de Netflix

### Ejercicio 2: Diseño
Diseña un plan jerárquico para "Organizar una conferencia académica"

### Ejercicio 3: Implementación
Implementa 5 reglas reactivas para un sistema de hogar inteligente

---

## 📚 Referencias y Lecturas Adicionales

1. **STRIPS**: Fikes & Nilsson (1971) - "STRIPS: A new approach to the application of theorem proving to problem solving"
2. **HTN Planning**: Nau et al. (2003) - "SHOP2: An HTN planning system"
3. **Reactive Planning**: Brooks (1986) - "A Robust Layered Control System for a Mobile Robot"
4. **BDI Architecture**: Rao & Georgeff (1995) - "BDI Agents: From Theory to Practice"

---

## 🔗 Recursos Relacionados

- [orchestration-guide.md](orchestration-guide.md) - Guía de orquestación de agentes
- [coordination-strategies.md](coordination-strategies.md) - Estrategias de coordinación

---

**Autor**: Módulo IL2.3 - Ingeniería de Soluciones con IA  
**Actualizado**: 2024  
**Licencia**: Uso Educativo

