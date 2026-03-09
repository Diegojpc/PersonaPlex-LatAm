# Tareas de Adaptación PersonaPlex (LatAm)

- [ ] Fase 1: Exploración y Baseline Empírico
  - [x] Crear entorno de experimentación aislado (`latam_experiments`).
  - [x] Implementar framework de logging de trazabilidad estricta.
  - [x] Desarrollar y ejecutar Auditoría de Tokenización (Evaluación de fertilidad en Español).
  - [x] Desarrollar y ejecutar Evaluador de Baseline Full-Duplex (Identificar puntos de colapso).
- [ ] Fase 2: Pipeline de Datos Acústico-Semánticos (Offline & CPU-friendly)
  - [ ] Diseñar script autogenerador de diálogos sintéticos (Español Colombia) usando LLMs.
  - [ ] Integrar motor TTS de la comunidad (ej. Coqui TTS/XTTSv2) para generar audio de manera local en lotes.
  - [ ] Procesamiento offline de audio a tokens RVQ discretos (extrayendo features de audio a texto sin requerir la GPU de inferencia completa).
- [ ] Fase 3: Estrategia Alternativa de Adaptación (Zero-Shot / Prompting Avanzado)
  - [ ] Diseñar inyección de contexto profundo (ICL) para forzar el comportamiento sin modificar los pesos.
  - [ ] Investigar *Quantized LoRA (QLoRA)* u otras técnicas extremas de bajo consumo si el usuario llegara a disponer de una GPU de consumidor (ej: RTX 3090/4090 de 24GB).
- [ ] Fase 4: Evaluación Objetiva y Trazabilidad MLOps
  - [ ] Implementar métricas acústicas analíticas (MCD, PESQ).
  - [ ] Implementar LLM-as-a-Judge script para medir fidelidad de Persona y degradación fonética en las salidas.
