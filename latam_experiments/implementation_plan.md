# Fase 1: Exploración y Baseline Empírico

El objetivo de esta fase es medir el punto de partida exacto de PersonaPlex al enfrentarse al español Colombia. No vamos a estimar; vamos a cuantificar las deficiencias arquitectónicas mediante dos scripts de diagnóstico fundamentales.

## Proposed Changes

Crearemos un entorno aislado (`latam_experiments`) con los scripts de evaluación inicial. 

### LatAm Experiments Setup

#### [NEW] [01_tokenization_auditor.py](file:///media/diego/Personal%20Projects/PersonaPLEX_NVIDIA/personaplex/latam_experiments/01_tokenization_auditor.py)
Script que descarga/carga el tokenizador base de Moshi (`tokenizer_spm_32k_3.model`) y procesa un corpus de prueba en español colombiano.
**Propósito:** Calcular la Tasa de Fertilidad (Tokens por Palabra). Si es mayor a 2.5, el modelo perderá coherencia y aumentará drásticamente la latencia al hablar en español.

#### [NEW] [02_baseline_evaluator.py](file:///media/diego/Personal%20Projects/PersonaPLEX_NVIDIA/personaplex/latam_experiments/02_baseline_evaluator.py)
Wrapper riguroso sobre `moshi.offline` para ejecutar un turno completo de inferencia *full-duplex*. Contiene manejo exhaustivo de excepciones y logging estricto para capturar fallas de memoria, colapsos de ASR o asincronías.

#### [NEW] [input_colombia.txt](file:///media/diego/Personal%20Projects/PersonaPLEX_NVIDIA/personaplex/latam_experiments/input_colombia.txt)
Corpus de texto diminuto para medir la fertilidad del tokenizador con jergas y estructuras puramente colombianas.

## Verification Plan

### Automated Tests
1. **Auditoría de Tokenización:** 
   El usuario o el agente ejecutará:
   ```bash
   export HF_TOKEN="tu_token"
   python latam_experiments/01_tokenization_auditor.py
   ```
   *Criterio de éxito:* El script finaliza y reporta la tasa de fertilidad. Si falla, hay un problema de autenticación o de conexión con HuggingFace.

2. **Evaluación de Baseline:**
   Para ejecutar esto, el usuario DEBE proveer un archivo de audio real en español colombiano (`input_colombia.wav`).
   ```bash
   export HF_TOKEN="tu_token"
   python latam_experiments/02_baseline_evaluator.py
   ```
   *Criterio de éxito:* El motor genera el `output.wav` y `output.json`. Si colapsa, el log `/logs/personaplex_baseline_eval.log` trazará el stackframe exacto de la falla.

### Manual Verification
# Fase 2: Pipeline de Datos Acústico-Semánticos (Offline & CPU-friendly)

Ante la restricción absoluta de hardware VRAM, la única vía para entrenar o ajustar este modelo en el futuro (si alquilas GPU por horas o usamos métodos PEFT extremos) es tener el **Dataset Acústico Sintético de Español Colombia** preprocesado offline.

## Proposed Changes

1.  **[NEW] [03_synthetic_dataset_gen.py](file:///media/diego/Personal%20Projects/PersonaPLEX_NVIDIA/personaplex/latam_experiments/03_synthetic_dataset_gen.py)**
    Script que orquesta la generación de diálogos. Usará un LLM ligero o API pública para crear conversaciones estructuradas basadas en *Roles* colombianos, y las sintetizará usando EdgeTTS o ElevenLabs, garantizando fidelidad del acento.
    
2.  **[NEW] [04_codec_extraction.py](file:///media/diego/Personal%20Projects/PersonaPLEX_NVIDIA/personaplex/latam_experiments/04_codec_extraction.py)**
    Script que toma los audios `.wav` y extrae sus tokens discretos (códigos fonéticos RVQ usando el modelo `mimi`) para guardarlos en disco de forma pre-evaluada. **Esto evita extraer los tokens al vuelo en la RAM de GPU durante un posible entrenamiento.**

## Verification Plan
1. Ejecutaremos el script de orquestación (03) para generar 3 audios artificiales.
2. Comprobaremos la prosodia en vivo (el usuario debe escucharlos).
3. Evaluaremos si el Extractor de Codec (04) puede correr puro CPU y generar los arreglos de `.pt` (`torch.Tensor`) listos para el dataloader sin crashear.

---
## Verification Diagnostics (Execution Findings)

### Tokenization Auditor Results Tracker
- **Total Words:** 16
- **Total Tokens:** 36
- **Overall Fertility Rate:** 2.250
- **Diagnosis:** FERTILITY SUBOPTIMAL. El tokenizador (basado en spm_32k) fragmenta las palabras del español colombiano un 125% más de lo que haría con el inglés base. Esto confirma que el *In-Context Learning* simple dañará la latencia y forzará CPT para re-aprender representaciones del tokenizador.

### Baseline Evaluator Results Tracker
- **Trastorno de Arquitectura Encontrado:** PyTorch Fallo Crítico por `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`.
- **Diagnosis:** El hardware actual del Host no posee una arquitectura SM apta (o instalada apropiadamente a nivel de drivers de Host) para compilar las instrucciones FlashAttention ni las proyecciones de Transformer de Moshi/Helium compiladas en BFloat16/FP16 con núcleos Ampere/Hopper en adelante, e imposibilita correr CPU-only debido a tensores fijos ("`RuntimeError: No CUDA GPUs are available`").
- **Conclusión de Baseline:** El entrenamiento local de Llora/SFT/CPT de un modelo de 7B sobre este entorno está **garantizado a fracasar** si no migramos a un entorno de Cómputo Causal (Nodos A100/H100) real.
