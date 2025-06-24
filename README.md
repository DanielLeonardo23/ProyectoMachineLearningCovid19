# Rama `tests_agents` – Dashboard Inteligente COVID-19

Este proyecto implementa un **sistema predictivo para la detección temprana de COVID-19** con un dashboard inteligente que explica el riesgo clínico y las recomendaciones personalizadas para cada paciente.

## ¿Qué contiene esta rama?

- **Agentes modulares** en `agents/`:
  - `IntelligentDashboardAgent`: Genera explicaciones inteligentes y personalizadas usando IA generativa (Gemini AI), enfocadas en el riesgo clínico y complicaciones.
  - `data_extractor`, `data_preprocessor`, `ml_predictor`: Encargados de extracción, preprocesamiento y predicción con modelos de aprendizaje supervisado (Random Forest).
- **Mejoras en el dashboard**:
  - Nueva vista `/dashboard` con gráficas de evaluación del modelo, explicación inteligente y métricas/resultados del paciente más reciente.
  - Explicaciones enfocadas en el riesgo de complicaciones si el paciente es positivo, y en prevención si es negativo.
- **Pruebas unitarias** para todos los agentes (ver carpeta `tests/`).

## ¿Cómo usar el dashboard inteligente?

1. **Instala los requisitos:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Ejecuta la aplicación:**
   ```sh
   python app.py
   ```
3. **Accede a la app:**
   - Página principal: [http://localhost:5000/](http://localhost:5000/)
   - Dashboard inteligente: [http://localhost:5000/dashboard](http://localhost:5000/dashboard)

4. **Flujo recomendado:**
   - Ingresa los datos del paciente y haz una predicción.
   - Luego visita `/dashboard` para ver todas las gráficas, métricas y la explicación personalizada.

## Características principales

- **Explicación médica avanzada:**
  - El dashboard explica el riesgo clínico, factores de riesgo, recomendaciones y señales de alarma.
  - Enfocado en prevención de complicaciones y hospitalización.
- **Visualización clara:**
  - Gráficas de matriz de confusión y feature importance.
  - Métricas del modelo y del paciente más reciente.
- **Pruebas unitarias:**
  - Scripts de test para cada agente y runner general.

## Créditos
- Desarrollado por John (Universidad Nacional Mayor de San Marcos)
- IA generativa: Google Gemini

---

¿Dudas o sugerencias? ¡Contribuye o abre un issue en el repo! 