🎯 Objetivo del Proyecto
Este sistema clasifica el riesgo de hospitalización por COVID-19 en 3 niveles (Alto/Medio/Bajo) para facilitar la prevención temprana y optimizar la asignación de recursos médicos.

🏗️ Arquitectura del Sistema
El proyecto implementa una arquitectura basada en agentes con los siguientes componentes:

📁 Estructura del Proyecto
ProyectoMachineLearningCovid19/
├── 🤖 agents/                 # Agentes especializados
│   ├── data_extractor.py      # Extracción de datos
│   ├── data_preprocessor.py   # Preprocesamiento
│   ├── ml_predictor.py        # Predicción ML
│   └── dashboard_agent.py     # Dashboard inteligente
├── 🧠 models/                 # Modelos entrenados
├── 🎨 templates/              # Templates HTML
├── 📁 static/                 # Archivos estáticos
├── 📊 covid.csv              # Dataset COVID-19
├── ⚙️ config.py              # Configuración
├── 🚀 app.py                 # Aplicación principal
└── 📋 requirements.txt       # Dependencias
🔧 Tecnologías Utilizadas
Backend: Flask, Python 3.8+
Machine Learning: Scikit-learn, Pandas, NumPy
Visualización: Plotly, Matplotlib, Seaborn
IA Generativa: Google Gemini API
Balanceo de Datos: Imbalanced-learn
Frontend: HTML, CSS, JavaScript
🚀 Instalación y Configuración
1. Clonar el Repositorio
Copygit clone https://github.com/Johnkl725/ProyectoMachineLearningCovid19.git
cd ProyectoMachineLearningCovid19
2. Crear Entorno Virtual
Copypython -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
3. Instalar Dependencias
Copypip install -r requirements.txt
4. Configurar Variables de Entorno
Crea un archivo .env en la raíz del proyecto:

GEMINI_API_KEY=tu_api_key_aqui
DEBUG=True
5. Ejecutar la Aplicación
Copypython app.py
La aplicación estará disponible en http://localhost:5000

📈 Funcionalidades Principales
🔄 Pipeline de Machine Learning Completo
Fase 1: Definición del Problema (Clasificación multiclase)
Fase 2: Recolección de Datos (Dataset COVID-19)
Fase 3: Preparación de Datos (Limpieza y codificación)
Fase 4: División de Datos (Train/Test)
Fase 5: Selección de Modelos (Múltiples algoritmos)
Fase 6: Entrenamiento (Con balanceo de clases)
Fase 7: Evaluación (Métricas comprehensivas)
Fase 8: Optimización y Selección del Mejor Modelo
🎯 Clasificación de Riesgo
El sistema clasifica el riesgo en 3 niveles:

🔴 Alto: Riesgo elevado de hospitalización
🟡 Medio: Riesgo moderado
🟢 Bajo: Riesgo mínimo
📊 Dashboard Inteligente
Visualizaciones interactivas con Plotly
Explicaciones generadas por IA (Gemini)
Métricas de rendimiento del modelo
Análisis de importancia de características
🛠️ API Endpoints
Endpoint	Método	Descripción
/	GET	Página principal
/train_model	POST	Entrenar modelo completo
/predict	POST	Realizar predicción
/load_model	POST	Cargar modelo preentrenado
/model_status	GET	Estado del modelo
/data_info	GET	Información del dataset
📊 Métricas de Evaluación
El sistema evalúa los modelos usando:

Accuracy: Precisión general
Recall: Sensibilidad por clase
Precision: Precisión por clase
F1-Score: Media armónica
Matriz de Confusión: Análisis detallado
🤖 Agentes Especializados
DataExtractor
Carga y validación de datos
Análisis de calidad del dataset
Información de variables objetivo
DataPreprocessor
Limpieza de datos
Codificación de variables categóricas
Escalado de características
Balanceo de clases
MLPredictor
Selección de múltiples modelos
Entrenamiento optimizado
Evaluación comprehensiva
Persistencia de modelos
DashboardAgent
Generación de dashboards inteligentes
Explicaciones con IA generativa
Visualizaciones interactivas
🔒 Seguridad y Consideraciones
API Key: Configurar correctamente la clave de Gemini
CORS: Configurado para desarrollo
Logging: Sistema de logs comprehensivo
Validación: Validación de entrada de datos
🚀 Próximas Mejoras
 Implementar autenticación de usuarios
 Agregar más algoritmos de ML
 Integración con bases de datos
 API REST documentada con Swagger
 Deployment en contenedores Docker
 Tests unitarios y de integración
🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor:

Fork el proyecto
Crea una rama para tu feature (git checkout -b feature/AmazingFeature)
Commit tus cambios (git commit -m 'Add some AmazingFeature')
Push a la rama (git push origin feature/AmazingFeature)
Abre un Pull Request
📄 Licencia
Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

👨‍💻 Autor
Johnkl725 - GitHub Profile

🙏 Reconocimientos
Dataset COVID-19 de fuentes públicas
Comunidad de Python y Machine Learning
Google Gemini API para IA generativa
📱 Capturas de Pantalla
Interfaz Principal
La aplicación cuenta con una interfaz web intuitiva para:

Entrada de datos del paciente
Entrenamiento del modelo
Visualización de predicciones
Dashboard de métricas
Lenguajes del Proyecto
Python: 64.4% (Lógica principal y ML)
HTML: 35.6% (Templates web)
