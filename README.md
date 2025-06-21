# 🧠 Sistema Predictivo de Hospitalización por COVID-19

🎯 **Objetivo del Proyecto**  
Este sistema clasifica el riesgo de hospitalización por COVID-19 en 3 niveles (**Alto**, **Medio**, **Bajo**) para facilitar la prevención temprana y optimizar la asignación de recursos médicos.

---

## 🏗️ Arquitectura del Sistema

El proyecto implementa una **arquitectura basada en agentes** con los siguientes componentes:

```
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
```

---

## 🔧 Tecnologías Utilizadas

- **Backend**: Flask, Python 3.8+  
- **Machine Learning**: Scikit-learn, Pandas, NumPy  
- **Visualización**: Plotly, Matplotlib, Seaborn  
- **IA Generativa**: Google Gemini API  
- **Balanceo de Datos**: Imbalanced-learn  
- **Frontend**: HTML, CSS, JavaScript  

---

## 🚀 Instalación y Configuración

1. **Clonar el Repositorio**

```bash
git clone https://github.com/Johnkl725/ProyectoMachineLearningCovid19.git
cd ProyectoMachineLearningCovid19
```

2. **Crear Entorno Virtual**

```bash
python -m venv venv
# Activar entorno virtual:
source venv/bin/activate          # En Mac/Linux
venv\Scripts\activate             # En Windows
```

3. **Instalar Dependencias**

```bash
pip install -r requirements.txt
```

4. **Configurar Variables de Entorno**  
Crear un archivo `.env` en la raíz del proyecto con:

```
GEMINI_API_KEY=tu_api_key_aqui
DEBUG=True
```

5. **Ejecutar la Aplicación**

```bash
python app.py
```

La aplicación estará disponible en: [http://localhost:5000](http://localhost:5000)

---

## 📈 Funcionalidades Principales

### 🔄 Pipeline de Machine Learning

1. Definición del Problema (Clasificación multiclase)  
2. Recolección de Datos  
3. Preparación de Datos (Limpieza, codificación)  
4. División Train/Test  
5. Selección de Modelos  
6. Entrenamiento con balanceo de clases  
7. Evaluación del modelo  
8. Optimización y Persistencia  

### 🎯 Clasificación de Riesgo

- 🔴 **Alto**: Riesgo elevado de hospitalización  
- 🟡 **Medio**: Riesgo moderado  
- 🟢 **Bajo**: Riesgo mínimo  

### 📊 Dashboard Inteligente

- Visualizaciones interactivas con Plotly  
- Explicaciones generadas con IA (Gemini)  
- Métricas de rendimiento  
- Análisis de importancia de variables  

---

## 🛠️ API Endpoints

| Endpoint        | Método | Descripción                   |
|----------------|--------|-------------------------------|
| `/`            | GET    | Página principal              |
| `/train_model` | POST   | Entrenar modelo completo      |
| `/predict`     | POST   | Realizar predicción           |
| `/load_model`  | POST   | Cargar modelo preentrenado    |
| `/model_status`| GET    | Estado del modelo             |
| `/data_info`   | GET    | Información del dataset       |

---

## 📊 Métricas de Evaluación

- **Accuracy**  
- **Recall** (Sensibilidad por clase)  
- **Precision**  
- **F1-Score**  
- **Matriz de Confusión**  

---

## 🤖 Agentes Especializados

### `DataExtractor`
- Carga y validación de datos  
- Análisis de calidad del dataset  

### `DataPreprocessor`
- Limpieza y codificación  
- Escalado y balanceo de clases  

### `MLPredictor`
- Entrenamiento y evaluación  
- Persistencia del modelo  

### `DashboardAgent`
- Generación de dashboards  
- Explicaciones con IA generativa  

---

## 🔒 Seguridad y Consideraciones

- API Key segura para Gemini  
- Validación de entradas  
- Logging avanzado  
- CORS habilitado en desarrollo  

---

## 🚀 Próximas Mejoras

- Autenticación de usuarios  
- Nuevos algoritmos de ML  
- Integración con bases de datos  
- Documentación REST con Swagger  
- Dockerización  
- Tests unitarios y de integración  

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!

```bash
# Pasos sugeridos:
1. Fork el proyecto
2. git checkout -b feature/NuevaFeature
3. git commit -m 'Agregar nueva funcionalidad'
4. git push origin feature/NuevaFeature
5. Abrir un Pull Request
```

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver archivo [LICENSE](LICENSE).

---

## 👨‍💻 Autor

**Johnkl725** – [GitHub Profile](https://github.com/Johnkl725)

---

## 🙏 Reconocimientos

- Dataset COVID-19 de fuentes públicas  
- Comunidad de Python y Machine Learning  
- Google Gemini API para IA generativa  

---

## 📱 Capturas de Pantalla

- Entrada de datos del paciente  
- Entrenamiento del modelo  
- Predicciones con IA  
- Dashboard interactivo

---

## 📊 Lenguajes del Proyecto

```text
Python: 64.4% (Lógica y ML)
HTML:   35.6% (Frontend web)
```
