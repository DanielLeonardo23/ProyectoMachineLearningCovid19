"""
Aplicación Principal - Sistema Predictivo COVID-19
Integra todos los agentes y proporciona la interfaz web
Enfocado en prevención temprana y clasificación de riesgo de hospitalización
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Importar agentes
from agents.data_extractor import DataExtractor
from agents.data_preprocessor import DataPreprocessor
from agents.ml_predictor import MLPredictor
from agents.dashboard_agent import IntelligentDashboardAgent
from config import GEMINI_API_KEY, MODEL_PATH, SCALER_PATH

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Inicializar agentes
data_extractor = DataExtractor()
data_preprocessor = DataPreprocessor()
ml_predictor = MLPredictor()
dashboard_agent = IntelligentDashboardAgent(GEMINI_API_KEY)

# Variables globales para el modelo
model_trained = False
model_metrics = {}

# Variables globales para almacenar la última predicción y datos del paciente
last_patient_data = None
last_risk_prediction = None
last_dashboard_html = None
last_model_metrics = None

@app.route('/')
def index():
    """Página principal con formulario de entrada"""
    return render_template('index.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Entrena el modelo completo para clasificación de riesgo en 3 niveles"""
    global model_trained, model_metrics
    
    try:
        logger.info("Iniciando entrenamiento del modelo...")
        
        # Fase 1: Definición del Problema
        logger.info("Fase 1: Definición del Problema")
        problem_definition = {
            "task": "clasificación_multiclase",
            "target_variable": "covid_res",
            "objective": "Clasificar el riesgo de hospitalización por COVID-19 en 3 niveles (Alto/Medio/Bajo) para prevención temprana",
            "risk_levels": {1: 'Alto', 2: 'Medio', 3: 'Bajo'},
            "metrics": ["accuracy", "recall", "precision", "f1_score"]
        }
        
        # Fase 2: Recolección de Datos
        logger.info("Fase 2: Recolección de Datos")
        raw_data = data_extractor.load_data()
        data_info = data_extractor.get_data_info()
        
        # Fase 3: Preparación de Datos
        logger.info("Fase 3: Preparación de Datos")
        cleaned_data = data_preprocessor.clean_data(raw_data)
        encoded_data = data_preprocessor.encode_categorical_variables(cleaned_data)
        X, y = data_preprocessor.prepare_features(encoded_data)
        
        # Balancear clases si es necesario
        X_balanced, y_balanced = data_preprocessor.handle_class_imbalance(X, y)
        
        # Escalar características
        X_scaled = data_preprocessor.scale_features(X_balanced)
        
        # Guardar artefactos de preprocesamiento
        data_preprocessor.save_preprocessing_artifacts()
        
        # Fase 4: División de Datos
        logger.info("Fase 4: División de Datos")
        X_train, X_test, y_train, y_test = ml_predictor.split_data(X_scaled, y_balanced)
        
        # Fase 5: Selección de Modelos
        logger.info("Fase 5: Selección de Modelos")
        ml_predictor.select_models()
        
        # Fase 6: Entrenamiento
        logger.info("Fase 6: Entrenamiento")
        ml_predictor.train_models(X_train, y_train)
        
        # Fase 7: Evaluación
        logger.info("Fase 7: Evaluación")
        evaluation_results = ml_predictor.evaluate_models(X_test, y_test)
        
        # Fase 8: Optimización y Selección del Mejor Modelo
        logger.info("Fase 8: Optimización")
        best_model_name = ml_predictor.select_best_model()
        feature_importance = ml_predictor.get_feature_importance(X_test)
        
        # Guardar modelo
        ml_predictor.save_model()
        
        # Crear gráficos de evaluación
        ml_predictor.create_evaluation_plots(X_test, y_test)
        
        # Actualizar estado global
        model_trained = True
        model_metrics = ml_predictor.model_metrics[best_model_name]
        
        # Preparar métricas para JSON
        metrics_for_json = {}
        for key, value in model_metrics.items():
            if isinstance(value, np.ndarray):
                metrics_for_json[key] = value.tolist()
            elif isinstance(value, np.integer):
                metrics_for_json[key] = int(value)
            elif isinstance(value, np.floating):
                metrics_for_json[key] = float(value)
            else:
                metrics_for_json[key] = value
        
        return jsonify({
            "success": True,
            "message": "Modelo entrenado exitosamente para clasificación de riesgo en 3 niveles",
            "best_model": best_model_name,
            "metrics": metrics_for_json,
            "data_info": data_info,
            "risk_levels": ml_predictor.covid_classes,
            "focus": "Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización"
        })
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        return jsonify({
            "success": False,
            "message": f"Error en el entrenamiento: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Realiza predicción de riesgo con datos del formulario"""
    global model_trained, model_metrics, last_patient_data, last_risk_prediction, last_dashboard_html, last_model_metrics
    
    if not model_trained:
        return jsonify({
            "success": False,
            "message": "El modelo no está entrenado. Por favor, entrena el modelo primero."
        }), 400
    
    try:
        # Obtener datos del formulario
        form_data = request.form.to_dict()
        
        # Convertir datos a formato numérico
        patient_data = {}
        for key, value in form_data.items():
            if value == '':
                patient_data[key] = 0
            elif value.lower() in ['si', 'yes', '1', 'true']:
                patient_data[key] = 1
            elif value.lower() in ['no', '0', 'false']:
                patient_data[key] = 0
            else:
                try:
                    patient_data[key] = int(value)
                except:
                    patient_data[key] = 0
        
        # Crear DataFrame con los datos del paciente
        patient_df = pd.DataFrame([patient_data])
        
        # Transformar datos usando el preprocesador
        X_transformed = data_preprocessor.transform_new_data(patient_df)
        
        # Realizar predicción de riesgo
        risk_prediction = ml_predictor.predict_risk(X_transformed)
        
        # Crear dashboard con explicaciones
        dashboard_html = dashboard_agent.create_prediction_dashboard(
            patient_data, risk_prediction, model_metrics
        )
        
        # Guardar la última predicción y datos
        last_patient_data = patient_data
        last_risk_prediction = risk_prediction
        last_dashboard_html = dashboard_html
        last_model_metrics = model_metrics
        
        # Preparar métricas para JSON
        metrics_for_json = {}
        if model_metrics:
            for key, value in model_metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_for_json[key] = value.tolist()
                elif isinstance(value, np.integer):
                    metrics_for_json[key] = int(value)
                elif isinstance(value, np.floating):
                    metrics_for_json[key] = float(value)
                else:
                    metrics_for_json[key] = value
        
        # Preparar risk_prediction para JSON
        risk_prediction_for_json = {}
        for key, value in risk_prediction.items():
            if isinstance(value, np.ndarray):
                risk_prediction_for_json[key] = value.tolist()
            elif isinstance(value, np.integer):
                risk_prediction_for_json[key] = int(value)
            elif isinstance(value, np.floating):
                risk_prediction_for_json[key] = float(value)
            else:
                risk_prediction_for_json[key] = value
        
        return jsonify({
            "success": True,
            "risk_prediction": risk_prediction_for_json,
            "dashboard": dashboard_html,
            "focus": "Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización"
        })
        
    except Exception as e:
        logger.error(f"Error en la predicción: {e}")
        return jsonify({
            "success": False,
            "message": f"Error en la predicción: {str(e)}"
        }), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Carga un modelo pre-entrenado"""
    global model_trained, model_metrics
    
    try:
        # Cargar artefactos de preprocesamiento
        data_preprocessor.load_preprocessing_artifacts()
        
        # Cargar modelo
        ml_predictor.load_model()
        
        if ml_predictor.best_model is not None:
            model_trained = True
            model_metrics = ml_predictor.model_metrics.get(ml_predictor.best_model_name, {})
            
            return jsonify({
                "success": True,
                "message": "Modelo pre-entrenado cargado exitosamente",
                "model_name": ml_predictor.best_model_name,
                "risk_levels": ml_predictor.covid_classes,
                "focus": "Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización"
            })
        else:
            return jsonify({
                "success": False,
                "message": "No se pudo cargar el modelo"
            }), 400
            
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        return jsonify({
            "success": False,
            "message": f"Error cargando modelo: {str(e)}"
        }), 500

@app.route('/model_status')
def model_status():
    """Retorna el estado actual del modelo"""
    # Preparar métricas para JSON
    metrics_for_json = {}
    if model_trained and model_metrics:
        for key, value in model_metrics.items():
            if isinstance(value, np.ndarray):
                metrics_for_json[key] = value.tolist()
            elif isinstance(value, np.integer):
                metrics_for_json[key] = int(value)
            elif isinstance(value, np.floating):
                metrics_for_json[key] = float(value)
            else:
                metrics_for_json[key] = value
    
    return jsonify({
        "trained": model_trained,
        "model_name": ml_predictor.best_model_name if model_trained else None,
        "metrics": metrics_for_json,
        "risk_levels": ml_predictor.covid_classes,
        "focus": "Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización"
    })

@app.route('/data_info')
def data_info():
    """Retorna información sobre el dataset"""
    try:
        raw_data = data_extractor.load_data()
        info = data_extractor.get_data_info()
        target_info = data_extractor.get_target_variable_info()
        quality_report = data_extractor.validate_data_quality()
        
        return jsonify({
            "dataset_info": info,
            "target_variable": target_info,
            "quality_report": quality_report,
            "focus": "Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización"
        })
    except Exception as e:
        logger.error(f"Error obteniendo información del dataset: {e}")
        return jsonify({
            "success": False,
            "message": f"Error obteniendo información del dataset: {str(e)}"
        }), 500

@app.route('/dashboard')
def dashboard():
    global last_patient_data, last_risk_prediction, last_dashboard_html, last_model_metrics
    # Si no hay predicción previa, redirigir al index
    if last_patient_data is None or last_risk_prediction is None or last_dashboard_html is None:
        return redirect(url_for('index'))
    # Renderizar la nueva plantilla dashboard.html
    return render_template('dashboard.html',
                           dashboard_html=last_dashboard_html,
                           risk_prediction=last_risk_prediction,
                           model_metrics=last_model_metrics)

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    """Ejecuta el preprocesamiento y control de calidad de los datos, devolviendo un resumen para la interfaz."""
    try:
        # Cargar datos originales
        raw_data = data_extractor.load_data()
        # Resumen de calidad antes de limpiar
        quality_before = data_extractor.validate_data_quality()
        stats_before = raw_data.describe(include='all').fillna('').to_dict()
        # Limpiar datos
        cleaned_data = data_preprocessor.clean_data(raw_data)
        # Resumen de calidad después de limpiar
        quality_after = {
            'total_rows': int(len(cleaned_data)),
            'total_columns': int(len(cleaned_data.columns)),
            'missing_data_percentage': float((cleaned_data.isnull().sum().sum() / (len(cleaned_data) * len(cleaned_data.columns))) * 100),
            'duplicate_rows': int(cleaned_data.duplicated().sum()),
            'data_types': {col: str(cleaned_data[col].dtype) for col in cleaned_data.columns},
        }
        stats_after = cleaned_data.describe(include='all').fillna('').to_dict()
        # Preview de datos
        preview_before = raw_data.head(5).fillna('').to_dict(orient='records')
        preview_after = cleaned_data.head(5).fillna('').to_dict(orient='records')
        # Top 3 columnas con más nulos antes y después
        nulls_before = raw_data.isnull().sum().sort_values(ascending=False)
        nulls_after = cleaned_data.isnull().sum().sort_values(ascending=False)
        top_nulls_before = nulls_before[nulls_before > 0].head(3).to_dict()
        top_nulls_after = nulls_after[nulls_after > 0].head(3).to_dict()
        return jsonify({
            'success': True,
            'quality_before': quality_before,
            'quality_after': quality_after,
            'stats_before': stats_before,
            'stats_after': stats_after,
            'preview_before': preview_before,
            'preview_after': preview_after,
            'top_nulls_before': top_nulls_before,
            'top_nulls_after': top_nulls_after
        })
    except Exception as e:
        logger.error(f"Error en el preprocesamiento: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Intentar cargar modelo pre-entrenado al iniciar
    try:
        data_preprocessor.load_preprocessing_artifacts()
        ml_predictor.load_model()
        if ml_predictor.best_model is not None:
            model_trained = True
            model_metrics = ml_predictor.model_metrics.get(ml_predictor.best_model_name, {})
            logger.info("Modelo pre-entrenado cargado exitosamente")
    except Exception as e:
        logger.warning(f"No se pudo cargar modelo pre-entrenado: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 