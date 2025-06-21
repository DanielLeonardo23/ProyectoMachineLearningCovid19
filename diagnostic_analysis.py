"""
Script de Diagnóstico Completo del Sistema de Predicción COVID-19
Analiza el score de riesgo, balance de clases, y rendimiento del modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging
import os

# Importar agentes
from agents.data_extractor import DataExtractor
from agents.data_preprocessor import DataPreprocessor
from agents.ml_predictor import MLPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_complete_diagnostic():
    """Ejecuta diagnóstico completo del sistema"""
    logger.info("=== INICIANDO DIAGNÓSTICO COMPLETO DEL SISTEMA ===")
    
    try:
        # 1. Cargar datos
        logger.info("1. Cargando datos...")
        extractor = DataExtractor()
        raw_data = extractor.load_data()
        logger.info(f"Datos cargados: {raw_data.shape}")
        
        # 2. Preprocesar datos
        logger.info("2. Preprocesando datos...")
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(raw_data)
        encoded_data = preprocessor.encode_categorical_variables(cleaned_data)
        
        # 3. Preparar características (esto incluye el score de riesgo)
        logger.info("3. Preparando características...")
        X, y = preprocessor.prepare_features(encoded_data)
        
        # 4. Análisis de balance de clases
        logger.info("4. Analizando balance de clases...")
        X_balanced, y_balanced = preprocessor.handle_class_imbalance(X, y)
        
        # 5. Escalar características
        logger.info("5. Escalando características...")
        X_scaled = preprocessor.scale_features(X_balanced)
        
        # 6. Entrenar y evaluar modelo
        logger.info("6. Entrenando y evaluando modelo...")
        predictor = MLPredictor()
        X_train, X_test, y_train, y_test = predictor.split_data(X_scaled, y_balanced)
        
        # Seleccionar y entrenar modelo
        models = predictor.select_models()
        trained_models = predictor.train_models(X_train, y_train)
        
        # Evaluar modelo
        evaluation_results = predictor.evaluate_models(X_test, y_test)
        
        # 7. Generar reportes visuales
        logger.info("7. Generando reportes visuales...")
        generate_diagnostic_plots(X_scaled, y_balanced, X_test, y_test, predictor)
        
        # 8. Análisis de importancia de características
        logger.info("8. Analizando importancia de características...")
        best_model_name = predictor.select_best_model()
        feature_importance = predictor.get_feature_importance(X_scaled)
        
        logger.info("=== RESUMEN DEL DIAGNÓSTICO ===")
        logger.info(f"Mejor modelo: {best_model_name}")
        logger.info(f"Accuracy: {evaluation_results[best_model_name]['accuracy']:.4f}")
        logger.info(f"F1-Score: {evaluation_results[best_model_name]['f1_score']:.4f}")
        
        # Mostrar top características importantes
        if not feature_importance.empty:
            logger.info("Top 10 características más importantes:")
            for i, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        logger.info("=== DIAGNÓSTICO COMPLETADO ===")
        
    except Exception as e:
        logger.error(f"Error en diagnóstico: {e}")
        raise

def generate_diagnostic_plots(X, y, X_test, y_test, predictor):
    """Genera gráficos de diagnóstico"""
    
    # Crear directorio para plots
    os.makedirs("static", exist_ok=True)
    
    # 1. Distribución de clases
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    class_counts = pd.Series(y).value_counts().sort_index()
    class_names = ['Alto', 'Medio', 'Bajo']
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    plt.pie(class_counts.values, labels=class_names, autopct='%1.1f%%', colors=colors)
    plt.title('Distribución de Clases de Riesgo')
    
    # 2. Distribución de scores de riesgo (si está disponible)
    if 'risk_score' in X.columns:
        plt.subplot(2, 3, 2)
        plt.hist(X['risk_score'], bins=30, alpha=0.7, color='skyblue')
        plt.title('Distribución de Scores de Riesgo')
        plt.xlabel('Score de Riesgo')
        plt.ylabel('Frecuencia')
    
    # 3. Matriz de confusión
    plt.subplot(2, 3, 3)
    if predictor.best_model is not None:
        y_pred = predictor.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
    
    # 4. Importancia de características
    plt.subplot(2, 3, 4)
    if predictor.feature_importance is not None:
        top_features = predictor.feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 10 Características Importantes')
        plt.xlabel('Importancia')
    
    # 5. Análisis de probabilidades
    plt.subplot(2, 3, 5)
    if predictor.best_model is not None:
        y_pred_proba = predictor.best_model.predict_proba(X_test)
        for i, class_name in enumerate(class_names):
            if i < y_pred_proba.shape[1]:
                plt.hist(y_pred_proba[:, i], alpha=0.5, label=class_name, bins=20)
        plt.title('Distribución de Probabilidades')
        plt.xlabel('Probabilidad')
        plt.ylabel('Frecuencia')
        plt.legend()
    
    # 6. Comparación real vs predicho
    plt.subplot(2, 3, 6)
    if predictor.best_model is not None:
        real_counts = pd.Series(y_test).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.bar(x - width/2, real_counts.values, width, label='Real', alpha=0.7)
        plt.bar(x + width/2, pred_counts.values, width, label='Predicho', alpha=0.7)
        plt.xlabel('Clase de Riesgo')
        plt.ylabel('Cantidad')
        plt.title('Real vs Predicho')
        plt.xticks(x, class_names)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('static/diagnostic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Gráficos de diagnóstico guardados en static/diagnostic_analysis.png")

def analyze_risk_score_distribution():
    """Análisis específico de la distribución del score de riesgo"""
    logger.info("=== ANÁLISIS ESPECÍFICO DEL SCORE DE RIESGO ===")
    
    # Cargar datos
    extractor = DataExtractor()
    raw_data = extractor.load_data()
    
    # Preprocesar hasta el score de riesgo
    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.clean_data(raw_data)
    encoded_data = preprocessor.encode_categorical_variables(cleaned_data)
    
    # Crear score de riesgo
    data_with_risk = preprocessor.create_risk_score(encoded_data)
    
    # Análisis detallado
    risk_scores = data_with_risk['risk_score']
    
    logger.info(f"Estadísticas del score de riesgo:")
    logger.info(f"  Mínimo: {risk_scores.min():.2f}")
    logger.info(f"  Máximo: {risk_scores.max():.2f}")
    logger.info(f"  Media: {risk_scores.mean():.2f}")
    logger.info(f"  Mediana: {risk_scores.median():.2f}")
    logger.info(f"  Desviación estándar: {risk_scores.std():.2f}")
    
    # Análisis de percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    logger.info("Percentiles:")
    for p in percentiles:
        logger.info(f"  P{p}: {risk_scores.quantile(p/100):.2f}")
    
    # Análisis de distribución
    logger.info("Análisis de distribución:")
    logger.info(f"  Asimetría: {risk_scores.skew():.3f}")
    logger.info(f"  Curtosis: {risk_scores.kurtosis():.3f}")
    
    # Sugerencias de umbrales
    logger.info("Sugerencias de umbrales:")
    
    # Opción 1: Percentiles
    p25 = risk_scores.quantile(0.25)
    p75 = risk_scores.quantile(0.75)
    logger.info(f"  Opción 1 (Percentiles): Bajo < {p25:.2f}, Alto >= {p75:.2f}")
    
    # Opción 2: Media ± desviación estándar
    mean_score = risk_scores.mean()
    std_score = risk_scores.std()
    low_thresh1 = mean_score - 0.5 * std_score
    high_thresh1 = mean_score + 0.5 * std_score
    logger.info(f"  Opción 2 (Media±0.5*std): Bajo < {low_thresh1:.2f}, Alto >= {high_thresh1:.2f}")
    
    # Opción 3: Media ± desviación estándar (más conservador)
    low_thresh2 = mean_score - std_score
    high_thresh2 = mean_score + std_score
    logger.info(f"  Opción 3 (Media±1*std): Bajo < {low_thresh2:.2f}, Alto >= {high_thresh2:.2f}")
    
    # Mostrar distribución resultante para cada opción
    logger.info("Distribución resultante para cada opción:")
    
    for i, (low, high, name) in enumerate([
        (p25, p75, "Percentiles"),
        (low_thresh1, high_thresh1, "Media±0.5*std"),
        (low_thresh2, high_thresh2, "Media±1*std")
    ]):
        bajo = np.sum(risk_scores < low)
        medio = np.sum((risk_scores >= low) & (risk_scores < high))
        alto = np.sum(risk_scores >= high)
        total = len(risk_scores)
        
        logger.info(f"  {name}:")
        logger.info(f"    Bajo: {bajo} ({bajo/total*100:.1f}%)")
        logger.info(f"    Medio: {medio} ({medio/total*100:.1f}%)")
        logger.info(f"    Alto: {alto} ({alto/total*100:.1f}%)")

if __name__ == "__main__":
    # Ejecutar análisis específico del score de riesgo
    analyze_risk_score_distribution()
    
    # Ejecutar diagnóstico completo
    run_complete_diagnostic() 