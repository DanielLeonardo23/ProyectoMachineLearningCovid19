"""
Agente 3: Predicción (Modelo ML)
Responsable del entrenamiento, evaluación y predicción del modelo
Implementa las fases 4-8 del ciclo de vida del aprendizaje supervisado
Enfocado en clasificación de riesgo de hospitalización en 3 niveles
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Tuple, List
import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    """Agente responsable del entrenamiento y predicción del modelo"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_metrics = {}
        self.feature_importance = None
        self.covid_classes = {1: 'Positivo', 2: 'Negativo'}
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Fase 4: División del Conjunto de Datos
        Divide los datos en conjuntos de entrenamiento y prueba
        """
        logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Conjunto de entrenamiento: {X_train.shape}")
        logger.info(f"Conjunto de prueba: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def select_models(self) -> Dict[str, Any]:
        """
        Fase 5: Selección del Modelo
        Define modelos para entrenamiento incluyendo Random Forest
        """
        logger.info("Seleccionando modelos para entrenamiento...")
        
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            }
        }
        
        self.models = models
        logger.info(f"Modelos seleccionados: {list(models.keys())}")
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Fase 6: Entrenamiento del Modelo
        Entrena todos los modelos con validación cruzada
        """
        logger.info("Iniciando entrenamiento de modelos...")
        
        trained_models = {}
        
        for name, model_info in self.models.items():
            logger.info(f"Entrenando modelo: {name}")
            
            # Crear pipeline con escalado
            pipeline = Pipeline([
                ('scaler', None),  # Los datos ya están escalados
                ('classifier', model_info['model'])
            ])
            
            # Grid Search con validación cruzada
            grid_search = GridSearchCV(
                pipeline,
                {'classifier__' + k: v for k, v in model_info['params'].items()},
                cv=5,
                scoring='f1_weighted',  # Cambiar a f1_weighted para multiclase
                n_jobs=-1,
                verbose=0
            )
            
            # Entrenar modelo
            grid_search.fit(X_train, y_train)
            
            trained_models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            logger.info(f"Mejor score para {name}: {grid_search.best_score_:.4f}")
        
        self.trained_models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Fase 7: Evaluación del Modelo
        Evalúa todos los modelos entrenados para clasificación binaria COVID
        """
        logger.info("Evaluando modelos...")
        
        evaluation_results = {}
        
        for name, model_info in self.trained_models.items():
            model = model_info['model']
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # DIAGNÓSTICO: Análisis detallado de predicciones
            logger.info(f"=== DIAGNÓSTICO DEL MODELO {name.upper()} ===")
            
            # Distribución de clases reales
            real_counts = y_test.value_counts().sort_index()
            logger.info("Distribución de clases reales:")
            for class_id, count in real_counts.items():
                class_name = self.covid_classes.get(int(class_id), f"Clase {class_id}")
                percentage = count / len(y_test) * 100
                logger.info(f"  {class_name}: {count} casos ({percentage:.1f}%)")
            
            # Distribución de predicciones
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            logger.info("Distribución de predicciones:")
            for class_id, count in pred_counts.items():
                class_name = self.covid_classes.get(int(class_id), f"Clase {class_id}")
                percentage = count / len(y_pred) * 100
                logger.info(f"  {class_name}: {count} casos ({percentage:.1f}%)")
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            logger.info("Matriz de confusión:")
            logger.info("  Predicción →")
            logger.info("Real ↓")
            
            class_names = ['Positivo', 'Negativo']
            for i, real_class in enumerate(class_names):
                row_str = f"  {real_class:8}"
                for j, pred_class in enumerate(class_names):
                    row_str += f" {cm[i][j]:6}"
                logger.info(row_str)
            
            # Análisis de errores por clase
            logger.info("Análisis de errores por clase:")
            for i, class_name in enumerate(class_names):
                if i < len(cm):
                    total_real = cm[i].sum()
                    correct = cm[i][i]
                    errors = total_real - correct
                    error_rate = errors / total_real * 100 if total_real > 0 else 0
                    logger.info(f"  {class_name}: {correct}/{total_real} correctos ({error_rate:.1f}% error)")
            
            # Métricas para clasificación binaria
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
                'classification_report': classification_report(y_test, y_pred, target_names=['Positivo', 'Negativo']),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            evaluation_results[name] = metrics
            logger.info(f"Métricas para {name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Análisis de probabilidades
            logger.info("Análisis de probabilidades:")
            for i, class_name in enumerate(class_names):
                if i < y_pred_proba.shape[1]:
                    class_probs = y_pred_proba[:, i]
                    logger.info(f"  {class_name}: min={class_probs.min():.3f}, max={class_probs.max():.3f}, mean={class_probs.mean():.3f}")
        
        self.model_metrics = evaluation_results
        return evaluation_results
    
    def select_best_model(self, metric: str = 'f1_score') -> str:
        """Selecciona el mejor modelo basado en una métrica específica"""
        logger.info(f"Seleccionando mejor modelo basado en {metric}...")
        
        best_score = -1
        best_model_name = None
        
        for name, metrics in self.model_metrics.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = name
        
        self.best_model = self.trained_models[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Mejor modelo seleccionado: {best_model_name} ({metric}: {best_score:.4f})")
        return best_model_name
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Obtiene la importancia de las características del mejor modelo"""
        if self.best_model is None:
            raise ValueError("Debe seleccionar el mejor modelo primero")
        
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            # Para Random Forest y Decision Tree
            importance = self.best_model.named_steps['classifier'].feature_importances_
        elif hasattr(self.best_model.named_steps['classifier'], 'coef_'):
            # Para Logistic Regression (promedio de coeficientes para multiclase)
            coef = self.best_model.named_steps['classifier'].coef_
            importance = np.mean(np.abs(coef), axis=0)
        else:
            return pd.DataFrame()
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones con el mejor modelo
        Retorna: (predicciones, probabilidades)
        """
        if self.best_model is None:
            raise ValueError("Debe entrenar y seleccionar el mejor modelo primero")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_risk(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Predice si el paciente es COVID positivo o negativo
        Retorna: diccionario con predicción y probabilidades
        """
        if self.best_model is None:
            raise ValueError("Debe entrenar y seleccionar el mejor modelo primero")
        
        # Realizar predicción
        predictions, probabilities = self.predict(X)
        
        # Para clasificación binaria, tomar la primera predicción
        risk_level = int(predictions[0])
        risk_probabilities = probabilities[0]
        
        # Mapear nivel numérico a texto
        risk_level_text = self.covid_classes.get(risk_level, 'Desconocido')
        
        # Obtener probabilidad del nivel predicho
        risk_probability = risk_probabilities[risk_level - 1]  # Ajustar índice
        
        # Crear resultado
        result = {
            'risk_level': risk_level,
            'risk_level_text': risk_level_text,
            'probability': risk_probability,
            'all_probabilities': {
                'Positivo': risk_probabilities[0],
                'Negativo': risk_probabilities[1]
            },
            'recommendation': self._get_recommendation(risk_level, risk_probability)
        }
        
        return result
    
    def _get_recommendation(self, covid_result: int, probability: float) -> str:
        """Genera recomendación basada en el resultado de COVID"""
        if covid_result == 1:  # Positivo
            if probability >= 0.9:
                return "ALTO RIESGO: Resultado COVID POSITIVO con alta confianza. Aislamiento inmediato y seguimiento médico urgente."
            elif probability >= 0.7:
                return "PROBABLE POSITIVO: Resultado COVID POSITIVO. Aislamiento preventivo y confirmación médica."
            else:
                return "POSITIVO: Resultado COVID POSITIVO. Seguir protocolos de aislamiento y monitoreo."
        else:  # Negativo
            if probability >= 0.9:
                return "NEGATIVO: Resultado COVID NEGATIVO con alta confianza. Mantener precauciones básicas."
            elif probability >= 0.7:
                return "PROBABLE NEGATIVO: Resultado COVID NEGATIVO. Mantener vigilancia de síntomas."
            else:
                return "NEGATIVO: Resultado COVID NEGATIVO. Seguir recomendaciones de prevención."
    
    def save_model(self, output_dir: str = "models"):
        """Guarda el mejor modelo entrenado"""
        if self.best_model is None:
            raise ValueError("Debe entrenar y seleccionar el mejor modelo primero")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Guardar modelo
        model_path = os.path.join(output_dir, 'covid_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Preparar métricas para serialización JSON
        metrics_for_json = {}
        if self.best_model_name and self.best_model_name in self.model_metrics:
            original_metrics = self.model_metrics[self.best_model_name]
            for key, value in original_metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_for_json[key] = value.tolist()
                elif isinstance(value, np.integer):
                    metrics_for_json[key] = int(value)
                elif isinstance(value, np.floating):
                    metrics_for_json[key] = float(value)
                else:
                    metrics_for_json[key] = value
        
        # Guardar información adicional
        model_info = {
            'model_name': self.best_model_name,
            'covid_classes': self.covid_classes,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'metrics': metrics_for_json,
            'focus': 'Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización'
        }
        
        info_path = os.path.join(output_dir, 'model_info.pkl')
        joblib.dump(model_info, info_path)
        
        logger.info(f"Modelo guardado en {model_path}")
    
    def load_model(self, input_dir: str = "models"):
        """Carga el modelo guardado"""
        try:
            model_path = os.path.join(input_dir, 'covid_model.pkl')
            self.best_model = joblib.load(model_path)
            
            # Cargar información adicional
            info_path = os.path.join(input_dir, 'model_info.pkl')
            if os.path.exists(info_path):
                model_info = joblib.load(info_path)
                self.best_model_name = model_info.get('model_name')
                self.covid_classes = model_info.get('covid_classes', self.covid_classes)
                self.feature_importance = pd.DataFrame(model_info.get('feature_importance', []))
                self.model_metrics = {self.best_model_name: model_info.get('metrics', {})}
            
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def create_evaluation_plots(self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "static"):
        """Crea gráficos de evaluación del modelo"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Matriz de confusión
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Positivo', 'Negativo'],
                   yticklabels=['Positivo', 'Negativo'])
        plt.title('Matriz de Confusión - Clasificación de Riesgo')
        plt.ylabel('Valor Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Importancia de características
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importancia')
            plt.title('Importancia de Características - Prevención Temprana')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Gráficos de evaluación creados")

if __name__ == "__main__":
    # Ejemplo de uso
    predictor = MLPredictor()
    print("Agente de predicción inicializado")
    print(f"Niveles de riesgo: {predictor.covid_classes}") 