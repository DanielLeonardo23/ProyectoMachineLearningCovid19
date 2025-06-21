"""
Agente 2: Preprocesamiento de Datos
Responsable de limpiar, transformar y preparar los datos para el modelado
Fase 3: Preparación de los Datos
Enfocado en prevención temprana y clasificación de riesgo de hospitalización
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Any
import logging
import joblib
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Agente responsable del preprocesamiento y limpieza de datos"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.processed_data = None
        self.feature_columns = None
        self.target_column = 'covid_res'
        self.risk_levels = {1: 'Alto', 2: 'Medio', 3: 'Bajo'}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza básica de datos
        - Eliminar duplicados
        - Manejar valores faltantes
        - Corregir tipos de datos
        """
        logger.info("Iniciando limpieza de datos...")
        
        # Crear copia para no modificar los datos originales
        cleaned_data = data.copy()
        
        # Eliminar duplicados
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        logger.info(f"Duplicados eliminados: {initial_rows - len(cleaned_data)}")
        
        # Manejar valores especiales (97, 99, 9999-99-99)
        cleaned_data = self._handle_special_values(cleaned_data)
        
        # Convertir columnas de fecha
        cleaned_data = self._convert_date_columns(cleaned_data)
        
        # Imputar valores faltantes
        cleaned_data = self._impute_missing_values(cleaned_data)
        
        logger.info("Limpieza de datos completada")
        return cleaned_data
    
    def _handle_special_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Maneja valores especiales como 97, 99, 9999-99-99"""
        # Reemplazar valores especiales con NaN
        special_values = [97, 99, '9999-99-99']
        
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].replace(special_values, np.nan)
            elif data[col].dtype == 'object':
                data[col] = data[col].replace(special_values, np.nan)
        
        return data
    
    def _convert_date_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convierte columnas de fecha a formato datetime"""
        date_columns = ['entry_date', 'date_symptoms']
        
        for col in date_columns:
            if col in data.columns:
                try:
                    # Especificar formato de fecha para evitar warning
                    data[col] = pd.to_datetime(data[col], format='%d-%m-%Y', errors='coerce')
                    # Extraer características de fecha
                    data[f'{col}_day'] = data[col].dt.day
                    data[f'{col}_month'] = data[col].dt.month
                    data[f'{col}_year'] = data[col].dt.year
                    # Eliminar columna original
                    data = data.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"No se pudo convertir la columna {col}: {e}")
                    # Si falla la conversión, eliminar la columna
                    if col in data.columns:
                        data = data.drop(columns=[col])
        
        return data
    
    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Imputa valores faltantes"""
        # Para variables categóricas, usar moda
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_value = data[col].mode()[0]
                data[col] = data[col].fillna(mode_value)
        
        # Para variables numéricas, usar mediana
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
        
        return data
    
    def create_risk_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crea un score de riesgo basado en síntomas y factores de riesgo
        para clasificación en 3 niveles: Alto, Medio, Bajo
        """
        logger.info("Creando score de riesgo para clasificación en 3 niveles...")
        
        risk_data = data.copy()
        
        # Definir factores de riesgo y síntomas importantes
        risk_factors = {
            'high_risk': [
                'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension', 
                'other_disease', 'cardiovascular', 'obesity', 'renal_chronic',
                'tobacco', 'pregnancy'
            ],
            'symptoms': [
                'fever', 'cough', 'fatigue', 'headache', 'sore_throat',
                'difficulty_breathing', 'chest_pain', 'loss_taste_smell'
            ],
            'demographics': ['age', 'sex']
        }
        
        # Calcular score de riesgo con pesos mejorados
        risk_score = np.zeros(len(risk_data))
        
        # Factores de riesgo (peso alto - 5 puntos cada uno)
        for factor in risk_factors['high_risk']:
            if factor in risk_data.columns:
                factor_values = risk_data[factor].astype(float).values
                risk_score += factor_values * 5
        
        # Síntomas (peso medio - 3 puntos cada uno)
        for symptom in risk_factors['symptoms']:
            if symptom in risk_data.columns:
                symptom_values = risk_data[symptom].astype(float).values
                risk_score += symptom_values * 3
        
        # Edad (factor de riesgo importante con lógica mejorada)
        if 'age' in risk_data.columns:
            age_values = risk_data['age'].astype(float).values
            # Lógica de edad más granular
            age_risk = np.where(
                (age_values < 18) | (age_values > 75),
                4,  # Muy alto riesgo para edades extremas
                np.where(
                    (age_values >= 65) & (age_values <= 75),
                    3,  # Alto riesgo para adultos mayores
                    np.where(
                        (age_values >= 50) & (age_values < 65),
                        2,  # Riesgo medio-alto
                        np.where(
                            (age_values >= 30) & (age_values < 50),
                            1,  # Riesgo bajo-medio
                            0   # Riesgo bajo para jóvenes
                        )
                    )
                )
            )
            risk_score += age_risk
        
        # Clasificar en 3 niveles basado en el score
        risk_data['risk_score'] = risk_score
        
        # DIAGNÓSTICO: Analizar distribución del score de riesgo
        logger.info("=== DIAGNÓSTICO DE SCORE DE RIESGO ===")
        logger.info(f"Estadísticas del score de riesgo:")
        logger.info(f"  Mínimo: {risk_score.min():.2f}")
        logger.info(f"  Máximo: {risk_score.max():.2f}")
        logger.info(f"  Media: {risk_score.mean():.2f}")
        logger.info(f"  Mediana: {np.median(risk_score):.2f}")
        logger.info(f"  Desviación estándar: {risk_score.std():.2f}")
        
        # Mostrar percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            logger.info(f"  Percentil {p}: {np.percentile(risk_score, p):.2f}")
        
        # Mostrar distribución de frecuencias
        unique_scores, counts = np.unique(risk_score, return_counts=True)
        logger.info(f"Valores únicos de score: {len(unique_scores)}")
        logger.info(f"Top 10 scores más frecuentes:")
        score_freq = list(zip(unique_scores, counts))
        score_freq.sort(key=lambda x: x[1], reverse=True)
        for score, count in score_freq[:10]:
            logger.info(f"  Score {score}: {count} casos ({count/len(risk_score)*100:.1f}%)")
        
        # UMBRALES FIJOS POR PERCENTILES
        low_threshold = np.percentile(risk_score, 25)
        high_threshold = np.percentile(risk_score, 75)
        logger.info(f"=== UMBRALES DE CLASIFICACIÓN (FIJOS 25/75) ===")
        logger.info(f"  Bajo riesgo (< {low_threshold:.2f}): {np.sum(risk_score < low_threshold)} casos")
        logger.info(f"  Medio riesgo ({low_threshold:.2f} - {high_threshold:.2f}): {np.sum((risk_score >= low_threshold) & (risk_score < high_threshold))} casos")
        logger.info(f"  Alto riesgo (>= {high_threshold:.2f}): {np.sum(risk_score >= high_threshold)} casos")
        
        # Clasificar riesgo
        risk_data['risk_level'] = np.where(
            risk_data['risk_score'] >= high_threshold,
            1,  # Alto riesgo
            np.where(
                risk_data['risk_score'] >= low_threshold,
                2,  # Riesgo medio
                3   # Riesgo bajo
            )
        )
        
        # Actualizar variable objetivo para clasificación de riesgo
        risk_data['covid_res'] = risk_data['risk_level']
        
        # DIAGNÓSTICO FINAL
        final_distribution = risk_data['risk_level'].value_counts().sort_index()
        logger.info(f"=== DISTRIBUCIÓN FINAL DE RIESGO ===")
        for level, count in final_distribution.items():
            level_name = self.risk_levels.get(int(level), f"Nivel {level}")
            percentage = count / len(risk_data) * 100
            logger.info(f"  {level_name}: {count} casos ({percentage:.1f}%)")
        
        logger.info(f"Umbrales finales - Bajo: {low_threshold:.2f}, Alto: {high_threshold:.2f}")
        
        return risk_data
    
    def encode_categorical_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Codifica variables categóricas usando Label Encoding"""
        logger.info("Codificando variables categóricas...")
        
        try:
            encoded_data = data.copy()
            categorical_cols = encoded_data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col != self.target_column:
                    try:
                        le = LabelEncoder()
                        # Asegurar que los datos sean strings y no tengan valores nulos
                        col_data = encoded_data[col].astype(str).fillna('unknown')
                        encoded_data[col] = le.fit_transform(col_data)
                        self.label_encoders[col] = le
                    except Exception as e:
                        logger.warning(f"Error codificando columna {col}: {e}")
                        # Si falla la codificación, eliminar la columna
                        encoded_data = encoded_data.drop(columns=[col])
            
            logger.info(f"Variables codificadas exitosamente")
            return encoded_data
            
        except Exception as e:
            logger.error(f"Error en codificación de variables categóricas: {e}")
            return data
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Aplica SMOTE para balancear las clases y mejorar detección de positivos"""
        logger.info("Aplicando SMOTE para balancear clases...")
        class_counts = y.value_counts()
        logger.info(f"Distribución de clases antes de SMOTE: {class_counts.to_dict()}")
        
        # Verificar si hay desbalance significativo
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        logger.info(f"Ratio de desbalance: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 1.2:  # Si hay desbalance significativo
            logger.info("Aplicando SMOTE para balancear clases...")
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            balanced_counts = pd.Series(y_balanced).value_counts()
            logger.info(f"Distribución de clases después de SMOTE: {balanced_counts.to_dict()}")
            return X_balanced, y_balanced
        else:
            logger.info("No se aplica SMOTE: las clases están razonablemente balanceadas")
            return X, y
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Escala las características usando StandardScaler"""
        logger.info("Escalando características...")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled_df
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara las características y la variable objetivo original (COVID positivo/negativo)
        """
        logger.info("Preparando características para clasificación COVID positivo/negativo...")
        
        # Usar la variable objetivo original: covid_res
        # 1 = COVID positivo, 2 = COVID negativo
        if 'covid_res' not in data.columns:
            raise ValueError("La columna 'covid_res' no está presente en los datos")
        
        # Definir columnas a excluir
        exclude_cols = ['id']
        self.feature_columns = [col for col in data.columns if col not in exclude_cols and col != self.target_column]
        
        # Separar características y objetivo
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Verificar que solo tenemos valores 1 y 2
        unique_values = y.unique()
        logger.info(f"Valores únicos en variable objetivo: {unique_values}")
        logger.info(f"Distribución de clases:")
        class_counts = y.value_counts().sort_index()
        for value, count in class_counts.items():
            class_name = "Positivo" if value == 1 else "Negativo"
            percentage = count / len(y) * 100
            logger.info(f"  {class_name} (valor {value}): {count} casos ({percentage:.1f}%)")
        
        logger.info(f"Características: {len(self.feature_columns)}")
        logger.info(f"Registros: {len(data)}")
        
        return X, y
    
    def save_preprocessing_artifacts(self, output_dir: str = "models"):
        """Guarda los artefactos de preprocesamiento"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Guardar scaler
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        # Guardar label encoders
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        
        # Guardar feature columns
        joblib.dump(self.feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))
        
        # Guardar información del modelo
        model_info = {
            'target_column': self.target_column,
            'risk_levels': self.risk_levels,
            'focus': 'Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización',
            'classification_type': '3 niveles de riesgo (Alto/Medio/Bajo)'
        }
        joblib.dump(model_info, os.path.join(output_dir, 'model_info.pkl'))
        
        logger.info(f"Artefactos guardados en {output_dir}")
    
    def load_preprocessing_artifacts(self, input_dir: str = "models"):
        """Carga los artefactos de preprocesamiento"""
        try:
            self.scaler = joblib.load(os.path.join(input_dir, 'scaler.pkl'))
            self.label_encoders = joblib.load(os.path.join(input_dir, 'label_encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(input_dir, 'feature_columns.pkl'))
            logger.info("Artefactos de preprocesamiento cargados")
        except Exception as e:
            logger.error(f"Error cargando artefactos: {e}")
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma nuevos datos usando los artefactos guardados"""
        if self.feature_columns is None:
            raise ValueError("Debe cargar los artefactos de preprocesamiento primero")
        
        # Aplicar las mismas transformaciones que en el entrenamiento
        transformed_data = data.copy()
        
        # Codificar variables categóricas
        for col, le in self.label_encoders.items():
            if col in transformed_data.columns:
                transformed_data[col] = le.transform(transformed_data[col].astype(str))
        
        # Seleccionar solo las características necesarias
        available_features = [col for col in self.feature_columns if col in transformed_data.columns]
        missing_features = [col for col in self.feature_columns if col not in transformed_data.columns]
        
        if missing_features:
            logger.warning(f"Características faltantes: {missing_features}")
            # Agregar columnas faltantes con valores por defecto
            for col in missing_features:
                transformed_data[col] = 0
        
        # Reordenar columnas según el orden de entrenamiento
        transformed_data = transformed_data[self.feature_columns]
        
        # Escalar características
        transformed_data = pd.DataFrame(
            self.scaler.transform(transformed_data),
            columns=transformed_data.columns,
            index=transformed_data.index
        )
        
        return transformed_data

if __name__ == "__main__":
    # Ejemplo de uso
    from data_extractor import DataExtractor
    
    # Cargar datos
    extractor = DataExtractor()
    raw_data = extractor.load_data()
    
    # Preprocesar
    preprocessor = DataPreprocessor()
    cleaned_data = preprocessor.clean_data(raw_data)
    encoded_data = preprocessor.encode_categorical_variables(cleaned_data)
    X, y = preprocessor.prepare_features(encoded_data)
    
    print("Datos preprocesados exitosamente")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}") 