"""
Agente 1: Extracción de Datos
Responsable de cargar y preparar el dataset inicial de COVID-19
Enfocado en prevención temprana y detección de riesgo de hospitalización
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    """Agente responsable de la extracción y carga inicial de datos"""
    
    def __init__(self, file_path: str = "covid.csv"):
        self.file_path = file_path
        self.raw_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Fase 2: Recolección de Datos
        Carga el dataset de COVID-19 desde el archivo CSV
        Filtra solo las instancias con covid_res en [1, 2] para clasificación binaria
        Elimina características irrelevantes para prevención temprana
        """
        try:
            logger.info("Cargando dataset de COVID-19...")
            df = pd.read_csv(self.file_path)
            logger.info(f"Dataset cargado exitosamente. Shape: {df.shape}")
            
            # Filtrar solo clases 1 y 2 (COVID positivo y negativo)
            df = df[df['covid_res'].isin([1, 2])].copy()
            logger.info(f"Dataset filtrado para clasificación binaria. Shape: {df.shape}")
            
            # Eliminar características irrelevantes para prevención temprana
            # Estas características indican ya hospitalización o casos severos
            irrelevant_features = [
                'pneumonia',      # Neumonía (ya hospitalizado)
                'intubed',        # Intubado (ya en UCI)
                'icu',            # UCI (ya hospitalizado)
                'date_died',      # Fecha de muerte (no relevante para prevención)
                'date_died_day',
                'date_died_month', 
                'date_died_year'
            ]
            
            # Eliminar columnas irrelevantes si existen
            existing_irrelevant = [col for col in irrelevant_features if col in df.columns]
            if existing_irrelevant:
                df = df.drop(columns=existing_irrelevant)
                logger.info(f"Características irrelevantes eliminadas: {existing_irrelevant}")
            
            self.raw_data = df
            return self.raw_data
        except Exception as e:
            logger.error(f"Error al cargar el dataset: {e}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """Obtiene información básica del dataset"""
        if self.raw_data is None:
            self.load_data()
        if self.raw_data is None or not isinstance(self.raw_data, pd.DataFrame):
            return {}
        info = {
            "shape": self.raw_data.shape,
            "columns": list(self.raw_data.columns),
            "dtypes": {col: str(self.raw_data[col].dtype) for col in self.raw_data.columns},
            "missing_values": self.raw_data.isnull().sum().to_dict(),
            "memory_usage": int(self.raw_data.memory_usage(deep=True).sum())
        }
        return info
    
    def get_target_variable_info(self) -> Dict[str, Any]:
        """Analiza la variable objetivo (covid_res)"""
        if self.raw_data is None:
            self.load_data()
        if self.raw_data is None or not isinstance(self.raw_data, pd.DataFrame):
            return {}
        target_col = 'covid_res'
        target_info = {
            "variable": target_col,
            "unique_values": self.raw_data[target_col].unique().tolist(),
            "value_counts": self.raw_data[target_col].value_counts().to_dict(),
            "missing_values": int(self.raw_data[target_col].isnull().sum()),
            "description": "1: COVID-19 Positivo, 2: COVID-19 Negativo"
        }
        return target_info
    
    def get_feature_variables(self) -> list:
        """Obtiene la lista de variables predictoras para prevención temprana"""
        if self.raw_data is None:
            self.load_data()
            
        # Excluir columnas que no son features para prevención temprana
        exclude_cols = [
            'id', 
            'entry_date', 
            'date_symptoms', 
            'covid_res',
            # Características de fechas que no son relevantes para predicción
            'entry_date_day', 'entry_date_month', 'entry_date_year',
            'date_symptoms_day', 'date_symptoms_month', 'date_symptoms_year'
        ]
        feature_cols = [col for col in self.raw_data.columns if col not in exclude_cols]
        
        return feature_cols
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Valida la calidad inicial de los datos"""
        if self.raw_data is None:
            self.load_data()
        if self.raw_data is None or not isinstance(self.raw_data, pd.DataFrame):
            return {}
        quality_report = {
            "total_rows": int(len(self.raw_data)),
            "total_columns": int(len(self.raw_data.columns)),
            "missing_data_percentage": float((self.raw_data.isnull().sum().sum() / (len(self.raw_data) * len(self.raw_data.columns))) * 100),
            "duplicate_rows": int(self.raw_data.duplicated().sum()),
            "data_types": {col: str(self.raw_data[col].dtype) for col in self.raw_data.columns},
            "focus": "Prevención temprana de COVID-19 y riesgo de hospitalización"
        }
        return quality_report

if __name__ == "__main__":
    # Ejemplo de uso
    extractor = DataExtractor()
    data = extractor.load_data()
    print("Información del dataset:")
    print(extractor.get_data_info())
    print("\nInformación de la variable objetivo:")
    print(extractor.get_target_variable_info())
    print("\nVariables predictoras:")
    print(extractor.get_feature_variables())
    print("\nReporte de calidad:")
    print(extractor.validate_data_quality()) 