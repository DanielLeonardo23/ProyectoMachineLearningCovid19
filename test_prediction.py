"""
Script de prueba para el sistema predictivo COVID-19
Prueba diferentes casos para verificar la clasificación binaria
"""

import pandas as pd
import numpy as np
import joblib
from agents.data_extractor import DataExtractor
from agents.data_preprocessor import DataPreprocessor
from agents.ml_predictor import MLPredictor

def test_covid_prediction():
    """Prueba el sistema de predicción COVID-19"""
    
    print("🧪 Pruebas de casos para el sistema predictivo COVID-19 (Modelo con SMOTE)")
    print("=" * 50)
    
    # Inicializar agentes
    extractor = DataExtractor()
    preprocessor = DataPreprocessor()
    predictor = MLPredictor()
    
    # Cargar modelo entrenado con SMOTE
    print("Cargando modelo entrenado con SMOTE...")
    try:
        model = joblib.load('models/covid_model.pkl')
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        print("Entrenando nuevo modelo...")
        # Cargar y preparar datos
        raw_data = extractor.load_data()
        cleaned_data = preprocessor.clean_data(raw_data)
        encoded_data = preprocessor.encode_categorical_variables(cleaned_data)
        X, y = preprocessor.prepare_features(encoded_data)
        X_balanced, y_balanced = preprocessor.handle_class_imbalance(X, y)
        X_scaled = preprocessor.scale_features(X_balanced)
        
        # Entrenar modelo
        X_train, X_test, y_train, y_test = predictor.split_data(X_scaled, y_balanced)
        models = predictor.select_models()
        trained_models = predictor.train_models(X_train, y_train)
        evaluation_results = predictor.evaluate_models(X_test, y_test)
        best_model_name = predictor.select_best_model()
        model = joblib.load('models/covid_model.pkl')
    
    # Cargar preprocesadores
    try:
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        print("✅ Preprocesadores cargados exitosamente")
    except Exception as e:
        print(f"❌ Error cargando preprocesadores: {e}")
        return
    
    # Cargar artefactos de preprocesamiento
    preprocessor.load_preprocessing_artifacts()
    
    # Casos de prueba
    test_cases = [
        {
            "name": "Caso 1: Adulto mayor con síntomas y comorbilidades (Probable positivo)",
            "data": {
                'age': 70, 'sex': 1, 'fever': 1, 'cough': 1, 'fatigue': 1,
                'difficulty_breathing': 1, 'diabetes': 1, 'hypertension': 1,
                'obesity': 1, 'cardiovascular': 1, 'headache': 0, 'sore_throat': 0,
                'chest_pain': 0, 'loss_taste_smell': 0, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 1
            }
        },
        {
            "name": "Caso 2: Adulto joven con síntomas leves (Posible positivo)",
            "data": {
                'age': 35, 'sex': 2, 'fever': 1, 'cough': 1, 'fatigue': 0,
                'difficulty_breathing': 0, 'diabetes': 0, 'hypertension': 0,
                'obesity': 0, 'cardiovascular': 0, 'headache': 1, 'sore_throat': 1,
                'chest_pain': 0, 'loss_taste_smell': 0, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 1
            }
        },
        {
            "name": "Caso 3: Joven sin síntomas ni factores de riesgo (Probable negativo)",
            "data": {
                'age': 25, 'sex': 1, 'fever': 0, 'cough': 0, 'fatigue': 0,
                'difficulty_breathing': 0, 'diabetes': 0, 'hypertension': 0,
                'obesity': 0, 'cardiovascular': 0, 'headache': 0, 'sore_throat': 0,
                'chest_pain': 0, 'loss_taste_smell': 0, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 0
            }
        },
        {
            "name": "Caso 4: Mujer embarazada con síntomas (Alto riesgo)",
            "data": {
                'age': 28, 'sex': 2, 'fever': 1, 'cough': 1, 'fatigue': 1,
                'difficulty_breathing': 0, 'diabetes': 0, 'hypertension': 0,
                'obesity': 0, 'cardiovascular': 0, 'headache': 1, 'sore_throat': 0,
                'chest_pain': 0, 'loss_taste_smell': 1, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 1, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 1
            }
        },
        {
            "name": "Caso 5: Adulto con síntomas pero sin comorbilidades",
            "data": {
                'age': 45, 'sex': 1, 'fever': 1, 'cough': 0, 'fatigue': 1,
                'difficulty_breathing': 0, 'diabetes': 0, 'hypertension': 0,
                'obesity': 0, 'cardiovascular': 0, 'headache': 1, 'sore_throat': 1,
                'chest_pain': 0, 'loss_taste_smell': 0, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 0
            }
        },
        {
            "name": "Caso 6: Adolescente sin síntomas, pero con contacto COVID",
            "data": {
                'age': 16, 'sex': 2, 'fever': 0, 'cough': 0, 'fatigue': 0,
                'difficulty_breathing': 0, 'diabetes': 0, 'hypertension': 0,
                'obesity': 0, 'cardiovascular': 0, 'headache': 0, 'sore_throat': 0,
                'chest_pain': 0, 'loss_taste_smell': 0, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 1
            }
        },
        # CASOS EXTREMOS PARA PROBAR PREDICCIONES POSITIVAS
        {
            "name": "Caso 7: CASO EXTREMO - Adulto mayor con TODOS los síntomas y comorbilidades",
            "data": {
                'age': 80, 'sex': 1, 'fever': 1, 'cough': 1, 'fatigue': 1,
                'difficulty_breathing': 1, 'diabetes': 1, 'hypertension': 1,
                'obesity': 1, 'cardiovascular': 1, 'headache': 1, 'sore_throat': 1,
                'chest_pain': 1, 'loss_taste_smell': 1, 'copd': 1, 'asthma': 1,
                'renal_chronic': 1, 'inmsupr': 1, 'pregnancy': 0, 'tobacco': 1,
                'other_disease': 1, 'contact_other_covid': 1
            }
        },
        {
            "name": "Caso 8: CASO EXTREMO - Paciente con síntomas respiratorios severos",
            "data": {
                'age': 65, 'sex': 2, 'fever': 1, 'cough': 1, 'fatigue': 1,
                'difficulty_breathing': 1, 'diabetes': 1, 'hypertension': 1,
                'obesity': 1, 'cardiovascular': 1, 'headache': 1, 'sore_throat': 1,
                'chest_pain': 1, 'loss_taste_smell': 1, 'copd': 1, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 1,
                'other_disease': 1, 'contact_other_covid': 1
            }
        },
        {
            "name": "Caso 9: CASO EXTREMO - Paciente inmunosuprimido con síntomas",
            "data": {
                'age': 55, 'sex': 1, 'fever': 1, 'cough': 1, 'fatigue': 1,
                'difficulty_breathing': 1, 'diabetes': 1, 'hypertension': 1,
                'obesity': 0, 'cardiovascular': 0, 'headache': 1, 'sore_throat': 1,
                'chest_pain': 0, 'loss_taste_smell': 1, 'copd': 0, 'asthma': 0,
                'renal_chronic': 1, 'inmsupr': 1, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 1, 'contact_other_covid': 1
            }
        },
        {
            "name": "Caso 10: CASO EXTREMO - Paciente con pérdida de gusto/olfato (síntoma específico COVID)",
            "data": {
                'age': 40, 'sex': 2, 'fever': 1, 'cough': 1, 'fatigue': 1,
                'difficulty_breathing': 0, 'diabetes': 0, 'hypertension': 0,
                'obesity': 0, 'cardiovascular': 0, 'headache': 1, 'sore_throat': 1,
                'chest_pain': 0, 'loss_taste_smell': 1, 'copd': 0, 'asthma': 0,
                'renal_chronic': 0, 'inmsupr': 0, 'pregnancy': 0, 'tobacco': 0,
                'other_disease': 0, 'contact_other_covid': 1
            }
        }
    ]
    
    # Probar cada caso
    for i, case in enumerate(test_cases, 1):
        print(f"\n=== {case['name']} ===")
        
        # Crear DataFrame con los datos del caso
        case_df = pd.DataFrame([case['data']])
        
        # Transformar datos usando el preprocesador
        try:
            transformed_data = preprocessor.transform_new_data(case_df)
            transformed_data = transformed_data[feature_columns]  # Solo las columnas usadas en entrenamiento
            
            # Escalar datos
            transformed_data_scaled = scaler.transform(transformed_data)
            
            # Realizar predicción usando el modelo cargado
            probabilities = model.predict_proba(transformed_data_scaled)[0]
            prediction_class = model.predict(transformed_data_scaled)[0]
            
            # Convertir a formato del sistema
            covid_result = "Positivo" if prediction_class == 1 else "Negativo"
            probability = probabilities[1] if prediction_class == 1 else probabilities[0]
            
            # Crear recomendación
            if prediction_class == 1:
                if probability > 0.8:
                    recommendation = f"POSITIVO: Resultado COVID POSITIVO con alta confianza ({probability:.1%}). Aislamiento inmediato requerido."
                elif probability > 0.6:
                    recommendation = f"POSITIVO: Resultado COVID POSITIVO con confianza moderada ({probability:.1%}). Aislamiento recomendado."
                else:
                    recommendation = f"POSITIVO: Resultado COVID POSITIVO con baja confianza ({probability:.1%}). Prueba confirmatoria recomendada."
            else:
                if probability > 0.8:
                    recommendation = f"NEGATIVO: Resultado COVID NEGATIVO con alta confianza ({probability:.1%}). Mantener precauciones básicas."
                elif probability > 0.6:
                    recommendation = f"NEGATIVO: Resultado COVID NEGATIVO con confianza moderada ({probability:.1%}). Mantener precauciones básicas."
                else:
                    recommendation = f"NEGATIVO: Resultado COVID NEGATIVO con baja confianza ({probability:.1%}). Monitoreo recomendado."
            
            # Mostrar resultados
            print(f"Resultado COVID: {covid_result}")
            print(f"Probabilidad: {probability:.2%}")
            print(f"Recomendación: {recommendation}")
            print(f"Probabilidades: {{'Positivo': {probabilities[1]}, 'Negativo': {probabilities[0]}}}")
            
        except Exception as e:
            print(f"Error en caso {i}: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Pruebas finalizadas. Revisa los resultados arriba.")
    print("🌐 Puedes acceder al dashboard en: http://127.0.0.1:5000")

if __name__ == "__main__":
    test_covid_prediction() 