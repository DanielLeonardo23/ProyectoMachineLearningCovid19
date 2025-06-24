"""
Script de prueba para verificar el dashboard inteligente
"""

import requests
import json

def test_dashboard():
    """Prueba el dashboard inteligente con datos de ejemplo"""
    
    # URL base de la aplicación
    base_url = "http://localhost:5000"
    
    # Datos de prueba para un paciente
    test_patient_data = {
        'sex': '1',  # Mujer
        'age': '45',
        'diabetes': '1',  # Sí tiene diabetes
        'copd': '0',
        'asthma': '0',
        'hypertension': '1',  # Sí tiene hipertensión
        'cardiovascular': '0',
        'obesity': '0',
        'renal_chronic': '0',
        'tobacco': '1',  # Sí fuma
        'pregnancy': '0',
        'fever': '1',  # Sí tiene fiebre
        'cough': '1',  # Sí tiene tos
        'fatigue': '0',
        'headache': '0',
        'sore_throat': '0',
        'difficulty_breathing': '0',
        'chest_pain': '0',
        'loss_taste_smell': '0',
        'other_disease': '0'
    }
    
    print("🧪 Probando el Dashboard Inteligente...")
    print("=" * 50)
    
    # 1. Verificar estado del modelo
    print("1. Verificando estado del modelo...")
    try:
        response = requests.get(f"{base_url}/model_status")
        if response.status_code == 200:
            status_data = response.json()
            print(f"   ✅ Estado del modelo: {status_data.get('trained', False)}")
            if status_data.get('trained'):
                print(f"   ✅ Modelo: {status_data.get('model_name', 'N/A')}")
            else:
                print("   ⚠️  El modelo no está entrenado")
        else:
            print(f"   ❌ Error al verificar estado: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
        return
    
    # 2. Cargar modelo si no está entrenado
    if not status_data.get('trained', False):
        print("2. Cargando modelo pre-entrenado...")
        try:
            response = requests.post(f"{base_url}/load_model")
            if response.status_code == 200:
                load_data = response.json()
                if load_data.get('success'):
                    print("   ✅ Modelo cargado exitosamente")
                else:
                    print(f"   ❌ Error al cargar modelo: {load_data.get('message')}")
            else:
                print(f"   ❌ Error al cargar modelo: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error de conexión: {e}")
            return
    
    # 3. Realizar predicción y obtener dashboard
    print("3. Realizando predicción con datos de prueba...")
    try:
        response = requests.post(f"{base_url}/predict", data=test_patient_data)
        if response.status_code == 200:
            prediction_data = response.json()
            if prediction_data.get('success'):
                print("   ✅ Predicción realizada exitosamente")
                
                # Mostrar resultados
                risk_prediction = prediction_data.get('risk_prediction', {})
                print(f"   📊 Resultado: {risk_prediction.get('risk_level_text', 'N/A')}")
                print(f"   📊 Probabilidad: {risk_prediction.get('probability', 0):.1%}")
                print(f"   📊 Recomendación: {risk_prediction.get('recommendation', 'N/A')}")
                
                # Verificar dashboard
                dashboard = prediction_data.get('dashboard', '')
                if dashboard:
                    print("   ✅ Dashboard inteligente generado")
                    print(f"   📏 Tamaño del dashboard: {len(dashboard)} caracteres")
                    
                    # Verificar componentes del dashboard
                    components = [
                        'Métricas del Modelo',
                        'Rendimiento del Modelo',
                        'Análisis del Paciente',
                        'Explicación Médica Inteligente',
                        'Recomendaciones Personalizadas',
                        'Factores de Riesgo Identificados',
                        'Próximos Pasos Clínicos',
                        'Resumen Clínico',
                        'Alertas Clínicas'
                    ]
                    
                    print("   🔍 Verificando componentes del dashboard:")
                    for component in components:
                        if component in dashboard:
                            print(f"      ✅ {component}")
                        else:
                            print(f"      ❌ {component}")
                    
                    # Verificar gráficas
                    graphics = [
                        'confusion_matrix.png',
                        'feature_importance.png',
                        'Factores de Riesgo del Paciente',
                        'Síntomas del Paciente',
                        'Métricas del Modelo Random Forest',
                        'Nivel de Confianza del Modelo'
                    ]
                    
                    print("   📈 Verificando gráficas:")
                    for graphic in graphics:
                        if graphic in dashboard:
                            print(f"      ✅ {graphic}")
                        else:
                            print(f"      ❌ {graphic}")
                    
                else:
                    print("   ❌ No se generó el dashboard")
            else:
                print(f"   ❌ Error en la predicción: {prediction_data.get('message')}")
        else:
            print(f"   ❌ Error en la predicción: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Prueba completada")
    print("\n💡 Para ver el dashboard completo, abre tu navegador en:")
    print("   http://localhost:5000")
    print("\n📝 Datos de prueba utilizados:")
    for key, value in test_patient_data.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_dashboard() 