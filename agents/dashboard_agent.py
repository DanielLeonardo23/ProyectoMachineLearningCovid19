"""
Agente 4: Dashboard y API de Gemini
Responsable de mostrar resultados y proporcionar explicaciones usando IA
Enfocado en prevención temprana y clasificación de riesgo de hospitalización
"""

import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardAgent:
    """Agente responsable del dashboard y explicaciones con IA"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.covid_classes = {1: 'Positivo', 2: 'Negativo'}
        
    def generate_explanation(self, risk_prediction: Dict[str, Any], 
                           patient_data: Dict[str, Any], 
                           model_metrics: Dict[str, Any]) -> str:
        """
        Genera una explicación personalizada usando Gemini para clasificación de riesgo
        """
        try:
            risk_level = risk_prediction['risk_level_text']
            probability = risk_prediction['probability']
            recommendation = risk_prediction['recommendation']
            
            # Crear prompt para Gemini
            prompt = f"""
            Eres un experto médico especializado en prevención temprana de COVID-19. Analiza los siguientes datos de un paciente y proporciona una explicación clara sobre el riesgo de hospitalización:

            DATOS DEL PACIENTE:
            {json.dumps(patient_data, indent=2, default=str)}

            EVALUACIÓN DE RIESGO:
            - Nivel de Riesgo: {risk_level}
            - Probabilidad: {probability:.2%}
            - Recomendación: {recommendation}

            MÉTRICAS DEL MODELO:
            - Precisión: {model_metrics.get('accuracy', 0):.2%}
            - Sensibilidad: {model_metrics.get('recall', 0):.2%}
            - Especificidad: {model_metrics.get('precision', 0):.2%}

            Por favor proporciona:
            1. Una explicación clara del nivel de riesgo en términos médicos simples
            2. Factores de riesgo identificados que contribuyen a este nivel
            3. Explicación de por qué el paciente tiene este nivel de riesgo
            4. Medidas preventivas específicas para evitar hospitalización
            5. Cuándo buscar atención médica basado en el nivel de riesgo
            6. Expectativas sobre la evolución del caso

            Responde en español de manera profesional pero comprensible para el paciente.
            Enfócate en PREVENCIÓN TEMPRANA para evitar hospitalización.
            """

            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error al generar explicación: {e}")
            return "No se pudo generar la explicación en este momento."
    
    def generate_recommendations(self, risk_prediction: Dict[str, Any], 
                               patient_data: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones específicas basadas en el nivel de riesgo
        """
        try:
            risk_level = risk_prediction['risk_level_text']
            probability = risk_prediction['probability']
            
            prompt = f"""
            Basándote en estos datos de un paciente evaluado para riesgo de hospitalización por COVID-19:

            DATOS DEL PACIENTE:
            {json.dumps(patient_data, indent=2, default=str)}

            EVALUACIÓN DE RIESGO:
            - Nivel de Riesgo: {risk_level}
            - Probabilidad: {probability:.2%}

            Genera 5-7 recomendaciones específicas y prácticas para PREVENIR la hospitalización.
            Incluye:
            - Medidas inmediatas de prevención
            - Cuándo contactar a un médico
            - Precauciones específicas según el nivel de riesgo
            - Monitoreo de síntomas críticos
            - Medidas preventivas para evitar complicaciones
            - Cuándo buscar atención de emergencia

            Responde solo con las recomendaciones, una por línea, en español.
            Enfócate en PREVENCIÓN TEMPRANA.
            """

            response = self.model.generate_content(prompt)
            recommendations = [rec.strip() for rec in response.text.split('\n') if rec.strip()]
            return recommendations[:7]  # Máximo 7 recomendaciones
            
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {e}")
            return self._get_default_recommendations(risk_prediction['risk_level'])
    
    def _get_default_recommendations(self, risk_level: int) -> List[str]:
        """Recomendaciones por defecto basadas en el nivel de riesgo"""
        if risk_level == 1:  # Alto riesgo
            return [
                "Aislamiento estricto inmediato",
                "Contactar médico en las próximas 24 horas",
                "Monitorear síntomas cada 2-4 horas",
                "Preparar información de contactos cercanos",
                "Tener plan de emergencia listo",
                "Evitar contacto con personas vulnerables",
                "Mantener oxímetro de pulso si está disponible"
            ]
        elif risk_level == 2:  # Riesgo medio
            return [
                "Aislamiento preventivo",
                "Consultar médico en 48-72 horas",
                "Monitorear síntomas diariamente",
                "Mantener precauciones de higiene",
                "Descansar adecuadamente",
                "Hidratarse bien",
                "Contactar médico si empeoran síntomas"
            ]
        else:  # Bajo riesgo
            return [
                "Mantener precauciones básicas",
                "Monitorear síntomas leves",
                "Seguir protocolos de prevención",
                "Mantener buena hidratación",
                "Descansar si es necesario",
                "Contactar médico si aparecen síntomas nuevos",
                "Mantener distancia social"
            ]
    
    def create_prediction_dashboard(self, patient_data: Dict[str, Any], 
                                  risk_prediction: Dict[str, Any],
                                  model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un dashboard completo con la evaluación de riesgo y explicaciones clínicas
        """
        # Generar explicación y recomendaciones
        explanation = self.generate_explanation(risk_prediction, patient_data, model_metrics)
        recommendations = self.generate_recommendations(risk_prediction, patient_data)
        
        # Generar alertas clínicas y explicación detallada
        clinical_alert = self.generate_clinical_alert(risk_prediction, patient_data)
        detailed_explanation = self.generate_detailed_explanation(risk_prediction, patient_data)
        
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
        
        # Crear dashboard completo
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "patient_data": patient_data,
            "risk_assessment": {
                "level": int(risk_prediction['risk_level']),
                "level_text": risk_prediction['risk_level_text'],
                "probability": float(risk_prediction['probability']),
                "confidence_level": self._get_confidence_level(risk_prediction['probability']),
                "all_probabilities": {
                    'Positivo': float(risk_prediction['all_probabilities']['Positivo']),
                    'Negativo': float(risk_prediction['all_probabilities']['Negativo'])
                },
                "recommendation": risk_prediction['recommendation']
            },
            "clinical_alert": clinical_alert,
            "detailed_explanation": detailed_explanation,
            "explanation": explanation,
            "recommendations": recommendations,
            "model_metrics": metrics_for_json,
            "risk_factors": self._identify_risk_factors(patient_data),
            "next_steps": self._get_next_steps(risk_prediction),
            "clinical_summary": self._generate_clinical_summary(risk_prediction, patient_data),
            "focus": "Prevención temprana de COVID-19 y clasificación de riesgo de hospitalización"
        }
        
        return dashboard
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determina el nivel de confianza basado en la probabilidad"""
        if probability >= 0.9:
            return "MUY ALTA"
        elif probability >= 0.8:
            return "ALTA"
        elif probability >= 0.7:
            return "MODERADA"
        elif probability >= 0.6:
            return "BAJA"
        else:
            return "MUY BAJA"
    
    def _identify_risk_factors(self, patient_data: Dict[str, Any]) -> List[str]:
        """Identifica factores de riesgo específicos del paciente"""
        risk_factors = []
        
        # Factores de riesgo por edad
        age = patient_data.get('age', 0)
        if age >= 65:
            risk_factors.append(f"Edad avanzada ({age} años) - Factor de riesgo significativo")
        elif age >= 50:
            risk_factors.append(f"Edad adulta ({age} años) - Factor de riesgo moderado")
        elif age < 18:
            risk_factors.append(f"Edad pediátrica ({age} años) - Requiere atención especial")
        
        # Condiciones médicas preexistentes
        medical_conditions = {
            'diabetes': 'Diabetes mellitus',
            'hypertension': 'Hipertensión arterial',
            'obesity': 'Obesidad',
            'asthma': 'Asma',
            'copd': 'EPOC',
            'cardiovascular': 'Enfermedad cardiovascular',
            'renal_chronic': 'Enfermedad renal crónica',
            'inmsupr': 'Inmunosupresión',
            'other_disease': 'Otras enfermedades',
            'pregnancy': 'Embarazo',
            'tobacco': 'Tabaquismo'
        }
        
        for condition, description in medical_conditions.items():
            if patient_data.get(condition, 0) == 1:
                risk_factors.append(f"{description} - Factor de riesgo importante")
        
        # Síntomas actuales
        symptoms = {
            'fever': 'Fiebre',
            'cough': 'Tos',
            'fatigue': 'Fatiga',
            'headache': 'Dolor de cabeza',
            'sore_throat': 'Dolor de garganta',
            'difficulty_breathing': 'Dificultad respiratoria',
            'chest_pain': 'Dolor en el pecho',
            'loss_taste_smell': 'Pérdida de gusto/olfato'
        }
        
        current_symptoms = []
        for symptom, description in symptoms.items():
            if patient_data.get(symptom, 0) == 1:
                current_symptoms.append(description)
        
        if current_symptoms:
            risk_factors.append(f"Síntomas actuales: {', '.join(current_symptoms)}")
        
        # Factores de riesgo adicionales
        if len(risk_factors) == 0:
            risk_factors.append("Sin factores de riesgo identificados")
        
        return risk_factors
    
    def _get_next_steps(self, risk_prediction: Dict[str, Any]) -> List[str]:
        """Define los siguientes pasos clínicos según el nivel de riesgo"""
        risk_level = risk_prediction['risk_level']
        probability = risk_prediction['probability']
        
        if risk_level == 1:  # Alto riesgo
            return [
                "🚨 URGENTE: Contactar médico en las próximas 6-12 horas",
                "🔴 Aislamiento estricto inmediato",
                "📊 Monitoreo de signos vitales cada 2-4 horas",
                "🏥 Preparar para posible hospitalización",
                "📞 Tener números de emergencia a mano",
                "👥 Informar contactos cercanos",
                "💊 Seguir tratamiento prescrito estrictamente"
            ]
        elif risk_level == 2:  # Riesgo medio
            return [
                "⚠️ Consultar médico en 24-48 horas",
                "🟡 Aislamiento preventivo",
                "📊 Monitoreo de síntomas diariamente",
                "🏠 Preparar plan de contingencia",
                "📞 Mantener comunicación con médico",
                "💊 Seguir recomendaciones médicas",
                "🔍 Observar cambios en síntomas"
            ]
        else:  # Bajo riesgo
            return [
                "✅ Seguimiento médico regular",
                "🟢 Mantener precauciones básicas",
                "📊 Monitoreo de síntomas leves",
                "🏠 Aislamiento preventivo si es necesario",
                "📞 Contactar médico si empeoran síntomas",
                "💊 Seguir tratamiento si está prescrito",
                "🔍 Mantener vigilancia de síntomas"
            ]
    
    def generate_clinical_alert(self, risk_prediction: Dict[str, Any], 
                              patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera alertas clínicas específicas"""
        risk_level = risk_prediction['risk_level']
        probability = risk_prediction['probability']
        
        alert = {
            "severity": "low",
            "message": "",
            "immediate_actions": [],
            "warning_signs": [],
            "clinical_notes": ""
        }
        
        if risk_level == 1:  # Alto riesgo
            alert["severity"] = "high"
            alert["message"] = "🚨 ALERTA CLÍNICA: Paciente en ALTO RIESGO de hospitalización"
            alert["immediate_actions"] = [
                "Contactar médico inmediatamente",
                "Iniciar aislamiento estricto",
                "Preparar para posible hospitalización",
                "Monitorear signos vitales frecuentemente"
            ]
            alert["warning_signs"] = [
                "Dificultad respiratoria progresiva",
                "Fiebre alta persistente (>39°C)",
                "Dolor en el pecho",
                "Confusión o alteración del estado mental",
                "Cianosis (coloración azulada)",
                "Disminución del nivel de consciencia"
            ]
            alert["clinical_notes"] = "Paciente requiere vigilancia intensiva y posible hospitalización"
            
        elif risk_level == 2:  # Riesgo medio
            alert["severity"] = "medium"
            alert["message"] = "⚠️ ADVERTENCIA CLÍNICA: Paciente en RIESGO MEDIO"
            alert["immediate_actions"] = [
                "Consultar médico en 24-48 horas",
                "Mantener aislamiento preventivo",
                "Monitorear síntomas diariamente",
                "Preparar plan de contingencia"
            ]
            alert["warning_signs"] = [
                "Empeoramiento de síntomas respiratorios",
                "Fiebre que no mejora",
                "Fatiga extrema",
                "Pérdida de apetito",
                "Dolor muscular intenso"
            ]
            alert["clinical_notes"] = "Paciente requiere seguimiento cercano y posible escalación de cuidados"
            
        else:  # Bajo riesgo
            alert["severity"] = "low"
            alert["message"] = "✅ PACIENTE EN BAJO RIESGO"
            alert["immediate_actions"] = [
                "Mantener precauciones básicas",
                "Seguimiento médico regular",
                "Monitoreo de síntomas leves",
                "Mantener buena hidratación"
            ]
            alert["warning_signs"] = [
                "Aparición de síntomas nuevos",
                "Empeoramiento de síntomas existentes",
                "Fiebre que persiste más de 3 días",
                "Dificultad para respirar"
            ]
            alert["clinical_notes"] = "Paciente con bajo riesgo, mantener vigilancia rutinaria"
        
        return alert
    
    def generate_detailed_explanation(self, risk_prediction: Dict[str, Any], 
                                    patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera explicación clínica detallada"""
        risk_level = risk_prediction['risk_level']
        probability = risk_prediction['probability']
        
        explanation = {
            "risk_interpretation": "",
            "clinical_reasoning": "",
            "prognosis": "",
            "monitoring_plan": "",
            "prevention_strategy": ""
        }
        
        if risk_level == 1:  # Alto riesgo
            explanation["risk_interpretation"] = "El paciente presenta un ALTO RIESGO de hospitalización por COVID-19. Los factores identificados sugieren una evolución clínica que podría requerir atención hospitalaria."
            explanation["clinical_reasoning"] = "La combinación de factores de riesgo y síntomas actuales indica una respuesta inmunológica comprometida y mayor probabilidad de complicaciones respiratorias."
            explanation["prognosis"] = "Sin intervención temprana, existe riesgo significativo de progresión a enfermedad severa que requiera hospitalización y posible ventilación mecánica."
            explanation["monitoring_plan"] = "Monitoreo intensivo de signos vitales, saturación de oxígeno, y síntomas respiratorios cada 2-4 horas. Evaluación médica inmediata si hay deterioro."
            explanation["prevention_strategy"] = "Intervención médica temprana, posible tratamiento antiviral, y preparación para hospitalización si es necesario."
            
        elif risk_level == 2:  # Riesgo medio
            explanation["risk_interpretation"] = "El paciente presenta un RIESGO MEDIO de hospitalización. Requiere vigilancia cercana pero puede manejarse ambulatoriamente con seguimiento médico."
            explanation["clinical_reasoning"] = "Los factores de riesgo identificados sugieren una evolución clínica moderada que puede controlarse con intervención médica oportuna."
            explanation["prognosis"] = "Con manejo adecuado, la mayoría de los pacientes en este grupo evolucionan favorablemente sin necesidad de hospitalización."
            explanation["monitoring_plan"] = "Monitoreo diario de síntomas, signos vitales una vez al día, y evaluación médica en 24-48 horas."
            explanation["prevention_strategy"] = "Seguimiento médico cercano, tratamiento sintomático, y escalación de cuidados si hay deterioro."
            
        else:  # Bajo riesgo
            explanation["risk_interpretation"] = "El paciente presenta un BAJO RIESGO de hospitalización. La evolución clínica esperada es favorable con manejo ambulatorio."
            explanation["clinical_reasoning"] = "Los factores de riesgo son mínimos y los síntomas sugieren una respuesta inmunológica adecuada al virus."
            explanation["prognosis"] = "Evolución clínica favorable esperada. Baja probabilidad de complicaciones severas o necesidad de hospitalización."
            explanation["monitoring_plan"] = "Monitoreo de síntomas leves, evaluación médica si hay cambios significativos, y seguimiento rutinario."
            explanation["prevention_strategy"] = "Manejo sintomático, medidas preventivas básicas, y vigilancia de posibles complicaciones."
        
        return explanation
    
    def create_visualization_data(self, patient_data: Dict[str, Any], 
                                risk_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea datos para visualizaciones del dashboard
        """
        # Datos para gráfico de probabilidades
        prob_data = risk_prediction['all_probabilities']
        
        # Datos para factores de riesgo
        risk_factors = self._identify_risk_factors(patient_data)
        
        # Datos para síntomas
        symptoms = []
        symptom_mapping = {
            'fever': 'Fiebre',
            'cough': 'Tos',
            'fatigue': 'Fatiga',
            'headache': 'Dolor de cabeza',
            'sore_throat': 'Dolor de garganta',
            'difficulty_breathing': 'Dificultad para respirar',
            'chest_pain': 'Dolor en el pecho',
            'loss_taste_smell': 'Pérdida de gusto/olfato'
        }
        
        for symptom_key, symptom_name in symptom_mapping.items():
            if patient_data.get(symptom_key, 0) == 1:
                symptoms.append(symptom_name)
        
        visualization_data = {
            "risk_probabilities": {
                "labels": list(prob_data.keys()),
                "values": list(prob_data.values()),
                "colors": ['#ff4444', '#ffaa00', '#44aa44']
            },
            "risk_factors": {
                "labels": risk_factors,
                "values": [1] * len(risk_factors)
            },
            "symptoms": {
                "labels": symptoms,
                "values": [1] * len(symptoms)
            },
            "risk_level": {
                "level": risk_prediction['risk_level'],
                "level_text": risk_prediction['risk_level_text'],
                "color": ['#ff4444', '#ffaa00', '#44aa44'][risk_prediction['risk_level'] - 1]
            }
        }
        
        return visualization_data
    
    def _generate_clinical_summary(self, risk_prediction: Dict[str, Any], 
                                 patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un resumen clínico conciso"""
        risk_level = risk_prediction['risk_level']
        probability = risk_prediction['probability']
        
        summary = {
            "key_message": "",
            "priority_level": "",
            "time_to_action": "",
            "clinical_urgency": "",
            "expected_course": ""
        }
        
        if risk_level == 1:  # Alto riesgo
            summary["key_message"] = "Paciente en ALTO RIESGO de hospitalización por COVID-19"
            summary["priority_level"] = "URGENTE"
            summary["time_to_action"] = "Inmediato (6-12 horas)"
            summary["clinical_urgency"] = "Requiere evaluación médica urgente"
            summary["expected_course"] = "Alto riesgo de progresión a enfermedad severa"
            
        elif risk_level == 2:  # Riesgo medio
            summary["key_message"] = "Paciente en RIESGO MEDIO de hospitalización"
            summary["priority_level"] = "MODERADO"
            summary["time_to_action"] = "24-48 horas"
            summary["clinical_urgency"] = "Requiere seguimiento médico cercano"
            summary["expected_course"] = "Evolución moderada con manejo adecuado"
            
        else:  # Bajo riesgo
            summary["key_message"] = "Paciente en BAJO RIESGO de hospitalización"
            summary["priority_level"] = "BAJO"
            summary["time_to_action"] = "Seguimiento regular"
            summary["clinical_urgency"] = "Manejo ambulatorio rutinario"
            summary["expected_course"] = "Evolución favorable esperada"
        
        return summary

if __name__ == "__main__":
    # Ejemplo de uso
    dashboard_agent = DashboardAgent("your_api_key_here")
    print("Agente de dashboard inicializado")
    print(f"Niveles de riesgo: {dashboard_agent.covid_classes}") 