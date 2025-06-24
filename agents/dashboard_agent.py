"""
Agente Inteligente: Dashboard y API de Gemini
Responsable de mostrar resultados y proporcionar explicaciones usando IA avanzada
Enfocado en prevención temprana y clasificación de riesgo de hospitalización
AGENTE INTELIGENTE PRINCIPAL - Utiliza IA generativa para explicaciones médicas
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
import requests
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentDashboardAgent:
    """Agente inteligente responsable del dashboard y explicaciones con IA avanzada"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.covid_classes = {1: 'Positivo', 2: 'Negativo'}
        self.conversation_history = []
        self.medical_knowledge_base = self._initialize_medical_knowledge()
        
    def _initialize_medical_knowledge(self) -> Dict[str, Any]:
        """Inicializa la base de conocimiento médico para el agente"""
        return {
            "risk_factors": {
                "high_risk": ["diabetes", "copd", "asthma", "inmsupr", "hypertension", "cardiovascular", "obesity", "renal_chronic"],
                "moderate_risk": ["tobacco", "pregnancy", "other_disease"],
                "age_risk": {
                    "very_high": (75, 120),
                    "high": (65, 74),
                    "moderate": (50, 64),
                    "low": (18, 49),
                    "very_low": (0, 17)
                }
            },
            "symptoms_severity": {
                "critical": ["difficulty_breathing", "chest_pain"],
                "moderate": ["fever", "cough", "fatigue"],
                "mild": ["headache", "sore_throat", "loss_taste_smell"]
            },
            "prevention_guidelines": {
                "isolation": "Aislamiento estricto para casos positivos",
                "monitoring": "Monitoreo de síntomas cada 4-6 horas",
                "hydration": "Hidratación abundante",
                "rest": "Descanso adecuado",
                "medical_contact": "Contacto médico inmediato si empeoran síntomas"
            }
        }
        
    def generate_intelligent_explanation(self, risk_prediction: Dict[str, Any], 
                                       patient_data: Dict[str, Any], 
                                       model_metrics: Dict[str, Any]) -> str:
        """
        Genera una explicación inteligente usando IA avanzada con contexto médico,
        enfocada en el riesgo clínico y complicaciones si el paciente es positivo,
        y en prevención si es negativo.
        """
        try:
            risk_level = risk_prediction['risk_level_text']
            probability = risk_prediction['probability']
            recommendation = risk_prediction['recommendation']
            
            # Analizar factores de riesgo específicos del paciente
            patient_risk_factors = self._analyze_patient_risk_factors(patient_data)
            symptom_analysis = self._analyze_symptoms(patient_data)
            age_risk_assessment = self._assess_age_risk(patient_data.get('age', 0))

            # Enfoque del prompt según resultado
            if 'positivo' in risk_level.lower():
                risk_focus = "El paciente ha dado POSITIVO para COVID-19. Explica el riesgo de complicaciones, hospitalización o gravedad, considerando los factores de riesgo presentes. Resalta qué condiciones o síntomas aumentan el riesgo de evolución desfavorable y qué medidas tomar para evitar complicaciones."
            else:
                risk_focus = "El paciente ha dado NEGATIVO para COVID-19. Explica por qué su riesgo de complicaciones es bajo, qué factores protectores tiene y qué precauciones debe mantener para evitar contagio o complicaciones futuras."

            # Crear prompt inteligente con contexto médico y enfoque en riesgo
            prompt = f"""
            Eres un médico experto en COVID-19 y medicina preventiva. Analiza los siguientes datos de un paciente y proporciona una explicación médica profesional y comprensible sobre el RIESGO CLÍNICO ante COVID-19:

            === DATOS DEL PACIENTE ===
            {json.dumps(patient_data, indent=2, default=str)}

            === ANÁLISIS DE RIESGO ===
            - Nivel de Riesgo: {risk_level}
            - Probabilidad: {probability:.2%}
            - Recomendación: {recommendation}

            === FACTORES DE RIESGO IDENTIFICADOS ===
            {json.dumps(patient_risk_factors, indent=2)}

            === ANÁLISIS DE SÍNTOMAS ===
            {json.dumps(symptom_analysis, indent=2)}

            === EVALUACIÓN POR EDAD ===
            {json.dumps(age_risk_assessment, indent=2)}

            === MÉTRICAS DEL MODELO ===
            - Precisión: {model_metrics.get('accuracy', 0):.2%}
            - Sensibilidad: {model_metrics.get('recall', 0):.2%}
            - Especificidad: {model_metrics.get('precision', 0):.2%}

            {risk_focus}

            Por favor proporciona una explicación médica que incluya:

            1. **RIESGO DE COMPLICACIONES**: Explica el riesgo de complicaciones o evolución grave si el paciente es positivo, o el bajo riesgo si es negativo.
            2. **FACTORES CONTRIBUYENTES**: Identifica los factores específicos que aumentan o disminuyen el riesgo.
            3. **JUSTIFICACIÓN MÉDICA**: Explica por qué el paciente tiene este nivel de riesgo y cómo influyen sus condiciones.
            4. **RECOMENDACIONES PERSONALIZADAS**: Medidas específicas para prevenir complicaciones o contagio.
            5. **SEÑALES DE ALERTA**: Cuándo debe buscar atención médica urgente.
            6. **EXPECTATIVAS**: Evolución esperada del caso según su perfil.
            7. **CONFIANZA DEL MODELO**: Explica la confiabilidad de la predicción.

            Responde en español de manera profesional pero comprensible.
            Enfócate en PREVENCIÓN DE COMPLICACIONES y EVITAR HOSPITALIZACIÓN.
            Usa terminología médica apropiada pero explica términos técnicos.
            """

            response = self.model.generate_content(prompt)
            explanation = response.text
            
            # Guardar en historial de conversación
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'patient_data': patient_data,
                'prediction': risk_prediction,
                'explanation': explanation
            })
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error al generar explicación inteligente: {e}")
            return self._generate_fallback_explanation(risk_prediction, patient_data)
    
    def _analyze_patient_risk_factors(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza los factores de riesgo específicos del paciente"""
        risk_analysis = {
            "high_risk_conditions": [],
            "moderate_risk_conditions": [],
            "total_risk_score": 0,
            "risk_level": "bajo"
        }
        
        # Analizar condiciones de alto riesgo
        for condition in self.medical_knowledge_base["risk_factors"]["high_risk"]:
            if patient_data.get(condition, 0) == 1:
                risk_analysis["high_risk_conditions"].append(condition)
                risk_analysis["total_risk_score"] += 5
        
        # Analizar condiciones de riesgo moderado
        for condition in self.medical_knowledge_base["risk_factors"]["moderate_risk"]:
            if patient_data.get(condition, 0) == 1:
                risk_analysis["moderate_risk_conditions"].append(condition)
                risk_analysis["total_risk_score"] += 3
        
        # Determinar nivel de riesgo
        if risk_analysis["total_risk_score"] >= 10:
            risk_analysis["risk_level"] = "muy alto"
        elif risk_analysis["total_risk_score"] >= 5:
            risk_analysis["risk_level"] = "alto"
        elif risk_analysis["total_risk_score"] >= 2:
            risk_analysis["risk_level"] = "moderado"
        else:
            risk_analysis["risk_level"] = "bajo"
            
        return risk_analysis
    
    def _analyze_symptoms(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza los síntomas del paciente"""
        symptom_analysis = {
            "critical_symptoms": [],
            "moderate_symptoms": [],
            "mild_symptoms": [],
            "total_symptom_score": 0
        }
        
        # Analizar síntomas críticos
        for symptom in self.medical_knowledge_base["symptoms_severity"]["critical"]:
            if patient_data.get(symptom, 0) == 1:
                symptom_analysis["critical_symptoms"].append(symptom)
                symptom_analysis["total_symptom_score"] += 5
        
        # Analizar síntomas moderados
        for symptom in self.medical_knowledge_base["symptoms_severity"]["moderate"]:
            if patient_data.get(symptom, 0) == 1:
                symptom_analysis["moderate_symptoms"].append(symptom)
                symptom_analysis["total_symptom_score"] += 3
        
        # Analizar síntomas leves
        for symptom in self.medical_knowledge_base["symptoms_severity"]["mild"]:
            if patient_data.get(symptom, 0) == 1:
                symptom_analysis["mild_symptoms"].append(symptom)
                symptom_analysis["total_symptom_score"] += 1
                
        return symptom_analysis
    
    def _assess_age_risk(self, age: int) -> Dict[str, Any]:
        """Evalúa el riesgo basado en la edad"""
        age_risk = {
            "age_group": "desconocido",
            "risk_level": "bajo",
            "risk_score": 0,
            "recommendations": []
        }
        
        for group, (min_age, max_age) in self.medical_knowledge_base["risk_factors"]["age_risk"].items():
            if min_age <= age <= max_age:
                age_risk["age_group"] = group
                if group == "very_high":
                    age_risk["risk_level"] = "muy alto"
                    age_risk["risk_score"] = 5
                    age_risk["recommendations"] = ["Monitoreo intensivo", "Contacto médico inmediato"]
                elif group == "high":
                    age_risk["risk_level"] = "alto"
                    age_risk["risk_score"] = 4
                    age_risk["recommendations"] = ["Monitoreo frecuente", "Contacto médico en 24h"]
                elif group == "moderate":
                    age_risk["risk_level"] = "moderado"
                    age_risk["risk_score"] = 2
                    age_risk["recommendations"] = ["Monitoreo regular"]
                elif group == "low":
                    age_risk["risk_level"] = "bajo"
                    age_risk["risk_score"] = 1
                    age_risk["recommendations"] = ["Monitoreo preventivo"]
                else:
                    age_risk["risk_level"] = "muy bajo"
                    age_risk["risk_score"] = 0
                    age_risk["recommendations"] = ["Monitoreo preventivo"]
                break
                
        return age_risk
    
    def _generate_fallback_explanation(self, risk_prediction: Dict[str, Any], 
                                     patient_data: Dict[str, Any]) -> str:
        """Genera una explicación de respaldo cuando falla la IA"""
        risk_level = risk_prediction['risk_level_text']
        probability = risk_prediction['probability']
        
        return f"""
        **Explicación Médica (Modo de Respaldo)**

        Basado en el análisis de los datos del paciente, se ha determinado un nivel de riesgo de **{risk_level}** 
        con una probabilidad del {probability:.1%}.

        **Interpretación Clínica:**
        - Nivel de Riesgo: {risk_level}
        - Confianza de la Predicción: {probability:.1%}

        **Recomendaciones Generales:**
        1. Mantener aislamiento preventivo
        2. Monitorear síntomas regularmente
        3. Contactar médico si aparecen síntomas nuevos
        4. Seguir protocolos de prevención COVID-19

        **Nota:** Esta explicación se generó en modo de respaldo. Para una explicación más detallada, 
        contacte al equipo técnico.
        """
    
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
                                  model_metrics: Dict[str, Any]) -> str:
        """
        Crea un dashboard HTML completo con explicaciones, gráficas y métricas
        """
        # Generar explicación y recomendaciones
        explanation = self.generate_intelligent_explanation(risk_prediction, patient_data, model_metrics)
        recommendations = self.generate_recommendations(risk_prediction, patient_data)
        clinical_alert = self.generate_clinical_alert(risk_prediction, patient_data)
        detailed_explanation = self.generate_detailed_explanation(risk_prediction, patient_data)
        risk_factors = self._identify_risk_factors(patient_data)
        next_steps = self._get_next_steps(risk_prediction)
        clinical_summary = self._generate_clinical_summary(risk_prediction, patient_data)

        # Crear gráficas adicionales para el dashboard
        patient_charts = self._create_patient_charts(patient_data, risk_prediction)
        model_performance_charts = self._create_model_performance_charts(model_metrics)
        risk_analysis_charts = self._create_risk_analysis_charts(patient_data, risk_prediction)

        html = f'''
        <div class="dashboard-xai">
            <!-- Métricas del Modelo -->
            <div class="model-metrics-section mb-4">
                <h5><i class="fas fa-chart-bar"></i> Métricas del Modelo (Random Forest)</h5>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{model_metrics.get('accuracy', 0):.1%}</div>
                        <div class="metric-label">Precisión</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{model_metrics.get('precision', 0):.1%}</div>
                        <div class="metric-label">Especificidad</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{model_metrics.get('recall', 0):.1%}</div>
                        <div class="metric-label">Sensibilidad</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{model_metrics.get('f1_score', 0):.1%}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{model_metrics.get('roc_auc', 0):.1%}</div>
                        <div class="metric-label">ROC-AUC</div>
                    </div>
                </div>
            </div>

            <!-- Gráficas de Rendimiento del Modelo -->
            <div class="model-charts-section mb-4">
                <h5><i class="fas fa-chart-line"></i> Rendimiento del Modelo</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <img src="/static/confusion_matrix.png" alt="Matriz de Confusión" class="img-fluid rounded shadow-sm">
                            <p class="text-center mt-2"><small>Matriz de Confusión - Evaluación del Modelo</small></p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <img src="/static/feature_importance.png" alt="Importancia de Características" class="img-fluid rounded shadow-sm">
                            <p class="text-center mt-2"><small>Importancia de Características - Random Forest</small></p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Gráficas de Análisis del Paciente -->
            <div class="patient-analysis-section mb-4">
                <h5><i class="fas fa-user-chart"></i> Análisis del Paciente</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            {patient_charts['risk_factors_chart']}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            {patient_charts['symptoms_chart']}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Gráficas de Rendimiento del Modelo -->
            <div class="performance-charts-section mb-4">
                <h5><i class="fas fa-tachometer-alt"></i> Métricas de Rendimiento</h5>
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            {model_performance_charts['metrics_radar']}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            {model_performance_charts['confidence_chart']}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Explicación Médica -->
            <div class="explanation-section mb-4">
                <h5><i class="fas fa-stethoscope"></i> Explicación Médica Inteligente</h5>
                <div class="explanation-box">{explanation}</div>
            </div>

            <!-- Recomendaciones -->
            <div class="recommendations-section mb-4">
                <h5><i class="fas fa-clipboard-list"></i> Recomendaciones Personalizadas</h5>
                <ul class="recommendations-list">
                    {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                </ul>
            </div>

            <!-- Factores de Riesgo -->
            <div class="risk-factors-section mb-4">
                <h5><i class="fas fa-exclamation-triangle"></i> Factores de Riesgo Identificados</h5>
                <ul class="recommendations-list">
                    {''.join(f'<li>{factor}</li>' for factor in risk_factors)}
                </ul>
            </div>

            <!-- Próximos Pasos -->
            <div class="next-steps-section mb-4">
                <h5><i class="fas fa-route"></i> Próximos Pasos Clínicos</h5>
                <ul class="recommendations-list">
                    {''.join(f'<li>{step}</li>' for step in next_steps)}
                </ul>
            </div>

            <!-- Resumen Clínico -->
            <div class="clinical-summary-section mb-4">
                <h5><i class="fas fa-file-medical"></i> Resumen Clínico</h5>
                <div class="clinical-summary">
                    <div class="summary-item">
                        <span class="summary-label">Mensaje Principal:</span>
                        <span class="summary-value">{clinical_summary.get('key_message','')}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Nivel de Prioridad:</span>
                        <span class="summary-value">{clinical_summary.get('priority_level','')}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Tiempo de Acción:</span>
                        <span class="summary-value">{clinical_summary.get('time_to_action','')}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Urgencia Clínica:</span>
                        <span class="summary-value">{clinical_summary.get('clinical_urgency','')}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Evolución Esperada:</span>
                        <span class="summary-value">{clinical_summary.get('expected_course','')}</span>
                    </div>
                </div>
            </div>

            <!-- Alertas Clínicas -->
            <div class="clinical-alerts-section mb-4">
                <h5><i class="fas fa-bell"></i> Alertas Clínicas</h5>
                <div class="clinical-alert {clinical_alert['severity']}">
                    <h6>{clinical_alert['message']}</h6>
                    <div class="mt-3">
                        <strong>Acciones Inmediatas:</strong>
                        <ul>
                            {''.join(f'<li>{action}</li>' for action in clinical_alert['immediate_actions'])}
                        </ul>
                    </div>
                    <div class="mt-3">
                        <strong>Señales de Alarma:</strong>
                        <ul>
                            {''.join(f'<li>{sign}</li>' for sign in clinical_alert['warning_signs'])}
                        </ul>
                    </div>
                    <div class="mt-3">
                        <strong>Notas Clínicas:</strong>
                        <p>{clinical_alert['clinical_notes']}</p>
                    </div>
                </div>
            </div>
        </div>
        '''
        return html

    def _create_patient_charts(self, patient_data: Dict[str, Any], risk_prediction: Dict[str, Any]) -> Dict[str, str]:
        """Crea gráficas específicas del paciente"""
        # Gráfica de factores de riesgo
        risk_factors = self._analyze_patient_risk_factors(patient_data)
        high_risk_count = len(risk_factors['high_risk_conditions'])
        moderate_risk_count = len(risk_factors['moderate_risk_conditions'])
        
        risk_factors_chart = f'''
        <div class="chart-card">
            <h6>Factores de Riesgo del Paciente</h6>
            <div class="risk-factors-chart">
                <div class="risk-factor-item high-risk">
                    <span class="risk-label">Alto Riesgo:</span>
                    <span class="risk-count">{high_risk_count}</span>
                </div>
                <div class="risk-factor-item moderate-risk">
                    <span class="risk-label">Riesgo Moderado:</span>
                    <span class="risk-count">{moderate_risk_count}</span>
                </div>
                <div class="risk-factor-item total-score">
                    <span class="risk-label">Score Total:</span>
                    <span class="risk-count">{risk_factors['total_risk_score']}</span>
                </div>
            </div>
        </div>
        '''
        
        # Gráfica de síntomas
        symptoms = self._analyze_symptoms(patient_data)
        critical_count = len(symptoms['critical_symptoms'])
        moderate_count = len(symptoms['moderate_symptoms'])
        mild_count = len(symptoms['mild_symptoms'])
        
        symptoms_chart = f'''
        <div class="chart-card">
            <h6>Síntomas del Paciente</h6>
            <div class="symptoms-chart">
                <div class="symptom-item critical">
                    <span class="symptom-label">Críticos:</span>
                    <span class="symptom-count">{critical_count}</span>
                </div>
                <div class="symptom-item moderate">
                    <span class="symptom-label">Moderados:</span>
                    <span class="symptom-count">{moderate_count}</span>
                </div>
                <div class="symptom-item mild">
                    <span class="symptom-label">Leves:</span>
                    <span class="symptom-count">{mild_count}</span>
                </div>
            </div>
        </div>
        '''
        
        return {
            'risk_factors_chart': risk_factors_chart,
            'symptoms_chart': symptoms_chart
        }

    def _create_model_performance_charts(self, model_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Crea gráficas de rendimiento del modelo"""
        # Gráfica de métricas en radar
        accuracy = model_metrics.get('accuracy', 0) * 100
        precision = model_metrics.get('precision', 0) * 100
        recall = model_metrics.get('recall', 0) * 100
        f1_score = model_metrics.get('f1_score', 0) * 100
        roc_auc = model_metrics.get('roc_auc', 0) * 100
        
        metrics_radar = f'''
        <div class="chart-card">
            <h6>Métricas del Modelo Random Forest</h6>
            <div class="metrics-radar">
                <div class="metric-item">
                    <span class="metric-name">Precisión</span>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {accuracy}%"></div>
                    </div>
                    <span class="metric-value">{accuracy:.1f}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">Especificidad</span>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {precision}%"></div>
                    </div>
                    <span class="metric-value">{precision:.1f}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">Sensibilidad</span>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {recall}%"></div>
                    </div>
                    <span class="metric-value">{recall:.1f}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">F1-Score</span>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {f1_score}%"></div>
                    </div>
                    <span class="metric-value">{f1_score:.1f}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-name">ROC-AUC</span>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {roc_auc}%"></div>
                    </div>
                    <span class="metric-value">{roc_auc:.1f}%</span>
                </div>
            </div>
        </div>
        '''
        
        # Gráfica de confianza
        confidence_level = self._get_confidence_level(model_metrics.get('accuracy', 0))
        confidence_chart = f'''
        <div class="chart-card">
            <h6>Nivel de Confianza del Modelo</h6>
            <div class="confidence-chart">
                <div class="confidence-circle {confidence_level.lower().replace(' ', '-')}">
                    <span class="confidence-text">{confidence_level}</span>
                </div>
                <p class="text-center mt-2">El modelo tiene un nivel de confianza <strong>{confidence_level}</strong> basado en su precisión de {accuracy:.1f}%</p>
            </div>
        </div>
        '''
        
        return {
            'metrics_radar': metrics_radar,
            'confidence_chart': confidence_chart
        }

    def _create_risk_analysis_charts(self, patient_data: Dict[str, Any], risk_prediction: Dict[str, Any]) -> Dict[str, str]:
        """Crea gráficas de análisis de riesgo"""
        # Esta función puede ser expandida para crear más gráficas específicas
        return {}
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determina el nivel de confianza basado en la probabilidad"""
        if probability >= 0.9:
            return "MUY ALTA"
        elif probability >= 0.7:
            return "ALTA"
        elif probability >= 0.6:
            return "MEDIA"
        else:
            return "BAJA"
    
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
    dashboard_agent = IntelligentDashboardAgent("your_api_key_here")
    print("Agente de dashboard inicializado")
    print(f"Niveles de riesgo: {dashboard_agent.covid_classes}") 