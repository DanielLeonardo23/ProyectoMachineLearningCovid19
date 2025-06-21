import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la API de Gemini
GEMINI_API_KEY = "AIzaSyBnHbrlnHk5shVeULXmWQRtIMjPh1tXpYU"

# Configuración del modelo
MODEL_PATH = "models/covid_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Configuración de la aplicación
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000 