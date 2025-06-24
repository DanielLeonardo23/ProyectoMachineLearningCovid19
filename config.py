import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la API de Gemini
GEMINI_API_KEY = "AIzaSyBLyYvFaJGWHPeHl9NH6O5Bxcq01fOdxV8"

# Configuración del modelo
MODEL_PATH = "models/covid_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Configuración de la aplicación
DEBUG = True
HOST = "0.0.0.0"
PORT = 5000 