import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de la API de Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBLyYvFaJGWHPeHl9NH6O5Bxcq01fOdxV8")

# Configuración del modelo
MODEL_PATH = os.environ.get("MODEL_PATH", "models/covid_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")

# Configuración de la aplicación
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000)) 