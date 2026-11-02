import joblib
import numpy as np
import pandas as pd
import requests
from requests_cache import CachedSession
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Configurar sesión con caché (expira en 15 minutos) para evitar errores 429 de Open-Meteo
session = CachedSession('openmeteo_cache', expire_after=900)

# ==========================================
# 1. CONFIGURACIÓN DE LOCALIDADES Y FINCAS (MENDOZA)
# ==========================================
FINCAS_MENDOZA = {
    'Gualtallary (Tupungato)': {
        'lat': -33.38,
        'lon': -69.18,
        'cultivo': 'Malbec (Brotación)',
        'umbral': -1.5,
        'localidad': 'Valle de Uco',
    },
    'Vista Flores (San Carlos)': {
        'lat': -33.65,
        'lon': -69.15,
        'cultivo': 'Chardonnay (Hojas)',
        'umbral': -1.0,
        'localidad': 'Valle de Uco',
    },
    'Los Chacayes (Tunuyán)': {
        'lat': -33.68,
        'lon': -69.31,
        'cultivo': 'Cabernet Franc (Brotación)',
        'umbral': -1.8,
        'localidad': 'Valle de Uco',
    },
    'Agrelo (Luján de Cuyo)': {
        'lat': -33.11,
        'lon': -68.89,
        'cultivo': 'Cabernet Sauv. (Yema)',
        'umbral': -3.0,
        'localidad': 'Primera Zona',
    },
    'Perdriel (Maipú)': {
        'lat': -33.07,
        'lon': -68.88,
        'cultivo': 'Malbec (Brotación)',
        'umbral': -1.8,
        'localidad': 'Primera Zona',
    },
    'Vistalba (Luján de Cuyo)': {
        'lat': -33.03,
        'lon': -68.91,
        'cultivo': 'Malbec (Floración)',
        'umbral': -2.0,
        'localidad': 'Primera Zona',
    },
    'Costa de Araujo (Lavalle)': {
        'lat': -32.74,
        'lon': -68.32,
        'cultivo': 'Uva de Mesa (Yema)',
        'umbral': -2.5,
        'localidad': 'Oasis Este',
    },
    'Alto Verde (San Martín)': {
        'lat': -33.08,
        'lon': -68.46,
        'cultivo': 'Bonarda (Brotación)',
        'umbral': -1.5,
        'localidad': 'Oasis Este',
    },
    'Las Catitas (Santa Rosa)': {
        'lat': -33.27,
        'lon': -68.04,
        'cultivo': 'Criolla (Hojas)',
        'umbral': -1.2,
        'localidad': 'Oasis Este',
    },
    'Rivadavia (Centro)': {
        'lat': -33.18,
        'lon': -68.47,
        'cultivo': 'Barbera (Brotación)',
        'umbral': -1.5,
        'localidad': 'Oasis Este',
    },
    'Rama Caída (San Rafael)': {
        'lat': -34.69,
        'lon': -68.38,
        'cultivo': 'Ciruelo (Floración)',
        'umbral': -2.2,
        'localidad': 'Oasis Sur',
    },
    'Ciudad (General Alvear)': {
        'lat': -34.96,
        'lon': -67.69,
        'cultivo': 'Durazno (Floración)',
        'umbral': -2.5,
        'localidad': 'Oasis Sur',
    },
    'Real del Padre (San Rafael)': {
        'lat': -34.48,
        'lon': -67.92,
        'cultivo': 'Peral (Yema)',
        'umbral': -2.0,
        'localidad': 'Oasis Sur',
    },
}

# ==========================================
# 2. CARGA DEL MODELO DE MACHINE LEARNING
# ==========================================
try:
  modelo_heladas = joblib.load('modelo_heladas_sk.pkl')
  print('Modelo Scikit-Learn cargado correctamente.')
except Exception as e:
  modelo_heladas = None
  print(f'Aviso: No se pudo cargar el modelo de ML: {e}')


# ==========================================
# 3. FUNCIÓN DE EXTRACCIÓN (ETL) - 4 DÍAS
# ==========================================
def obtener_datos_clima(lat, lon):
  url = 'https://api.open-meteo.com/v1/forecast'
  params = {
      'latitude': lat,
      'longitude': lon,
      'hourly': [
          'temperature_2m',
          'relative_humidity_2m',
          'dew_point_2m',
          'cloud_cover',
          'wind_speed_10m',
          'wind_direction_10m',
          'soil_temperature_0cm',
          'soil_moisture_0_to_1cm',
          'et0_fao_evapotranspiration',
      ],
      'timezone': 'America/Argentina/Mendoza',
      'forecast_days': 4,
  }

  response = session.get(url, params=params)
  if response.status_code == 429:
    raise Exception(
        'Límite de solicitudes superado en Open-Meteo (Error 429). Espere'
        ' unos minutos.'
    )
  if response.status_code != 200:
    raise Exception(f'Error en API Open-Meteo. Código: {response.status_code}')

  data = response.json()
  hourly_data = data['hourly']

  df = pd.DataFrame({
      'fecha_hora': pd.to_datetime(hourly_data['time']),
      'temp_aire': hourly_data['temperature_2m'],
      'humedad': hourly_data['relative_humidity_2m'],
      'punto_rocio': hourly_data['dew_point_2m'],
      'nubosidad': hourly_data['cloud_cover'],
      'vel_viento': hourly_data['wind_speed_10m'],
      'dir_viento': hourly_data['wind_direction_10m'],
      'temp_suelo': hourly_data['soil_temperature_0cm'],
      'hum_suelo': hourly_data['soil_moisture_0_to_1cm'],
      'evapotranspiracion': hourly_data['et0_fao_evapotranspiration'],
  })
  return df


# ==========================================
# 4. BACKEND ANALÍTICO (ESTADÍSTICA + ML)
# ==========================================
def ejecutar_backend_analitico(df, umbral):
  # Pronóstico estadístico (Tendencia polinómica)
  df['t_index'] = np.arange(len(df))
  X = df[['t_index']]
  y = df['temp_aire']

  poly_model = make_pipeline(PolynomialFeatures(4), LinearRegression())
  poly_model.fit(X, y)
  df['temp_tendencia_estadistica'] = poly_model.predict(X)

  # Inferencia predictiva con Scikit-Learn
  features = np.column_stack([
      np.clip((df['temp_aire'] - 15) / 15, -1, 1),
      np.clip(df['humedad'] / 100, 0, 1),
      np.clip(df['vel_viento'] / 20, 0, 1),
      np.clip(df['nubosidad'] / 100, 0, 1),
  ])

  if modelo_heladas is not None:
    try:
      probabilidades = modelo_heladas.predict_proba(features)
      df['prob_helada_dl'] = probabilidades[:, 1] * 100
    except Exception:
      df['prob_helada_dl'] = np.clip((umbral - df['temp_aire']) * 20, 0, 100)
  else:
    df['prob_helada_dl'] = np.clip((umbral - df['temp_aire']) * 20, 0, 100)

  return df