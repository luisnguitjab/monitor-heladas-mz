import joblib
import numpy as np
import pandas as pd
import requests

# Cargar modelo ultraliviano de Scikit-Learn
try:
  modelo_heladas = joblib.load('modelo_heladas_sk.pkl')
  print('Modelo Scikit-Learn cargado correctamente en el backend.')
except Exception as e:
  print(f'Aviso: No se pudo cargar el modelo de ML. Error: {e}')


def ejecutar_backend_analitico(df, umbral):
  # Extraer características meteorológicas
  features = np.column_stack([
      np.clip((df['temp_aire'] - 15) / 15, -1, 1),
      np.clip(df['humedad'] / 100, 0, 1),
      np.clip(df['vel_viento'] / 20, 0, 1),
      np.clip(df['nubosidad'] / 100, 0, 1),
  ])

  # Probabilidad de clase positiva (riesgo de helada)
  probabilidades = modelo_heladas.predict_proba(features)
  df['prob_helada_dl'] = probabilidades[:, 1] * 100

  return df