import requests
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# Definición Ampliada de Fincas y Localidades en Mendoza
# ==========================================
FINCAS_MENDOZA = {
    'Gualtallary (Tupungato)': {
        'lat': -33.3833, 
        'lon': -69.2167, 
        'umbral': 1.0, 
        'cultivo': 'Vid / Malbec', 
        'localidad': 'Tupungato, Valle de Uco',
        'viento_limite': 10.0
    },
    'La Consulta (San Carlos)': {
        'lat': -33.7333, 
        'lon': -69.1000, 
        'umbral': 0.0, 
        'cultivo': 'Vid / Altamira - Frutales', 
        'localidad': 'San Carlos, Valle de Uco',
        'viento_limite': 8.0
    },
    'El Cepillo (San Carlos)': {
        'lat': -33.7833, 
        'lon': -68.9333, 
        'umbral': 0.0, 
        'cultivo': 'Vid / Frutales', 
        'localidad': 'San Carlos, Valle de Uco',
        'viento_limite': 8.0
    },
    'Agrelo (Luján de Cuyo)': {
        'lat': -33.1167, 
        'lon': -68.8833, 
        'umbral': 0.5, 
        'cultivo': 'Vid / Cabernet Sauvignon', 
        'localidad': 'Luján de Cuyo',
        'viento_limite': 10.0
    },
    'Perdriel (Luján Sur)': {
        'lat': -33.0500, 
        'lon': -68.8833, 
        'umbral': 0.5, 
        'cultivo': 'Vid / Malbec Antiguo', 
        'localidad': 'Luján de Cuyo',
        'viento_limite': 10.0
    },
    'Barrancas (Maipú)': {
        'lat': -33.0167, 
        'lon': -68.6167, 
        'umbral': 1.0, 
        'cultivo': 'Vid / Syrah - Olivos', 
        'localidad': 'Maipú, Oasis Norte',
        'viento_limite': 11.0
    },
    'Rivadavia Oasis Norte': {
        'lat': -33.1833, 
        'lon': -68.4667, 
        'umbral': 1.5, 
        'cultivo': 'Hortalizas / Tomate', 
        'localidad': 'Rivadavia, Este',
        'viento_limite': 12.0
    },
    'Alto Verde (San Martín)': {
        'lat': -33.0833, 
        'lon': -68.4833, 
        'umbral': 1.0, 
        'cultivo': 'Vid / Frutales de mesa', 
        'localidad': 'San Martín, Este',
        'viento_limite': 11.0
    },
    'Cuadro Benegas (San Rafael)': {
        'lat': -34.6333, 
        'lon': -68.4500, 
        'umbral': -0.5, 
        'cultivo': 'Frutales de carozo / Ciruela', 
        'localidad': 'San Rafael, Oasis Sur',
        'viento_limite': 9.0
    },
    'General Alvear Sur': {
        'lat': -34.9833, 
        'lon': -67.6833, 
        'umbral': 0.0, 
        'cultivo': 'Frutales / Durazno', 
        'localidad': 'General Alvear, Oasis Sur',
        'viento_limite': 9.0
    }
}

# ==========================================
# Generador de datos sintéticos de respaldo (Fallback)
# ==========================================
def _generar_datos_fallback():
    fechas = pd.date_range(end=pd.Timestamp.now(), periods=168, freq='h')
    n = len(fechas)
    horas = fechas.hour.values
    
    # Simulación de patrón térmico diurno/nocturno
    temp_base = 12 + 8 * np.sin((horas - 8) * np.pi / 12) + np.random.normal(0, 0.5, n)
    hum_base = 50 - 30 * np.sin((horas - 8) * np.pi / 12) + np.random.normal(0, 2, n)
    punto_rocio = temp_base - ((100 - np.clip(hum_base, 10, 100)) / 5)
    vel_viento = np.abs(np.random.normal(8, 3, n))
    dir_viento = np.random.uniform(0, 360, n)
    temp_suelo = temp_base + np.random.normal(1, 0.5, n)
    hum_suelo = np.full(n, 0.22)
    nubosidad = np.clip(np.random.normal(30, 25, n), 0, 100)
    evapotranspiracion = np.clip(0.1 * np.sin((horas - 6) * np.pi / 12), 0, 0.8)
    radiacion_solar = np.clip(800 * np.sin((horas - 6) * np.pi / 12), 0, 1000)

    return pd.DataFrame({
        'fecha_hora': fechas.strftime('%Y-%m-%dT%H:%M'),
        'temp_aire': np.round(temp_base, 1),
        'humedad': np.round(np.clip(hum_base, 10, 100), 1),
        'punto_rocio': np.round(punto_rocio, 1),
        'vel_viento': np.round(vel_viento, 1),
        'dir_viento': np.round(dir_viento, 1),
        'temp_suelo': np.round(temp_suelo, 1),
        'hum_suelo': np.round(hum_suelo, 2),
        'nubosidad': np.round(nubosidad, 1),
        'evapotranspiracion': np.round(evapotranspiracion, 2),
        'radiacion_solar': np.round(radiacion_solar, 1)
    })

# ==========================================
# Función robusta para consumir Open-Meteo
# ==========================================
def obtener_datos_clima(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,wind_speed_10m,wind_direction_10m,soil_temperature_0cm,soil_moisture_0_to_1cm,cloud_cover,evapotranspiration,shortwave_radiation",
        "timezone": "America/Argentina/Mendoza"
    }
    
    # Cabeceras personalizadas para evitar bloqueos en Render/Cloud
    headers = {
        "User-Agent": "MonitorHeladasMZ/2.4 (AgroMeteorology DSS Mendoza; contact@monitor-heladas-mz.org)"
    }
    
    # Intentar hasta 3 veces con tiempo de espera de 15 segundos
    for intento in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if "hourly" in data:
                    hourly = data["hourly"]
                    df = pd.DataFrame({
                        'fecha_hora': hourly.get("time", []),
                        'temp_aire': hourly.get("temperature_2m", []),
                        'humedad': hourly.get("relative_humidity_2m", []),
                        'punto_rocio': hourly.get("dew_point_2m", []),
                        'vel_viento': hourly.get("wind_speed_10m", []),
                        'dir_viento': hourly.get("wind_direction_10m", []),
                        'temp_suelo': hourly.get("soil_temperature_0cm", []),
                        'hum_suelo': hourly.get("soil_moisture_0_to_1cm", []),
                        'nubosidad': hourly.get("cloud_cover", []),
                        'evapotranspiracion': hourly.get("evapotranspiration", []),
                        'radiacion_solar': hourly.get("shortwave_radiation", [])
                    })
                    if not df.empty:
                        return df.fillna(0)
        except Exception as e:
            print(f"[ADVERTENCIA] Intento {intento + 1} fallido al consultar API: {e}")
            
    print("[INFO] No se pudo obtener respuesta de la API. Usando datos climatológicos simulados.")
    return _generar_datos_fallback()

# ==========================================
# Backend Analítico y Deep Learning Optimizado (6 Variables)
# ==========================================
def ejecutar_backend_analitico(df, umbral):
    if df.empty:
        df = _generar_datos_fallback()

    # 1. Tendencia estadística
    x = np.arange(len(df))
    y = df['temp_aire'].values
    grado = min(3, len(x) - 1) if len(x) > 3 else 1
    
    if len(x) > 3:
        coefs = np.polyfit(x, y, grado)
        df['temp_tendencia_estadistica'] = np.polyval(coefs, x)
    else:
        df['temp_tendencia_estadistica'] = y

    # 2. Aprendizaje Profundo (MLPClassifier)
    X_train_base = np.array([
        [-3.0, -5.0, 1.5,  0.0, 0.15, 0.0],   # Helada radiativa severa
        [-1.0, -3.0, 2.0, 10.0, 0.20, 0.0],   # Helada moderada
        [ 0.0, -1.0, 3.0, 20.0, 0.25, 10.0],  # Helada leve
        [ 2.0, -1.5, 4.0, 30.0, 0.30, 50.0],  # Sin riesgo directo
        [ 5.0,  1.0, 8.0, 75.0, 0.45, 200.0], # Nublado / Viento
        [12.0,  5.0, 6.0, 50.0, 0.40, 500.0], # Condición templada
        [18.0,  9.0, 4.0, 20.0, 0.35, 800.0], # Día soleado
        [-0.5, -2.0, 1.0,  5.0, 0.18, 0.0]    # Helada radiativa
    ])
    y_train_base = np.array([1, 1, 1, 0, 0, 0, 0, 1])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)

    mlp = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train_base)

    X_real = df[['temp_aire', 'punto_rocio', 'vel_viento', 'nubosidad', 'hum_suelo', 'radiacion_solar']].values
    X_real_scaled = scaler.transform(X_real)

    try:
        probabilidades = mlp.predict_proba(X_real_scaled)[:, 1] * 100
    except Exception:
        probabilidades = np.where(df['temp_aire'] <= umbral, 90.0, 10.0)

    df['prob_helada_dl'] = probabilidades
    return df