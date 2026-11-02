import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generar datos históricos simulados para Mendoza
np.random.seed(42)
X_train = np.random.rand(2000, 4)  # [temp, humedad, viento, nubosidad]
y_train = ((X_train[:, 0] < 0.4) & (X_train[:, 3] < 0.3)).astype(
    int
)  # Regla de helada

# Entrenar modelo clásico de Scikit-Learn
modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_rf.fit(X_train, y_train)

# Guardar modelo en formato binario ultraliviano (.pkl)
joblib.dump(modelo_rf, 'modelo_heladas_sk.pkl')
print('Modelo Scikit-Learn guardado exitosamente como modelo_heladas_sk.pkl')