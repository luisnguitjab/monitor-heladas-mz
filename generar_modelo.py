import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Generar datos de entrenamiento simulados para el agrosistema de Mendoza
np.random.seed(42)
X_train = np.random.rand(2000, 4)  # [Temp, Humedad, Viento, Nubosidad]
y_train = ((X_train[:, 0] < 0.4) & (X_train[:, 3] < 0.3)).astype(int)

# Entrenar modelo clásico de Scikit-Learn
modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_rf.fit(X_train, y_train)

# Guardar el archivo serializado listo para producción
joblib.dump(modelo_rf, 'modelo_heladas_sk.pkl')
print(
    '¡Archivo modelo_heladas_sk.pkl generado con éxito! Súbelo a tu repositorio'
    ' de GitHub.'
)