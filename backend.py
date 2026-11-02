import numpy as np
import pandas as pd
import requests
import tflite_runtime.interpreter as tflite

# Cargar el modelo optimizado de TFLite (mucho más ligero que tensorflow completo)
interpreter = tflite.Interpreter(model_path="modelo_heladas_mendoza.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predecir_con_tflite(features_array):
  # Inferencia eficiente con TFLite
  input_data = np.array(features_array, dtype=np.float32)
  interpreter.set_tensor(input_details[0]["index"], input_data)
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]["index"])
  return output_data.flatten()