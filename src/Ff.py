import tensorflow as tf
import numpy as np

# Cargar el modelo entrenado
modelo_path = "trained_model.keras"
model = tf.keras.models.load_model(modelo_path)

# Función para normalizar valores (asegúrate de usar la misma normalización que en el entrenamiento)
def normalizar(valores):
    # Ejemplo: normalización mínima-máxima (ajusta según tu caso)
    min_val = 0.0
    max_val = 1.0
    return (valores - min_val) / (max_val - min_val)

# Función para ejecutar el modelo y mostrar valores intermedios
def ejecutar_modelo_con_detalles(valores_entrada):
    """
    Ejecuta el modelo cargado con los
