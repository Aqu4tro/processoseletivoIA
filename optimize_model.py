import tensorflow as tf
import os

print("\n==============================\n")

# Otimiza o modelo para Edge AI usando TensorFlow Lite
print("Carregando modelo...")
model = tf.keras.models.load_model('model.h5')

# Configura o conversor TFLite, a partir do modelo Keras carregado
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aplica Dynamic Range Quantization
# Faz a converção dos pesos de Float32 para Int8, reduzindo o tamanho drasticamente
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Convertendo e quantizando...")
tflite_model = converter.convert()

# Salvando o binário otimizado
tflite_filename = 'model.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"O arquivo {tflite_filename} foi salvo e está pronto para implantação.\n")

# Exibe a diferença de tamanho entre os modelos original e otimizado
print("Comparando Tamanhos e verificando otimização")
size_h5 = os.path.getsize('model.h5') / 1024
size_tflite = os.path.getsize(tflite_filename) / 1024

print(f"Modelo Original (.h5):     {size_h5:.2f} KB")
print(f"Modelo Otimizado (.tflite): {size_tflite:.2f} KB")

print("\n==============================\n")

