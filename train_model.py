import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("\n==============================\n")

print("Carregando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normaliza os pixels (0 a 1) para acelerar a convergência, ajuda ao programa aprender mais rápido
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Adiciona a dimensão do canal (necessário para a Conv2D)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("Preparando o modelo...")
# Arquitetura 
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax") # 10 classes (0-9)
])

# Compila o modelo com otimizador e função de perda
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Treinando o modelo...")
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Avaliação final no conjunto de teste
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[!] Acurácia final no teste: {test_acc:.4f}")

# Salva o modelo
model.save("model.h5")
print("Modelo salvo como 'model.h5'")

print("\n==============================\n")