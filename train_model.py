import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("\n======================================================\n")

# --- Preparação dos Dados ---
print("Carregando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalização para escala [0, 1] auxilia na estabilidade do gradiente e convergência
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionamento para (28, 28, 1) conforme exigido por camadas Convolucionais
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# --- Definição da Arquitetura ---
print("Preparando o modelo Edge AI...")

# Optamos por uma arquitetura enxuta (16/32 filtros) para garantir viabilidade em microcontroladores
model = models.Sequential([
    # Camada Input explícita: Prática recomendada para evitar warnings de compatibilidade
    layers.Input(shape=(28, 28, 1)),
    
    # Bloco 1: Extração de características simples
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    # Bloco 2: Aumento da profundidade de filtros com redução espacial
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    
    # Cabeçalho de Classificação: Flatten seguido de densas para decisão final
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax") 
])

# Configuração do processo de aprendizado
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --- Ciclo de Treinamento ---
print("Treinando o modelo...")

# 5 épocas são suficientes para >98% de acurácia no MNIST sem risco de overfitting severo
model.fit(
    x_train, y_train,
    epochs=5,      
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


# --- Avaliação e Exportação ---
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n[!] Acurácia final no teste: {test_acc:.4f}")

# O formato .h5 é mantido para total compatibilidade com o pipeline de CI do projeto
model.save("model.h5")

print("Modelo salvo com sucesso: 'model.h5'")
print("\n======================================================\n")