# Processo Seletivo – Intensivo Maker | AI

## 📝 Relatório do Candidato

👤 **Identificação:** Jonathas Levi Pascoal Palmeira

### 1️⃣ Resumo da Arquitetura do Modelo

Foi projetada uma Rede Neural Convolucional (CNN) leve, estritamente pensada para restrições de memória em sistemas embarcados (Edge AI). A arquitetura é composta por:
- **Camadas de Extração de Características:** Duas camadas `Conv2D` (com 16 e 32 filtros, respectivamente, kernel 3x3 e ativação ReLU), intercaladas por camadas `MaxPooling2D` (2x2) para redução de dimensionalidade espacial.
- **Camadas de Classificação:** Uma camada `Flatten` seguida de uma camada `Dense` oculta de 64 neurônios (ReLU) e a camada de saída `Dense` com 10 neurônios (Softmax) para as 10 classes do MNIST.
O modelo foi compilado utilizando o otimizador Adam e treinado em mini-batches de 64 por 5 épocas, garantindo uma rápida convergência compatível com as restrições da esteira de CI.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow / Keras (v2.x):** Framework principal utilizado para construção, treinamento e conversão do modelo.
- **NumPy:** Utilizado para manipulação eficiente dos arrays e redimensionamento dos dados de entrada (adição do canal de cor).

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada a técnica de **Dynamic Range Quantization** (Quantização de Faixa Dinâmica) nativa do TensorFlow Lite.
Essa técnica converte os pesos do modelo treinado de ponto flutuante de 32 bits (Float32) para inteiros de 8 bits (Int8) pós-treinamento. Isso reduz o tamanho do modelo armazenado em aproximadamente 4 vezes e acelera a inferência em microcontroladores sem unidade de ponto flutuante (FPU) dedicada, mantendo uma degradação quase nula na acurácia.

### 4️⃣ Resultados Obtidos

O modelo base em ponto flutuante (Float32) alcançou uma acurácia final de aproximadamente **98.91%** no conjunto de testes. 

O processo fluiu com sucesso desde o treinamento em CPU até a conversão para `.tflite`. Como foi utilizada a quantização para Int8, espera-se uma degradação mínima (fração de porcentagem) na acurácia real de inferência no dispositivo final, o que é um *trade-off* excelente considerando a redução brutal no consumo de memória e armazenamento.

### 5️⃣ Comentários Adicionais

- **Decisão Técnica:** A escolha por reduzir o número de filtros (16 e 32) nas camadas convolucionais foi intencional. Em problemas relativamente simples como o MNIST, arquiteturas muito profundas geram *overkill* computacional. A abordagem enxuta garante viabilidade de deploy em hardwares restritos, como a família ESP32 ou ARM Cortex-M.
- **Ambiente:** O desenvolvimento foi isolado através de um ambiente Docker (Dev Container), garantindo padronização de dependências e total compatibilidade com o pipeline automatizado via GitHub Actions.

****
