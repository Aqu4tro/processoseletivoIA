# Processo Seletivo – Intensivo Maker | AI

### 📝 Relatório do Candidato

👤 **Identificação:** Jonathas Levi Pascoal Palmeira

#### 1️⃣ Resumo da Arquitetura do Modelo
Foi projetada uma Rede Neural Convolucional (CNN) leve, estritamente pensada para restrições de memória em sistemas embarcados (Edge AI). A arquitetura é composta por:
* **Camadas de Entrada e Extração:** Uma camada de entrada explícita `Input(shape=(28, 28, 1))` seguida de duas camadas `Conv2D` (com 16 e 32 filtros, respectivamente, kernel 3x3 e ativação ReLU), intercaladas por camadas `MaxPooling2D` (2x2) para redução de dimensionalidade espacial.
* **Camadas de Classificação:** Uma camada `Flatten` seguida de uma camada `Dense` oculta de 64 neurônios (ReLU) e a camada de saída `Dense` com 10 neurônios (Softmax) para as 10 classes do MNIST.

O modelo foi compilado utilizando o otimizador Adam e treinado em mini-batches de 64 por 5 épocas, garantindo uma rápida convergência compatível com as restrições da esteira de CI.

#### 2️⃣ Bibliotecas Utilizadas
* **TensorFlow / Keras (v2.x):** Framework principal utilizado para construção, treinamento e conversão do modelo.
* **NumPy:** Utilizado para manipulação eficiente dos arrays e para a construção da função geradora (*Representative Dataset*) exigida na quantização avançada.
* **OS:** Utilizado para extrair as métricas de tamanho de arquivo (em KB) no comparativo de otimização.

#### 3️⃣ Técnica de Otimização do Modelo
Para demonstrar aprofundamento em estratégias de Edge AI, o script `optimize_model.py` explora e compara **duas técnicas distintas** de otimização nativas do TensorFlow Lite:
1.  **Dynamic Range Quantization:** Converte os pesos para Int8, mas executa inferência em Float32. Excelente compromisso de compressão para placas que suportam ponto flutuante via software. *Esta foi a técnica escolhida para exportação (`model.tflite`) visando total compatibilidade com o pipeline de correção automatizada.*
2.  **Full Integer Quantization (Avançada):** Transforma entrada, pesos, cálculos e saídas estritamente em Int8 (8 bits), utilizando um *Representative Dataset* com 100 amostras para calibrar as ativações. É a técnica definitiva para microcontroladores puros sem unidade de ponto flutuante (FPU).

#### 4️⃣ Resultados Obtidos
O modelo base em ponto flutuante (Float32) alcançou uma acurácia final de aproximadamente **98.6%** no conjunto de testes (podendo variar levemente entre execuções devido à inicialização aleatória dos pesos).

Além da acurácia, foram implementadas **múltiplas métricas de eficiência** (Tamanho Absoluto e Taxa de Compressão), comparando as duas técnicas em relação ao modelo original (`.h5` - 703.94 KB):
* **Full Integer Quantization (Int8):** Reduziu o tamanho para **63.38 KB**, alcançando uma taxa de compressão impressionante de **11.11x**.
* **Dynamic Range Quantization:** Reduziu o tamanho para **63.54 KB**, alcançando uma taxa de compressão de **11.08x**.

O formato TFLite em conjunto com as técnicas de quantização eliminou todo o *overhead* do HDF5. Espera-se uma degradação quase nula na acurácia real de inferência no dispositivo final, configurando um *trade-off* excelente para cenários de processamento e armazenamento altamente restritos.

#### 5️⃣ Comentários Adicionais
* **Decisão Técnica de Engenharia:** A escolha por reduzir o número de filtros (16 e 32) nas camadas convolucionais foi intencional, pois arquiteturas profundas geram *overkill* computacional no MNIST. Além disso, a conversão da técnica "Full Integer" foi intencionalmente processada apenas na memória RAM do script. Isso permitiu demonstrar domínio técnico avançado no cálculo das métricas comparativas sem violar as restrições rígidas da estrutura de arquivos do repositório (que exige apenas um arquivo `model.tflite`).
* **Ambiente:** O desenvolvimento foi isolado através de um ambiente Docker (Dev Container), garantindo padronização de dependências e total compatibilidade com o pipeline automatizado via GitHub Actions.
****
