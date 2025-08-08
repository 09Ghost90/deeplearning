# Arquitetura da CNN
Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool

# Fluxo da Inferencia atualmente:
Imagem -> Preprocessamento -> Modelo -> Saída -> Softmax -> Classe + Confiança

CNN -> Cria o modeo da rede neural
model.load_state_dict() -> Carregamos os pesos treinados
model.eval() -> Prepara o modelo para inferência
preprocess_imagem -> Lê, redimensiona, normaliza e transforma a imagem em tensor
predict_single_image -> Faz inferência em uma imagem e retorna classe + confiança
predict_batch -> Inferência em várias imagens e retornar lista de classes e confianças

# Next Stage 
* Implementar CIFAR-10

    Contéudo: 60k de imagens coloridas
    Dimensões: Cada imagem tem 32x32 pixels
    Classes: 10 classes de objetos, com 6k imagens por classes
    Divisão: 50k imagens para treino e 10k para teste

    Estrutura da rede CNN do MNIST
    convolução -> Pooling -> Convolução -> pooling -> fully connected
    Dropout, ReLU, MaxPooling

    Classes do CIFAR-10:
    1. airplane
    2. automobile
    3. bird
    4. cat
    5. deet
    6. dog
    7. frog
    8. horse
    9. ship
    10. truck

# Resultados
* CNN -> Utilizando optimizer Adam: 
Acurácia: 0.9858
Precisão: 0.9858
Revocação: 0.9858

CNN -> Utilizando optimizer SGD:
Acurácia: 0.9713
Precisão: 0.9713
Revocação: 0.9713

# Empregar CNN + Reinforcement Learning
    CNN -> É utilizada para extrair características de imagens (feature extrator)
    RL -> vai usar essas características como entrada para decidir ações, receber recompensas e aprender uma política ótima.

# Criar o dataset para os grãos de soja
    1. Label Studio → rotular cada grão (via bounding box + classe).

    2. Detector de objetos (YOLOv8, Faster R-CNN) para automatizar localização dos grãos.

    3. Classificador CNN para prever a classe do grão.

    4. Avaliar métricas (acurácia, recall por classe).

    5. Depois que tiver bom desempenho, integrar RL se quiser otimizar decisões no processo industrial.