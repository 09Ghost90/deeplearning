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

# Resultados
* CNN -> Utilizando optimizer Adam: 
Acurácia: 0.9858
Precisão: 0.9858
Revocação: 0.9858

CNN -> Utilizando optimizer SGD:
Acurácia: 0.9858
Precisão: 0.9858
Revocação: 0.9858