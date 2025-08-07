import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy, Precision, Recall


# Hiperparâmetros
batch_size = 60
num_epochs = 10
num_class = 10
PATH = "./DeepLearning"

train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Adicionar teste de Validação para verificar o Overfitting e outros parâmetros importantes

# in_channels=1 -> Indica a quantida de e canais que serão enviados
# num_classes -> Define o número de categorias que o modelo deve aprender a classificar

# Criar um lote aleatório de dígitos:
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis('off')
    plt.savefig('lote_mnist.png')
    print("Imagem salva como 'lote_mnist.png'")

# Pegar Imagens de treinamento aleatoriamente
dataiter = iter(train_loader)
images, labels = next(dataiter)

print(labels)

# Mostrar Imagens
imshow(torchvision.utils.make_grid(images))

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network

        Parameters:
            * in_channels: Number of Channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from 0 to 9).
        """

        super(CNN, self).__init__()

        # Primeira Camada Convolucional
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        # Adicionando após a camada FC1 um Dropout para evitar Overfitting:
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Define the forward pass of the neural network

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network
        """
        
        # Chamar funções para o modelo CNN ir rodando:
        x = F.relu(self.conv1(x)) # Aplicar a primeira camada convolucional e ReLU ativação
        x = self.pool(x) # Aplicando Max Pooling
        x = F.relu(self.conv2(x)) # Aplicando a segunda camada convolucional e ReLU ativação
        x = self.pool(x) # Aplicando o Max Pooling novamente 
        x = x.reshape(x.shape[0], -1) # Transformar o tensor multidimensional em um tensor unidimensional
        x = self.fc1(x) # Aplicar fully connected layer
        x = self.dropout(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=1, num_classes=10).to(device)
print(model)

# Treinamento do Modelo CNN

# Define Loss Function:
criterion = nn.CrossEntropyLoss()
# Define o Optimezer -> Cálculo 3 (∇Gradiente∇):
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

# Avaliação do Modelo
# Torchmetrics
acc = Accuracy(task="multiclass", num_classes=10) # Percentual de Acerto
precision = Precision(task="multiclass", num_classes=10) # Precisão - Quantas vezes ele estava correto
recall = Recall(task="multiclass", num_classes=10) # Dentre todos os "7" reais, quantos o modelo acertou?

# Interando sobre o dataset batches
model.eval() # Modelo no modo avaliação
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        acc.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)

# Resultados
print(f"Acurácia: {acc.compute():.4f}")
print(f"Precisão: {precision.compute():.4f}")
print(f"Revocação: {recall.compute():.4f}")

# Salvar os pesos
torch.save(model.state_dict(), PATH)

"""
Trocar Adam por SGD com momentum para comparar o resultados
Técnica weight decay (regularização L2)
Treinar com outras bases -> FashionMNIST, CIFAR-10 e comparar o desemepnho
Salvar checkpoints (torch.save) e (torch.load)
"""