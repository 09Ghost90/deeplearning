import os
import torch
from main import CNN, PATH

# Carregando o modelo para inferência
# Os parãmetros precisam ser os mesmos passados usados no treinamento
in_channels = 1
num_classes = 10
model = CNN(in_channels, num_classes)

if os.path.exists(PATH):
    print(f"Arquivo de pesos encontrado. Carregando...")
    model.load_state_dict(torch.load(PATH))

model.load_state_dict(torch.load(PATH))

# Colocar o modelo em modo de avaliação para inferência
model.eval()

print(f"Modelo carregado com sucesso para inferência")