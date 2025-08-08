import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from cifar import CNN, PATH

in_channels = 3
num_classes = 10
model = CNN(in_channels, num_classes)

if os.path.exists(PATH):
    print(f"Arquivo de pesos encontrado. Carregando...")
    model.load_state_dict(torch.load(PATH))
else:
    raise FileNotFoundError(f"Arquivo de pesos não encontrado: {PATH}")

# Colocar o modelo em modo de avaliação para inferência
model.eval()

def preprocess_image(image_path):
    """
    Imagem para Inferencia
    """
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normaliza o RGB
    ])
    
    image = Image.open(image_path)
    image = image.convert('RGB')
    image_tensor = transform(image)
    return image_tensor

def predict_single_image(model, image_path):
    """
    Faz predição para uma única imagem
    """
    # Processar a imagem
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.unsqueeze(0)
    
    # Inferência
    with torch.no_grad():
        output = model(input_tensor)
        
        predicted_class = torch.argmax(output,dim=1).item()
        
        # Obter probabilidades (softmax)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()
        
    return predicted_class, confidence

"""def predict_batch(model, image_paths):
    batch_tensors = []
    
    for image_path in image_paths:
        tensor = preprocess_image(image_path)
        batch_tensors.append(tensor.squeeze(0))
        
    # Criar um batch 
    batch_tensor = torch.stack(batch_tensors)
    
    # Fazer a inferência no batch
    with torch.no_grad():
        outputs = model(batch_tensor)
        predict_classes = torch.argmax(outputs, dim=1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
        
    return predict_classes.tolist(), confidences.tolist"""()

if __name__ == "__main__":
    image_path = "./img.png"
    predicted_class, confidence = predict_single_image(model, image_path)
    print(f"Classe predita: {predicted_class}, Confiança: {confidence:.4f}")
    
    """
    Adicionar: Testar com conjunto maior de imagem
    """
    