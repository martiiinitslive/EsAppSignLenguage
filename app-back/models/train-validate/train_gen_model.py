"""
Script de entrenamiento para TextToDictaModel.
Debes adaptar la carga de datos a tu dataset de imágenes de dictadología.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from models.generator_model import TextToDictaModel

# Configuración
DATASET_DIR = 'data/dictadologia'  # Cambia esta ruta a tu dataset
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
EMBEDDING_DIM = 32
VOCAB = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'  # Incluye todas las letras de tu dataset

# Verificar distribución de imágenes por carpeta
for letter in VOCAB:
    letter_dir = os.path.join(DATASET_DIR, letter)
    if os.path.exists(letter_dir):
        imgs = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{letter}: {len(imgs)} imágenes")
    else:
        print(f"{letter}: carpeta no existe")


# Dataset personalizado
class DictaDataset(Dataset):
    def __init__(self, data_dir, vocab, img_size):
        self.data = []
        self.vocab = vocab
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        # Adaptado a la estructura: data/dictadologia/A/*.png, B/*.png, ...
        for idx, letter in enumerate(vocab):
            letter_dir = os.path.join(data_dir, letter)
            if not os.path.exists(letter_dir):
                continue
            for img_name in os.listdir(letter_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(letter_dir, img_name)
                    self.data.append((idx, img_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, img_path = self.data[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return torch.tensor([label]), image


# Cargar datos y dividir en entrenamiento y validación (80/20)
from torch.utils.data import random_split
full_dataset = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo
model = TextToDictaModel(vocab_size=len(VOCAB), embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Entrenamiento
train_losses = []
val_losses = []

import matplotlib.pyplot as plt

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (labels, images) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(labels)
        # Ajusta el shape para comparar correctamente
        if outputs.shape != images.shape:
            # Si outputs es [batch, 64, 64] y images es [batch, 1, 64, 64], elimina el canal
            images = images.squeeze(1)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss/len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}')

    # Validación simple
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for labels, images in val_loader:
            outputs = model(labels)
            if outputs.shape != images.shape:
                images = images.squeeze(1)
            loss = criterion(outputs, images)
            val_loss += loss.item()
    avg_val_loss = val_loss/len(val_loader) if len(val_loader) > 0 else 0
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}')

    # Guardar la gráfica de seguimiento en cada epoch
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evolución de la pérdida modelo de texto a dictadologia')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'text_to_dicta_loss_epoch_{epoch+1}.png')
    plt.show()
    plt.close()


# Guardar modelo
torch.save(model.state_dict(), 'text_to_dicta_model.pth')
print('Entrenamiento finalizado y modelo guardado.')

# ===================
# SEGUIMIENTO DEL ENTRENAMIENTO (puedes comentar o quitar este bloque)
# ===================
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evolución de la pérdida')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# ===================