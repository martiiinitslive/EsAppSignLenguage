"""
Modelo generador: convierte texto en imágenes de dictadología.
"""

# Importa PyTorch y su módulo de redes neuronales
import torch
import torch.nn as nn

# Definición de la clase del modelo generador
class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, img_size):
        # Constructor de la clase. Recibe el tamaño del vocabulario, la dimensión del embedding y el tamaño de la imagen.
        super(TextToDictaModel, self).__init__()
        # Capa de embedding: convierte índices de letras en vectores de tamaño embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Capa lineal (fully connected) que transforma el embedding en un vector plano del tamaño de la imagen
        self.fc = nn.Linear(embedding_dim, img_size * img_size)
        # Guarda el tamaño de la imagen para usarlo después
        self.img_size = img_size

    def forward(self, x):
        # x: tensor de índices de letras (batch, seq_len)
        # Convierte los índices de letras en vectores de embedding
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        # Pasa los embeddings por la capa lineal para obtener un vector plano que representa la imagen
        imgs = self.fc(emb)  # (batch, seq_len, img_size*img_size)
        # Reestructura el vector plano en una imagen 2D (por ejemplo, de 4096 a 64x64)
        imgs = imgs.view(-1, self.img_size, self.img_size)  # (batch*seq_len, img_size, img_size)
        # Devuelve las imágenes generadas
        return imgs

# Ejemplo de uso:
# model = TextToDictaModel(vocab_size=27, embedding_dim=32, img_size=64)  # Crea el modelo
# texto = torch.tensor([[1, 2, 3]])  # índices de letras
# imgs = model(texto)  # Obtiene las imágenes generadas
