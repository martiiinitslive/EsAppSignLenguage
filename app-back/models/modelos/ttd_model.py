"""
Modelo generador: convierte texto en imágenes de dictadología.
"""

# Importa PyTorch y su módulo de redes neuronales
import torch
import torch.nn as nn
import sys, os

import importlib.util

# Define config_path as the directory containing this file
config_path = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train-validate', 'scripts_train_ttd'))
config_ttd_path = os.path.join(config_path, 'config_ttd.py')
spec = importlib.util.spec_from_file_location("config_ttd", config_ttd_path)
config_ttd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_ttd)
EMBEDDING_DIM = config_ttd.EMBEDDING_DIM
IMG_SIZE = config_ttd.IMG_SIZE
DROP_PROB = getattr(config_ttd, 'DROPOUT_PROB', 0.2)

# Definición de la clase del modelo generador
# Discriminador: distingue entre imágenes reales y generadas
class DictaDiscriminator(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super(DictaDiscriminator, self).__init__()
        self.img_size = img_size
        from torch.nn.utils import spectral_norm
        self.model = nn.Sequential(
            # Entrada: (batch, 3, 128, 128)
            spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)),  # (batch, 32, 64, 64)
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_PROB),

            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),  # (batch, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_PROB),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # (batch, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_PROB),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # (batch, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_PROB),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()  # Salida: probabilidad de ser real
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

# Modelo generador: convierte texto (índice de letra) en imágenes RGB de dictadología
class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE):
        super(TextToDictaModel, self).__init__()

        # Capa de embedding: convierte el índice de la letra en un vector denso
        # Permite que el modelo aprenda una representación numérica para cada letra
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.img_size = img_size
        self.init_map_size = 8  # Tamaño inicial del mapa de características (8x8)
        self.init_channels = 512  # Número de canales iniciales para el mapa de características

        # Capa totalmente conectada (fully connected):
        # Proyecta el embedding a un vector largo y lo reestructura a un mapa de características inicial
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, self.init_channels * self.init_map_size * self.init_map_size),
            nn.ReLU()  # Activación no lineal
        )

        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.init_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 8x8x512 → 16x16x256
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 16x16x256 → 32x32x128
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 32x32x128 → 64x64x64
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 64x64x64 → 128x128x32

        # Decoder (upsampling) con skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 128x128x32 → 64x64x64
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 64x64x128 → 32x32x128
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128*2, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 32x32x256 → 16x16x256
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256*2, self.init_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(),
            nn.Dropout(DROP_PROB),
        ) # 16x16x512 → 8x8x512

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.init_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Inicialización de pesos recomendada para mejorar el aprendizaje
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Inicializa los pesos de las capas lineales y convolucionales con Xavier
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: tensor de índices de letras (batch,)
        emb = self.embedding(x)
        if emb.dim() > 2:
            emb = emb.squeeze(1)
        out = self.fc(emb)
        out = out.view(-1, self.init_channels, self.init_map_size, self.init_map_size)

        # Encoder
        e1 = self.enc1(out)  # 16x16x256
        e2 = self.enc2(e1)   # 32x32x128
        e3 = self.enc3(e2)   # 64x64x64
        e4 = self.enc4(e3)   # 128x128x32

        # Decoder con skip connections
        d1 = self.dec1(e4)   # 64x64x64
        d1 = torch.cat([d1, e3], dim=1)  # Skip connection
        d2 = self.dec2(d1)   # 32x32x128
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)   # 16x16x256
        d3 = torch.cat([d3, e1], dim=1)
        d4 = self.dec4(d3)   # 8x8x512

        imgs = self.final_conv(d4)
        return imgs

## Ejemplo de uso:
# model = TextToDictaModel(vocab_size=27, embedding_dim=32, img_size=128)  # Crea el modelo
# texto = torch.tensor([[1, 2, 3]])  # índices de letras

# -------------------------------------------------------------
# Resumen de la arquitectura:
#   Es una red neuronal generativa convolucional (DCGAN-like generator)
#   esta arquitectura se conoce como "Deep Convolutional Generator"
#
# 1. Embedding Layer
#    - Convierte el índice de la letra (por ejemplo, 'd') en un vector denso de tamaño embedding_dim (por defecto 128).
#    - Permite que el modelo aprenda una representación numérica para cada letra.
#
# 2. Proyección Inicial (Fully Connected)
#    - Una capa lineal (nn.Linear) transforma el embedding en un vector largo.
#    - Este vector se reestructura a un mapa de características inicial de tamaño 8x8x256 (8x8 espacial, 256 canales).
#    - Se aplica una activación ReLU.
#
# 3. Bloques Deconvolucionales (ConvTranspose2d)
#    - Cuatro bloques de transposed convolutions expanden el mapa de características:
#      - 8x8x256 → 16x16x128 (ConvTranspose2d + BatchNorm + ReLU)
#      - 16x16x128 → 32x32x64 (ConvTranspose2d + BatchNorm + ReLU)
#      - 32x32x64 → 64x64x32 (ConvTranspose2d + BatchNorm + ReLU)
#      - 64x64x32 → 128x128x16 (ConvTranspose2d + BatchNorm + ReLU)
#    - Cada bloque aumenta el tamaño espacial y reduce el número de canales.
#
# 4. Capa Final
#    - Una convolución normal (nn.Conv2d) reduce los canales a 3 (imagen RGB), manteniendo el tamaño 128x128.
#    - Se aplica una activación Tanh para limitar la salida al rango [-1, 1], compatible con la normalización de tus datos.
#
# Resumen del flujo:
# Índice de letra → Embedding → Mapa de características inicial → Expansión espacial con deconvoluciones → Imagen final 128x128x3 normalizada.
# -------------------------------------------------------------
#
# Diagrama del flujo de datos:
#
# [Índice de letra]
#       │
#       ▼
# [Embedding Layer]
#       │
#       ▼
# [Linear (FC) → ReLU]
#       │
#       ▼
# [Reshape a mapa de características inicial: 8x8x256]
#       │
#       ▼
# [ConvTranspose2d: 8x8x256 → 16x16x128] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 16x16x128 → 32x32x64] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 32x32x64 → 64x64x32] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [ConvTranspose2d: 64x64x32 → 128x128x16] → [BatchNorm] → [ReLU]
#       │
#       ▼
# [Conv2d: 128x128x16 → 128x128x3] → [Tanh]
#       │
#       ▼
# [Imagen generada 128x128x3 (normalizada en [-1, 1])]
