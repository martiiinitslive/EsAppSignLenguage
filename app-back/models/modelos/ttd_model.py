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
INIT_MAP_SIZE = config_ttd.INIT_MAP_SIZE
INIT_CHANNELS = config_ttd.INIT_CHANNELS
DROPOUT_PROB = config_ttd.DROPOUT_PROB
LEAKY_RELU_SLOPE = config_ttd.LEAKY_RELU_SLOPE

# Definición de la clase del modelo generador
class TextToDictaModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE):
        super(TextToDictaModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.img_size = img_size
        self.init_map_size = INIT_MAP_SIZE
        self.init_channels = INIT_CHANNELS
        self.dropout_prob = DROPOUT_PROB
        self.leaky_relu_slope = LEAKY_RELU_SLOPE

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, self.init_channels * self.init_map_size * self.init_map_size),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(self.dropout_prob)
        )

        # Encoder (bloques descendentes)
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.init_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )

        # Decoder (bloques ascendentes con skip connections)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(320, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(544, 16, kernel_size=4, stride=2, padding=1),  # <-- Cambia 64 por 544
            nn.BatchNorm2d(16),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout2d(self.dropout_prob)
        )
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: tensor de índices de letras (batch,)
        # Paso 1: convertir el índice de la letra en su embedding
        emb = self.embedding(x)  # (batch, embedding_dim)
        if emb.dim() > 2:
            emb = emb.squeeze(1)
        out = self.fc(emb)
        out = out.view(-1, self.init_channels, self.init_map_size, self.init_map_size)

        # Encoder
        e1 = self.enc1(out)  # (batch, 256, 8, 8)
        e2 = self.enc2(e1)   # (batch, 128, 8, 8)

        # Decoder con skip connections
        d1 = self.dec1(e2)   # (batch, 128, 16, 16)
        e2_up = nn.functional.interpolate(e2, size=d1.shape[2:], mode='nearest')  # Upsample e2
        d1_cat = torch.cat([d1, e2_up], dim=1)  # (batch, 256, 16, 16)
        d2 = self.dec2(d1_cat)  # (batch, 64, 32, 32)
        e1_up = nn.functional.interpolate(e1, size=d2.shape[2:], mode='nearest')  # Upsample e1
        d2_cat = torch.cat([d2, e1_up], dim=1)  # (batch, 128, 32, 32)
        d3 = self.dec3(d2_cat)  # (batch, 32, 64, 64)
        out_up = nn.functional.interpolate(out, size=d3.shape[2:], mode='nearest')  # Upsample out
        d3_cat = torch.cat([d3, out_up], dim=1)  # (batch, 544, 64, 64)
        d4 = self.dec4(d3_cat)  # (batch, 16, 128, 128)
        imgs = self.final_conv(d4)
        imgs = self.tanh(imgs)
        return imgs

## Ejemplo de uso:
# model = TextToDictaModel(vocab_size=27, embedding_dim=32, img_size=128)  # Crea el modelo
# texto = torch.tensor([[1, 2, 3]])  # índices de letras

# -------------------------------------------------------------
# Resumen de la arquitectura:
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
