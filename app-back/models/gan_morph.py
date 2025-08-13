"""
GAN para morphing entre dos posiciones de la mano (letras).
Este esqueleto define la estructura básica de una GAN para generar imágenes intermedias entre dos letras.
"""

import torch
import torch.nn as nn

# Generador: recibe imagen inicial, imagen final y un parámetro t (0 a 1) para generar la imagen intermedia
def make_generator(img_size):
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(img_size*img_size*2 + 1, 512),
                nn.ReLU(),
                nn.Linear(512, img_size*img_size),
                nn.Tanh()
            )
        def forward(self, img_start, img_end, t):
            # img_start, img_end: (batch, img_size, img_size)
            # t: (batch, 1)
            x = torch.cat([
                img_start.view(img_start.size(0), -1),
                img_end.view(img_end.size(0), -1),
                t
            ], dim=1)
            out = self.fc(x)
            out = out.view(-1, img_size, img_size)
            return out
    return Generator()

# Discriminador: recibe imagen intermedia y las imágenes de inicio/fin, predice si la transición es real o generada
def make_discriminator(img_size):
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(img_size*img_size*3, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        def forward(self, img_start, img_end, img_mid):
            x = torch.cat([
                img_start.view(img_start.size(0), -1),
                img_end.view(img_end.size(0), -1),
                img_mid.view(img_mid.size(0), -1)
            ], dim=1)
            out = self.fc(x)
            return out
    return Discriminator()

# Ejemplo de uso:
# gen = make_generator(img_size=64)
# disc = make_discriminator(img_size=64)
