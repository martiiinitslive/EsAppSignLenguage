# Dataset personalizado para dictadología
# Cada muestra es una imagen y su índice de letra
# Adaptado para cargar imágenes de dictadología, normalizadas para el modelo generativo.
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Dataset personalizado para dictadología
# Cada muestra es una imagen y su índice de letra
# Adaptado para cargar imágenes de dictadología, normalizadas para el modelo generativo.
class DictaDataset(Dataset):
    def __init__(self, data_dir, vocab, img_size):
        """
        Inicializa el dataset de dictadología.

        Args:
            data_dir (str): Ruta al directorio raíz del dataset. Debe contener subcarpetas por letra.
            vocab (str): Cadena con todas las letras presentes en el dataset.
            img_size (int): Tamaño al que se redimensionan las imágenes (alto y ancho).
        """
        self.data = []  # Lista de tuplas (índice de letra, ruta de imagen)
        self.vocab = vocab
        self.img_size = img_size

        # Transformaciones para las imágenes:
        # 1. Convierte a tensor (canal, alto, ancho)
        # 2. Normaliza a rango [-1, 1] para facilitar el aprendizaje del modelo
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # Redimensiona a img_size x img_size
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Recorre cada letra del vocabulario y añade las imágenes encontradas en su carpeta
        for idx, letter in enumerate(vocab):
            letter_dir = os.path.join(data_dir, letter)
            if not os.path.exists(letter_dir):
                continue  # Si la carpeta de la letra no existe, la salta
            for img_name in os.listdir(letter_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(letter_dir, img_name)
                    self.data.append((idx, img_path))  # Guarda el índice y la ruta de la imagen


    def __len__(self):
        """
        Devuelve el número total de muestras en el dataset.
        """
        return len(self.data)


    def __getitem__(self, idx):
        """
        Devuelve una muestra del dataset.

        Args:
            idx (int): Índice de la muestra

        Returns:
            label (torch.Tensor): Índice de la letra (tensor escalar)
            image (torch.Tensor): Imagen transformada (tensor normalizado)
        """
        label, img_path = self.data[idx]
        try:
            image = Image.open(img_path)
            image = self.transform(image)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar imagen: {img_path}. Error: {e}")
            # Si la imagen falla, devuelve un tensor de ceros con el tamaño adecuado
            image = torch.zeros(1, self.img_size, self.img_size)
        return torch.tensor(label), image
