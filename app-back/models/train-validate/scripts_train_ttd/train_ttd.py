# Script de entrenamiento para TextToDictaModel (ttd)
# Este script entrena un modelo generativo que convierte texto (letra) en una imagen de dictadología.
# Modularizado: importa dataset, modelo y pérdidas desde archivos separados.
# Incluye barra de progreso, guardado de ejemplos, gráficas de pérdida y comentarios explicativos.

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import models

# Importación absoluta de los módulos
import sys, os
modelos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'modelos'))
sys.path.append(modelos_path)
from ttd_model import TextToDictaModel, DictaDiscriminator
from dataset_ttd import DictaDataset
from losses_ttd import PerceptualLoss
from config_ttd import IMG_SIZE, EMBEDDING_DIM, VOCAB, BATCH_SIZE, EPOCHS, LAMBDA_PERCEPTUAL

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'dictadologia'))  # Ruta absoluta al dataset


if __name__ == '__main__':
    # Carga el dataset y lo divide en entrenamiento/validación (80/20)
    print("[INFO] Cargando dataset completo...")
    full_dataset = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE)
    print(f"[INFO] Total de muestras en el dataset: {len(full_dataset)}")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(f"[INFO] Dividiendo en {train_size} entrenamiento y {val_size} validación...")
    # Data augmentation solo en entrenamiento
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_dataset.dataset.transform = DictaDataset(DATASET_DIR, VOCAB, IMG_SIZE, augment=True).transform
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("[INFO] Dataset cargado y dividido correctamente.")

    # Instancia el modelo generador y el discriminador
    print("[INFO] Instanciando modelos...")
    model = TextToDictaModel(vocab_size=len(VOCAB), embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE)
    discriminator = DictaDiscriminator(img_size=IMG_SIZE)
    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion_adv = nn.BCELoss()  # Pérdida adversarial para GAN
    optimizer_G = optim.Adam(model.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    lambda_perceptual = LAMBDA_PERCEPTUAL
    print("[INFO] Modelos instanciados.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Iniciando entrenamiento GAN...")
    print(f"[INFO] Entrenando en dispositivo: {device}")
    model.to(device)
    discriminator.to(device)

    train_losses = []  # Pérdida en entrenamiento (generador)
    val_losses = []    # Pérdida en validación (generador)
    d_losses = []      # Pérdida del discriminador
    val_adv_losses = [] # Pérdida adversarial en validación

    # Medición de tiempo total
    start_time = time.time()

    # Configuración de carpetas para guardar resultados
    RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'train-ttd-model-epoch'))
    os.makedirs(RESULTS_PATH, exist_ok=True)
    EXAMPLES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'ejemplos-generados-ttd-model'))
    os.makedirs(EXAMPLES_PATH, exist_ok=True)
    # Carpeta para ejemplos reales del dataset
    REAL_EXAMPLES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'imagenes', 'ejemplos-reales-dataset'))
    os.makedirs(REAL_EXAMPLES_PATH, exist_ok=True)

    # Guardar ejemplos reales del dataset usando el primer batch del val_loader para que coincidan con los generados
    import torchvision.utils as vutils
    val_batch = next(iter(val_loader))
    val_labels, val_images = val_batch
    val_images = (val_images + 1) / 2  # Desnormaliza a [0, 1]
    for i in range(min(4, val_images.size(0))):
        img = val_images[i].cpu()
        idx_letra = val_labels[i].item()
        letra = VOCAB[idx_letra]
        from torchvision import transforms as T
        img_pil = T.ToPILImage()(img)
        img_name = f'ejemplo_real_{i+1}_{letra}.png'
        img_path = os.path.join(REAL_EXAMPLES_PATH, img_name)
        img_pil.save(img_path)
    print(f"Ejemplos reales del dataset guardados en: {REAL_EXAMPLES_PATH}")

    # Bucle principal de entrenamiento
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"[INFO] Comenzando epoch {epoch+1}/{EPOCHS}...")
        model.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        for i, (labels, images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Train")):
            labels, images = labels.to(device), images.to(device)
            batch_size = images.size(0)
            # --- Entrenamiento Discriminador ---
            optimizer_D.zero_grad()
            # Imágenes reales
            real_labels = torch.ones(batch_size, 1, device=device)
            output_real = discriminator(images)
            loss_real = criterion_adv(output_real, real_labels)
            # Imágenes generadas
            fake_images = model(labels).detach()
            fake_labels = torch.zeros(batch_size, 1, device=device)
            output_fake = discriminator(fake_images)
            loss_fake = criterion_adv(output_fake, fake_labels)
            # Pérdida total discriminador
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            running_loss_D += loss_D.item()

            # --- Entrenamiento Generador ---
            optimizer_G.zero_grad()
            gen_images = model(labels)
            output_gen = discriminator(gen_images)
            adv_loss = criterion_adv(output_gen, real_labels)
            loss_mse = criterion_mse(gen_images, images)
            loss_perceptual = criterion_perceptual(gen_images, images)
            # Pérdida total generador: adversarial + mse + perceptual
            loss_G = adv_loss + loss_mse + lambda_perceptual * loss_perceptual
            loss_G.backward()
            optimizer_G.step()
            running_loss_G += loss_G.item()

        avg_train_loss = running_loss_G / len(train_loader)
        avg_d_loss = running_loss_D / len(train_loader)
        train_losses.append(avg_train_loss)
        d_losses.append(avg_d_loss)
        epoch_end = time.time()
        total_time_so_far = epoch_end - start_time
        total_hours = int(total_time_so_far // 3600)
        total_minutes = int((total_time_so_far % 3600) // 60)
        print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Tiempo total: {total_hours}:{total_minutes:02d}')

        # Validación por batch
        model.eval()
        discriminator.eval()
        val_loss = 0.0
        val_adv_loss = 0.0
        with torch.no_grad():
            for labels, images in tqdm(val_loader, desc=f"Epoch {epoch+1} - Val"):
                labels, images = labels.to(device), images.to(device)
                outputs = model(labels)
                loss_mse = criterion_mse(outputs, images)
                loss_perceptual = criterion_perceptual(outputs, images)
                loss = loss_mse + lambda_perceptual * loss_perceptual
                val_loss += loss.item()
                # Pérdida adversarial en validación
                real_labels = torch.ones(images.size(0), 1, device=device)
                adv_loss = criterion_adv(discriminator(outputs), real_labels)
                val_adv_loss += adv_loss.item()
        avg_val_loss = val_loss/len(val_loader) if len(val_loader) > 0 else 0
        avg_val_adv_loss = val_adv_loss/len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        val_adv_losses.append(avg_val_adv_loss)
        print(f'[INFO] Epoch {epoch+1}/{EPOCHS}, Val Loss: {avg_val_loss:.4f}, Val Adv Loss: {avg_val_adv_loss:.4f}')

        # Guardar y mostrar la gráfica de seguimiento en cada epoch
        plt.figure(figsize=(10,6))
        plt.plot(train_losses, label='Gen Train Loss')
        plt.plot(val_losses, label='Gen Val Loss')
        plt.plot(val_adv_losses, label='Gen Val Adv Loss')
        plt.plot(d_losses, label='Disc Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evolución de las pérdidas (Generador/Discriminador)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        save_path = os.path.join(RESULTS_PATH, f'text_to_dicta_loss_epoch_{epoch+1}.png')
        plt.savefig(save_path)
        print(f"[INFO] Gráfica guardada en: {save_path}")
        plt.close()

        # Guardar ejemplos generados por el modelo en cada epoch
        with torch.no_grad():
            # Usar el mismo batch de validación para los ejemplos generados
            val_labels_batch = val_labels.to(device)
            outputs = model(val_labels_batch)
            outputs = (outputs + 1) / 2
            for i in range(min(4, outputs.size(0))):
                img = outputs[i].cpu()
                img = img.squeeze(0) if img.dim() == 3 and img.size(0) == 1 else img
                from torchvision import transforms as T
                img_pil = T.ToPILImage()(img)
                idx_letra = val_labels[i].item()
                letra = VOCAB[idx_letra]
                img_name = f'epoch_{epoch+1}_example_{i+1}_{letra}.png'
                img_path = os.path.join(EXAMPLES_PATH, img_name)
                img_pil.save(img_path)
            print(f"[INFO] Ejemplos generados guardados en: {EXAMPLES_PATH}")

    # Guardar el modelo entrenado al finalizar
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelEntrenado', 'ttd_model_trained.pth'))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    end_time = time.time()
    total_time = end_time - start_time
    total_hours = int(total_time // 3600)
    total_minutes = int((total_time % 3600) // 60)
    total_seconds = int(total_time % 60)
    #los segundos no se muestran
    print(f'[INFO] Entrenamiento finalizado y modelo guardado en: {MODEL_PATH}')
    print(f'[INFO] Tiempo total de entrenamiento: {total_hours}:{total_minutes:02d}:{total_seconds:02d}')
