# Configuración centralizada de hiperparámetros y parámetros globales
# Parámetros del modelo generador
IMG_SIZE = 128           # Tamaño (alto y ancho) de las imágenes procesadas por el modelo (ej: 128x128 píxeles)
EMBEDDING_DIM = 256      # Dimensión del vector de embedding para cada letra
VOCAB = 'ABCDEFGHIJKLMNÑOPQRSTUVWXYZ'  # Letras reconocidas por el modelo (vocabulario)
INIT_MAP_SIZE = 8        # Tamaño inicial del mapa de características
INIT_CHANNELS = 512      # Número de canales iniciales
DROPOUT_PROB = 0.2       # Probabilidad de dropout en el generador y discriminador
LEAKY_RELU_SLOPE = 0.2   # Parámetro 'negative_slope' para LeakyReLU. Controla cuánto deja pasar la activación para valores negativos (evita neuronas muertas).

# Parámetros de entrenamiento
BATCH_SIZE = 32          # Número de muestras procesadas en cada batch de entrenamiento
EPOCHS = 30              # Número de épocas
LAMBDA_PERCEPTUAL = 0.2  # Peso de la perceptual loss en la función de coste total
LR = 0.001               # Tasa de aprendizaje del optimizador
DATASET_DIR = None       # Ruta al directorio del dataset (se puede completar en el script principal)
NUM_EJEMPLOS = 1         # Número de ejemplos reales y generados a guardar por epoch
