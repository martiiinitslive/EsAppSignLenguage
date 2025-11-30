# EsAppSingLenguageAI — Documentación técnica

Este documento describe en detalle la arquitectura, el pipeline de datos, los componentes clave y cómo ejecutar y probar el proyecto **EsAppSingLenguageAI**. Está pensado para desarrolladores que quieren entender o continuar el desarrollo.

Contenido:
- **Resumen del proyecto**
- **Arquitectura y flujo de datos (pipeline)**
- **Estructura del repositorio y descripción de archivos importantes**
- **Cómo ejecutar localmente (backend + frontend)**
- **Dependencias y requisitos del sistema**
- **Testing**
- **Limitaciones conocidas y próximos pasos**

---

**Resumen del proyecto**

EsAppSingLenguageAI es una aplicación para la generación de vídeos de interpretación en lengua de signos (dictadología) a partir de texto o de contenidos de vídeo/audio (subidos o descargados desde YouTube). El backend procesa y convierte audio→texto, normaliza texto, genera una secuencia de poses y renderiza un vídeo final que se sirve al frontend.

---

**Arquitectura y pipeline de datos**

Resumen del flujo de datos (alto nivel):

1. Entrada (frontend):
     - Texto manual (campo de texto).
     - Enlace de YouTube (URL).
     - Archivo de audio/video subido.

2. Backend API (FastAPI): endpoints principales:
     - `POST /generate_from_text/` — recibe JSON `{text}` → genera vídeo desde texto.
     - `POST /transcribe_youtube/` — recibe `{url}` → descarga, extrae audio, transcribe, genera vídeo.
     - `POST /procesar_video/` — recibe archivo (multipart form) → extrae audio, transcribe, genera vídeo.
     - `GET /download_video/{filename}` — sirve el MP4 generado.

3. Componentes internos (backend):
     - Audio extraction: `extract_audio_from_video(video_path, out_audio_path)` — usa `moviepy`.
     - Speech-to-text: `speech_to_text(audio_path)` — wrapper (actual implementación: `SpeechRecognition` / Google API or placeholder).
     - Text normalization: `normalize_text_for_renderer(text)` — limpieza y normalización para el renderer.
     - YouTube downloader: `download_youtube(url, out_dir)` — wrapper sobre `yt-dlp`.
     - Renderer: `app-back/mp/run_pose_to_video_mediapipe.py` — genera el vídeo final a partir de una secuencia de poses y un JSON base de poses.

4. Output: vídeo MP4 (`mp/output_mp/render_*.mp4`) y (opcionalmente) ficheros sidecar de subtítulos (`.ass`, `.srt`). El frontend consume la URL de descarga y reproduce el vídeo.

Flujo de datos detallado para cada entrada:

- A) Texto directo
    1. Frontend POST a `/generate_from_text/` con `{text}`.
    2. Backend normaliza el texto con `format_text_for_renderer.normalize_text_for_renderer()`.
    3. Backend llama a `run_pose_to_video_mediapipe.text_to_pose_sequence()` para obtener la secuencia de poses.
    4. Backend llama a `run_pose_to_video_mediapipe.render_sequence_from_json()` para renderizar vídeo (pasa `poses_mediapipe_video.json`).
    5. Devuelve `{video_path, download_url}` al cliente.

- B) Enlace de YouTube
    1. Frontend POST a `/transcribe_youtube/` con `{url}`.
    2. Backend usa `download_youtube()` (internamente llama `yt-dlp`) para obtener un MP4 temporal.
    3. Extrae audio con `extract_audio_from_video()`.
    4. Transcribe audio con `speech_to_text()`.
    5. Borra el vídeo descargado y el audio (se limpian temporalmente).
    6. Sigue los pasos A.2–A.4 para renderizar basado en el texto transcrito.

- C) Archivo subido (audio/video)
    1. Frontend envía multipart POST a `/procesar_video/` con `file`.
    2. Backend guarda temporalmente el archivo en `app-back/temp/`.
    3. Extrae audio con `extract_audio_from_video()` y transcribe con `speech_to_text()`.
    4. Borra el upload y audio temporal.
    5. Sigue los pasos A.2–A.4.

---

**Estructura del repositorio y descripción de archivos importantes**

Árbol (resumido) — paths más relevantes:

```
README.MD
requirements.txt
launcher.py
app-back/
    run_api.py
    main.py
    mp/
        run_pose_to_video_mediapipe.py
        poses_mediapipe_video.json
        output_mp/
    src/components/
        audio_extractor.py
        speech_to_text.py
        format_text_for_renderer.py
        downloader.py
app-front/
    package.json
    src/components/MainTranslator.js
test/
    conftest.py
    test_*.py
```

Descripción (archivo por archivo — sólo los más relevantes):

- `app-back/main.py`
    - Punto central del backend (FastAPI).
    - Declara rutas: `/generate_from_text/`, `/procesar_video/`, `/transcribe_youtube/`, `/download_video/{filename}` y `/`.
    - Contiene helper `_render_from_text(text)` que hace la importación dinámica del renderer `run_pose_to_video_mediapipe.py`, normaliza el texto y ejecuta `text_to_pose_sequence()` + `render_sequence_from_json()`.
    - Tiene lógica de limpieza de archivos generados (`cleanup_generated_outputs`) ejecutada en shutdown y utilizada por `launcher.py`.
    - Importante: la importación del renderer se hace dinámicamente con `importlib.util.spec_from_file_location()` (se añadió un fallback que soporta loaders personalizados usados en tests).

- `app-back/mp/run_pose_to_video_mediapipe.py`
    - Script de rendering que convierte una secuencia de poses a un vídeo final.
    - Funcionalidad principal: `text_to_pose_sequence()` (mapea texto a tokens/poses) y `render_sequence_from_json(json_path, seq, out_path, ...)` (dibuja manos/landmarks, compone frames, opcionalmente genera `.ass` y quema subtítulos con `ffmpeg`).
    - Alto uso de OpenCV (`cv2`) y NumPy; incluye funciones complejas de dibujo (wireframe y estilo “realistic”) y utilidades para localizar `ffmpeg`.
    - Parámetros configurables al inicio: `FRAMES_PER_POSE`, `SPEED_FACTOR`, `DRAW_SUBTITLES_INLINE`, parámetros de subtítulos, estilos de piel/colores, etc.
    - Producción: escribe en `app-back/mp/output_mp/` archivos `render_*.mp4` y opcionalmente `.ass/.srt`.

- `app-back/src/components/audio_extractor.py`
    - Usa `moviepy.video.io.VideoFileClip.VideoFileClip` para abrir el vídeo y escribir la pista de audio a disco (`audio.write_audiofile(output_audio_path)`).
    - Excepciones: si no hay pista de audio lanza `ValueError`.

- `app-back/src/components/speech_to_text.py`
    - Wrapper para realizar transcripción de audio a texto. (Implementación actual: `SpeechRecognition` / Google Web Speech — revisar el fichero para detalles de la integración y límites de cuota).

- `app-back/src/components/format_text_for_renderer.py`
    - Normaliza texto para que el renderer lo pueda procesar: elimina diacríticos, sustituye signos de apertura españoles, reduce caracteres no soportados a espacios, colapsa espacios múltiples.

- `app-back/src/components/downloader.py`
    - Wrapper ligero sobre `yt-dlp` (comprobar si `yt-dlp` está en PATH).
    - Descarga preferente: `bestaudio[ext=m4a]+bestvideo[ext=mp4]/best`, guarda en `out_dir` con template `%(id)s.%(ext)s`.

- `app-front/src/components/MainTranslator.js`
    - React component principal. Permite seleccionar input type (`text` / `file` / `link`), envía peticiones a la API y reproduce el `video` devuelto.
    - URL base del backend: `API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000'`.
    - Manejo robusto de respuestas (intenta parsear JSON y muestra errores legibles), y control de velocidad de reproducción.

- `launcher.py`
    - Script Windows-oriented para arrancar backend (`app-back/run_api.py`) y frontend (`npm start`) en consolas separadas.
    - Gestiona terminación (Ctrl-C) y limpia `render_*.mp4` y sidecars en `app-back/mp/output_mp/` al salir.

- `run_api.py`
    - Script que arranca `uvicorn` con la app (ajustado para `reload=False` en este repo para evitar issues con subprocess durante tests/desarrollo).

- `requirements.txt`
    - Suma las dependencias Python necesarias: `fastapi`, `uvicorn`, `moviepy`, `yt-dlp`, `SpeechRecognition`, `pytest`, `httpx` (para tests), `mediapipe`, `opencv-python`, `numpy`, etc.

- `test/` (pytest)
    - Tests unitarios que usan `TestClient` y fixtures en `conftest.py`. Las pruebas mockean el renderer y componentes pesados para hacer las pruebas rápidas y reproducibles.

---

**Cómo ejecutar localmente (dev)**

Requisitos preinstalados en el sistema:
- `ffmpeg` disponible en PATH (obligatorio para algunas features del renderer y para `moviepy` en producción).
- `yt-dlp` en PATH para la opción de descarga de YouTube (o instalar `yt-dlp` desde pip, pero el helper usa el binario en PATH).

Pasos (Windows, desde la raíz del repo):

1) Crear y activar virtualenv (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Ejecutar backend:

```powershell
cd app-back
python run_api.py
```

3) Ejecutar frontend (en otra consola):

```powershell
cd app-front
npm install
npm start
```

4) Alternativa: usar `launcher.py` desde la raíz para arrancar ambos (Windows):

```powershell
python launcher.py
```

Nota: `launcher.py` abre consolas separadas y `http://localhost:3000` en el navegador.

---

**Testing**

Tests unitarios con `pytest` (ya incluidos en `test/`). Recomendaciones:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

Detalles importantes del suite de tests:
- `test/conftest.py` ofrece fixtures que cargan dinámicamente `app-back/main.py` y mockean la importación del renderer para evitar render pesado.
- Las pruebas comprueban endpoints y helpers (`format_text_for_renderer`, downloader error path) sin depender de `ffmpeg` o `OpenCV` en el entorno de CI.

---

**Dependencias y requisitos del sistema**

- Python 3.10 (o 3.8+ recomendado)
- FFmpeg (en PATH) — requerido por `moviepy` y para quemar subtítulos si `DRAW_SUBTITLES_INLINE=False`.
- Node.js + npm — para el frontend.
- yt-dlp (en PATH) para descargar vídeos de YouTube.
- Recomendado: instalar paquetes de `requirements.txt` dentro de `.venv`.

---

**Limitaciones conocidas, notas y recomendaciones**

- Render pesado: `run_pose_to_video_mediapipe.py` usa OpenCV y puede requerir GPU o tiempo CPU intensivo. En producción es mejor offload a un worker (Celery/RQ) y retornar al cliente un job-id.
- Subtítulos: el script genera `.ass` y hace un paso con `ffmpeg` para quemar subtítulos — esto exige que `ffmpeg` esté instalado y disponible.
- Transcripción: `speech_to_text.py` usa `SpeechRecognition` (Google Web Speech) por defecto — esto tiene límites y latencia. Recomiendo evaluar `whisper` (local) o un servicio de pago para mayor fiabilidad.
- Seguridad: actualmente CORS está abierto (`allow_origins=['*']`) para desarrollo — restringir en producción.
- Manejo de archivos: el backend borra uploads y vídeos descargados en flujo normal, pero hay que vigilar concurrencia y race conditions si hay múltiples jobs simultáneos (usar directorios únicos por job).

---

**Siguientes pasos recomendados**

1. Añadir un worker (e.g., Celery/RQ) para ejecutar render en background y evitar bloqueo del servidor.
2. Añadir integración con Whisper o servicio de STT para mayor robustez.
3. Crear tests de integración que realmente lancen ffmpeg y el renderer en un entorno controlado (CI con imágenes que incluyen ffmpeg).
4. Mejorar la política de limpieza y uso de directorios por job (uuid temporales para evitar colisiones).
5. Documentar la API con ejemplos curl y añadir contrato OpenAPI más completo (FastAPI ya expone docs en `/docs`).

---

Si quieres, genero ahora un fichero `README.MD` más corto para `app-back/README.md` y otro para `app-front/README.md` con instrucciones específicas por componente, o puedo crear scripts de ayuda (Makefile / PowerShell) para automatizar provisioning y ejecución.

Si quieres que profundice en un fichero concreto (e.g., revisar `run_pose_to_video_mediapipe.py` por secciones y explicar cada función), dime cuál y lo hago a continuación.

---

Autor: Generated/updated by assistant — si detectas que algo importante falta en la documentación (por ejemplo, dependencias privadas o variables de entorno), dímelo y lo añado.
