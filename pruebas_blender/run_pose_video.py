# run_pose_video.py
import os, shutil, subprocess, sys
from pathlib import Path

# ========= CONFIGURACIÓN =========
# BLENDER_PATH sigue siendo absoluto por defecto (instalación de sistema).
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

# Resolver rutas relativas respecto al directorio del script
BASE_DIR = Path(__file__).resolve().parent

BLEND_FILE   = str(BASE_DIR / "cuerpo_humano_rigged.blend")
SCRIPT_PATH  = str(BASE_DIR / "text_to_video_from_poses.py")
POSES_JSON   = str(BASE_DIR / "poses_library.json")
CAMERA_JSON  = str(BASE_DIR / "camera_library.json")
ARMATURE     = "Human.rig"
POSES        = "B,A"           # o "B,A" o "A,B,C" etc.
FPS          = 24
HOLD         = 12
TRANSITION   = 12
ENGINE       = "EEVEE"         # o "CYCLES"
OUT_PATH     = str(BASE_DIR / "output" / "B_A.mp4")
WIDTH        = 1080
HEIGHT       = 1080
CAMERA_NAME  = "Cam_01"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"  # ajusta según donde extrajiste ffmpeg
SPEED = 2.5                 # 1.0 = sin cambio, >1 = más lento (2.0 -> 2x más lento), <1 = más rápido (0.5 -> 2x más rápido)
# =================================

LOG = "[PIPELINE]"

def log(msg): print(f"{LOG} {msg}")
def check_exists(path, desc):
    if not Path(path).exists():
        log(f"❌ No se encontró {desc}: {path}")
        sys.exit(1)

def _ffmpeg_chain_atempo_factors(target):
    # devuelve lista de factores [..] que multiplicados dan target (target = 1/speed)
    factors = []
    t = float(target)
    # limitar por el rango permitido de atempo (0.5..2.0) encadenando factores
    while t > 2.0 + 1e-9:
        factors.append(2.0)
        t /= 2.0
    while t < 0.5 - 1e-9:
        factors.append(0.5)
        t /= 0.5
    # añadir resto (puede estar en [0.5,2.0])
    factors.append(max(0.000001, t))
    return factors

def adjust_video_speed_with_ffmpeg(src_path, speed, ffmpeg_cmd=FFMPEG_PATH):
    """
    Ajusta la velocidad del video src_path multiplicando tiempos por `speed`.
    speed>1.0 -> video más lento (setpts=speed*PTS)
    speed<1.0 -> video más rápido
    Devuelve la path del fichero generado o lanza RuntimeError.
    """
    src = Path(src_path)
    if not src.exists():
        raise RuntimeError(f"Input no encontrado: {src}")

    # resolver ejecutable ffmpeg
    if ffmpeg_cmd == "ffmpeg":
        ffmpeg_exec = shutil.which("ffmpeg")
        if not ffmpeg_exec:
            raise RuntimeError("ffmpeg no encontrado en PATH. Instala ffmpeg o configura FFMPEG_PATH con la ruta completa al ejecutable (p.e. C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe).")
    else:
        ffmpeg_exec = str(Path(ffmpeg_cmd))
        if not Path(ffmpeg_exec).exists():
            raise RuntimeError(f"ffmpeg no encontrado en la ruta configurada: {ffmpeg_exec}")

    suffix = str(speed).replace(".", "p")
    out = src.with_name(f"{src.stem}_speed{suffix}{src.suffix}")

    # preparar filtros: video setpts, audio atempo (1/speed)
    setpts = f"{speed}*PTS"
    audio_target = 1.0 / float(speed)
    atempo_factors = _ffmpeg_chain_atempo_factors(audio_target)
    audio_filter = ",".join(f"atempo={f:.8f}" for f in atempo_factors)

    # construir comando filter_complex con mapeo
    if src.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
        fc = f"[0:v]setpts={setpts}[v];[0:a]{audio_filter}[a]"
        cmd = [
            ffmpeg_exec, "-y", "-i", str(src),
            "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(out)
        ]
    else:
        cmd = [ffmpeg_exec, "-y", "-i", str(src), "-vf", f"setpts={setpts}", "-c:v", "libx264", str(out)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falló (rc={proc.returncode}):\n{proc.stdout}")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg no produjo salida válida: {out}")
    return str(out)

def main():
    log("Iniciando pipeline automatizado para generación de vídeo...")
    check_exists(BLENDER_PATH, "Blender")
    check_exists(BLEND_FILE, ".blend")
    check_exists(SCRIPT_PATH, "script interno de Blender")
    check_exists(POSES_JSON, "poses_library.json")
    # camera json es opcional pero si existe lo usamos
    if Path(CAMERA_JSON).exists():
        log(f"Usando camera lib: {CAMERA_JSON}")
    else:
        log("Aviso: no se encontró camera_library.json (se usará la cámara de la escena).")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Comando que ejecutará Blender
    cmd = [
        BLENDER_PATH,
        "-b", BLEND_FILE,
        "--python", SCRIPT_PATH,
        "--",
        "--library", POSES_JSON,
        "--camera_lib", CAMERA_JSON,
        "--camera_name", CAMERA_NAME,
        "--armature", ARMATURE,
        "--fps", str(FPS),
        "--hold", str(HOLD),
        "--transition", str(TRANSITION),
        "--engine", ENGINE,
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--out", OUT_PATH,
        "--poses", POSES
    ]

    log("-----------------------------------------------------------")
    log("Ejecutando comando Blender:")
    log(" ".join(f'"{p}"' if " " in p else p for p in cmd))
    log("-----------------------------------------------------------")

    # ejecutar y retransmitir salida en tiempo real
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print(f"{LOG} Error lanzando Blender: {e}")
        sys.exit(1)

    # stream stdout
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip())

    proc.wait()
    rc = proc.returncode
    if rc != 0:
        print(f"{LOG} Blender salió con código {rc}. No se generó el vídeo.")
        sys.exit(rc)

    # comprobar que el archivo de salida existe y tiene tamaño
    out_path = Path(OUT_PATH)
    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"{LOG} Error: Blender terminó con código 0 pero no se encontró un fichero válido en: {OUT_PATH}")
        sys.exit(1)

    print(f"{LOG} ✅ Vídeo generado correctamente en:\n   {OUT_PATH}")

    # suponiendo que OUT_PATH es el fichero producido por Blender
    if SPEED != 1.0:
        try:
            # resolver ejecutable ffmpeg de forma segura
            ff_exec = None
            if FFMPEG_PATH == "ffmpeg":
                ff_exec = shutil.which("ffmpeg")
            else:
                p = Path(FFMPEG_PATH)
                ff_exec = str(p) if p.exists() else None

            if not ff_exec:
                log("⚠️ ffmpeg no encontrado en PATH ni en FFMPEG_PATH. Se omite ajuste de velocidad.")
            else:
                log(f"Ajustando velocidad con ffmpeg: {ff_exec}")
                adjusted = adjust_video_speed_with_ffmpeg(OUT_PATH, SPEED, ffmpeg_cmd=ff_exec)
                log(f"Archivo ajustado: {adjusted}")
        except Exception as e:
            log(f"Error ajustando velocidad con ffmpeg: {e}")
    else:
        log("SPEED=1.0 -> no se aplica ajuste de velocidad.")

    # DEBUG: mostrar PATH que ve este proceso y búsqueda de ffmpeg
    log(f"ENV PATH (truncated): {os.environ.get('PATH','')[:200]}...")
    which_ff = shutil.which("ffmpeg")
    log(f"shutil.which('ffmpeg') -> {which_ff}")

    # si quieres forzar ruta absoluta aquí, descomenta y ajusta:
    # FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

    # detección robusta
    if FFMPEG_PATH == "ffmpeg":
        ff_exec = which_ff
    else:
        ff_exec = FFMPEG_PATH if os.path.exists(FFMPEG_PATH) else None

    if not ff_exec:
        log("⚠️ ffmpeg no encontrado en PATH ni en FFMPEG_PATH. Si lo instalaste, asegúrate de reiniciar la terminal/VS Code o usa ruta absoluta en FFMPEG_PATH.")
    else:
        log(f"ffmpeg detectado en: {ff_exec}")

if __name__ == "__main__":
    main()
