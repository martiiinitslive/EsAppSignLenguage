# Guarda la configuración (ubicación + enfoque) de la cámara activa en un JSON junto al .blend.
#para ponerlo en script de blender y ejecutarlo
# guarda la pose actual del armature activo en una biblioteca JSON
import bpy, json, os

# ========== CONFIG ==========
CAM_NAME = "Cam_01"                             # nombre bajo el que se guarda esta cámara en la biblioteca

# Detección robusta de la carpeta donde está el script / fichero abierto en el Editor de texto de Blender.
def _detect_script_dir():
    # 1) si __file__ existe (ejecución desde fichero .py)
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        pass
    # 2) si el script está abierto en el Text Editor y tiene filepath (abrir desde disco)
    try:
        for txt in bpy.data.texts:
            fp = getattr(txt, "filepath", "")
            if fp:
                # bpy.path.abspath convierte rutas relativas tipo "//" a absolutas
                return os.path.dirname(bpy.path.abspath(fp))
    except Exception:
        pass
    # 3) usar la carpeta del .blend (si el .blend está guardado)
    try:
        blend_dir = bpy.path.abspath("//")
        if blend_dir:
            return blend_dir
    except Exception:
        pass
    # 4) fallback al cwd del proceso
    return os.getcwd()

SCRIPT_DIR = _detect_script_dir()
LIB_PATH = os.path.join(SCRIPT_DIR, "camera_library.json")

# Mostrar ruta usada para depuración (se verá en la consola de Blender)
print(f"[DEBUG] camera_library: SCRIPT_DIR = {SCRIPT_DIR}")
print(f"[DEBUG] camera_library: LIB_PATH    = {os.path.abspath(LIB_PATH)}")
# ============================

def find_camera():
    # Preferir la cámara activa de la escena
    scn = bpy.context.scene if hasattr(bpy.context, "scene") else None
    if scn and getattr(scn, "camera", None):
        return scn.camera
    # fallback: objeto activo/seleccionado
    obj = bpy.context.object
    if obj and obj.type == "CAMERA":
        return obj
    for o in bpy.context.selected_objects:
        if o.type == "CAMERA":
            return o
    # última opción: primera cámara en la escena
    for o in bpy.data.objects:
        if o.type == "CAMERA":
            return o
    return None

def cam_transform(cam_obj):
    # Usar matriz mundial (world space) para location y rotación robusta
    mw = cam_obj.matrix_world
    loc = mw.to_translation()
    try:
        eul = mw.to_euler(cam_obj.rotation_mode)
        rot = {"rotation_mode": cam_obj.rotation_mode, "rotation_euler": [float(eul.x), float(eul.y), float(eul.z)]}
    except Exception:
        q = mw.to_quaternion()
        rot = {"rotation_mode": "QUATERNION", "rotation_quaternion": [float(q.w), float(q.x), float(q.y), float(q.z)]}
    return {"location": [float(loc.x), float(loc.y), float(loc.z)], **rot}

def cam_dof(cam_obj):
    cam_data = cam_obj.data
    d = {}
    d["lens_mm"] = float(getattr(cam_data, "lens", 0.0))
    d["sensor_width"] = float(getattr(cam_data, "sensor_width", 0.0))
    d["sensor_height"] = float(getattr(cam_data, "sensor_height", 0.0))
    d["sensor_fit"] = str(getattr(cam_data, "sensor_fit", "AUTO"))
    d["shift_x"] = float(getattr(cam_data, "shift_x", 0.0))
    d["shift_y"] = float(getattr(cam_data, "shift_y", 0.0))
    # DOF: intentar focus_object directo
    foc_obj = None
    try:
        foc_obj = getattr(cam_data.dof, "focus_object", None)
    except Exception:
        foc_obj = None
    d["dof_focus_object"] = foc_obj.name if foc_obj is not None else None
    # fallback: buscar target en constraints (Track To, Damped Track, Locked Track, ChildOf etc.)
    if not d["dof_focus_object"]:
        try:
            for c in cam_obj.constraints:
                tgt = getattr(c, "target", None)
                if tgt is not None:
                    d["dof_focus_object"] = tgt.name
                    break
        except Exception:
            pass
    d["dof_focus_distance"] = float(getattr(cam_data.dof, "focus_distance", 0.0))
    # guardar apertura si existe
    try:
        d["aperture_fstop"] = float(getattr(cam_data.dof, "aperture_fstop", 0.0))
    except Exception:
        d["aperture_fstop"] = None
    return d

def load_lib(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {"created_by": "camera_library.py"}, "cameras": {}}

def save_lib(path, lib):
    dirp = os.path.dirname(path) or os.getcwd()
    try:
        os.makedirs(dirp, exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lib, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DEBUG] save_lib: error escribiendo -> {e}")
        raise
    print(f"[DEBUG] save_lib: archivo escrito -> {os.path.abspath(path)} (exists={os.path.exists(path)})")

def main():
    try:
        cam = find_camera()
        if not cam:
            print("[DEBUG] No camera encontrada en escena.")
            return
        lib = load_lib(LIB_PATH)
        if not isinstance(lib, dict):
            lib = {}
        cams = lib.get("cameras", {})
        cams[CAM_NAME] = {
            "object_name": cam.name,
            "transform": cam_transform(cam),
            "data": cam_dof(cam),
            "scene_camera": bpy.context.scene.camera.name if getattr(bpy.context.scene, "camera", None) else None,
            "frame": int(bpy.context.scene.frame_current) if getattr(bpy.context, "scene", None) else None
        }
        lib["cameras"] = cams
        save_lib(LIB_PATH, lib)
        print(f"[DEBUG] Cámara guardada en {os.path.abspath(LIB_PATH)} -> objeto: {cam.name}")
    except Exception as e:
        print(f"[DEBUG] main error: {e}")
        raise

if __name__ == "__main__":
    main()